"""
radial_diffusion.py

Módulo numérico genérico para la resolución de la ecuación de difusión radial 1D
en coordenadas cilíndricas:

    ∂u/∂t = (1/r) ∂/∂r [ r D ∂u/∂r ]

Diseñado como infraestructura matemática para modelos físicos de combustible
(e.g., Nelson 2000, NFDRS-2016), completamente desacoplado de la física forestal
y del motor de propagación de incendios.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np


class RadialDiffusionError(ValueError):
    """Excepción base para errores del solver de difusión radial."""
    pass


def solve_tridiagonal(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray
) -> np.ndarray:
    """
    Resuelve el sistema tridiagonal A * x = d utilizando el Algoritmo de Thomas (TDMA).
    
    Parámetros:
    - a: Subdiagonal inferior (longitud N, a[0] = 0.0 no utilizado).
    - b: Diagonal principal (longitud N).
    - c: Superdiagonal superior (longitud N, c[N-1] = 0.0 no utilizado).
    - d: Vector de términos independientes (longitud N).
    
    Retorna:
    - x: Vector solución (longitud N).
    """
    n = len(d)
    c_prime = np.zeros(n, dtype=np.float64)
    d_prime = np.zeros(n, dtype=np.float64)
    x = np.zeros(n, dtype=np.float64)

    # Barrido hacia adelante (Forward sweep)
    b0 = b[0]
    if np.isclose(b0, 0.0):
        raise RadialDiffusionError("Pivote nulo en diagonal principal en j=0.")
    c_prime[0] = c[0] / b0
    d_prime[0] = d[0] / b0

    for j in range(1, n):
        denom = b[j] - a[j] * c_prime[j - 1]
        if np.isclose(denom, 0.0):
            raise RadialDiffusionError(f"Pivote nulo en diagonal principal en j={j}.")
        if j < n - 1:
            c_prime[j] = c[j] / denom
        d_prime[j] = (d[j] - a[j] * d_prime[j - 1]) / denom

    # Sustitución hacia atrás (Back substitution)
    x[n - 1] = d_prime[n - 1]
    for j in range(n - 2, -1, -1):
        x[j] = d_prime[j] - c_prime[j] * x[j + 1]

    return x


class RadialDiffusionConfig:
    """
    Configuración geométrica y numérica para la malla radial 1D.
    """
    def __init__(
        self,
        radius_m: float,
        n_radial_nodes: int = 11,
        diffusivity_m2_s: float = 1.0e-7,
        theta: float = 1.0
    ):
        """
        Parámetros:
        - radius_m: Radio exterior del cilindro en metros (R > 0).
        - n_radial_nodes: Número de nodos radiales discretos (N >= 3).
        - diffusivity_m2_s: Coeficiente de difusión D en m²/s (D > 0).
        - theta: Ponderación temporal del método theta:
                 theta = 0.0 -> Forward Euler (Explícito)
                 theta = 0.5 -> Crank-Nicolson (Segundo orden)
                 theta = 1.0 -> Backward Euler (Implícito de primer orden, por defecto)
        """
        if radius_m is None or np.isnan(radius_m) or np.isinf(radius_m) or radius_m <= 0.0:
            raise RadialDiffusionError(f"radius_m debe ser finito y estrictamente positivo: {radius_m}")
        if n_radial_nodes is None or n_radial_nodes < 3:
            raise RadialDiffusionError(f"n_radial_nodes debe ser entero >= 3: {n_radial_nodes}")
        if diffusivity_m2_s is None or np.isnan(diffusivity_m2_s) or np.isinf(diffusivity_m2_s) or diffusivity_m2_s <= 0.0:
            raise RadialDiffusionError(f"diffusivity_m2_s debe ser finito y estrictamente positivo: {diffusivity_m2_s}")
        if theta is None or np.isnan(theta) or not (0.0 <= theta <= 1.0):
            raise RadialDiffusionError(f"theta debe estar en [0.0, 1.0]: {theta}")

        self.radius_m = float(radius_m)
        self.n_radial_nodes = int(n_radial_nodes)
        self.diffusivity_m2_s = float(diffusivity_m2_s)
        self.theta = float(theta)

        # Malla radial uniforme r_j = j * dr, j in {0, ..., N-1}
        self.dr = self.radius_m / (self.n_radial_nodes - 1)
        self.r_nodes = np.linspace(0.0, self.radius_m, self.n_radial_nodes, dtype=np.float64)

        # Pesos de integración volumétrica exacta para cilindros
        # V_0 = pi * (dr/2)^2 = pi * dr^2 / 4
        # V_j = 2 * pi * r_j * dr = 2 * pi * j * dr^2 (para j = 1..N-2)
        # V_{N-1} = pi * (R^2 - (R - dr/2)^2) = pi * dr^2 * (N - 5/4)
        # Sum(V_j) = pi * R^2
        v_weights = np.zeros(self.n_radial_nodes, dtype=np.float64)
        v_weights[0] = 0.25
        for j in range(1, self.n_radial_nodes - 1):
            v_weights[j] = 2.0 * j
        v_weights[-1] = (self.n_radial_nodes - 1) - 0.25
        total_v = (self.n_radial_nodes - 1) ** 2
        self.volumetric_weights = v_weights / total_v


class RadialDiffusionSolver:
    """
    Solver numérico de diferencias finitas / volúmenes finitos para la ecuación
    de difusión radial 1D en coordenadas cilíndricas.
    """
    def __init__(self, config: RadialDiffusionConfig):
        self.config = config
        self.dr = config.dr
        self.r_nodes = config.r_nodes
        self.n_nodes = config.n_radial_nodes
        self.D = config.diffusivity_m2_s
        self.theta = config.theta
        self.volumetric_weights = config.volumetric_weights

    def compute_spatial_operator(self, u: np.ndarray, bc_type: str, bc_val: float) -> np.ndarray:
        """
        Aplica el operador Laplaciano radial discreto L(u) = (1/r) d/dr [ r D du/dr ].
        """
        lu = np.zeros(self.n_nodes, dtype=np.float64)
        dr2 = self.dr ** 2

        # 1. Centro r = 0 (L'Hopital / Balance de celda central: 4 D (u_1 - u_0) / dr^2)
        lu[0] = 4.0 * self.D * (u[1] - u[0]) / dr2

        # 2. Nodos interiores j = 1, ..., N-2
        for j in range(1, self.n_nodes - 1):
            factor_minus = 1.0 - 1.0 / (2.0 * j)
            factor_plus = 1.0 + 1.0 / (2.0 * j)
            lu[j] = (self.D / dr2) * (factor_minus * u[j - 1] - 2.0 * u[j] + factor_plus * u[j + 1])

        # 3. Superficie exterior r = R (nodo N-1)
        if bc_type.lower() == "dirichlet":
            # Para Dirichlet, el valor está fijado en la superficie
            lu[-1] = 0.0
        elif bc_type.lower() == "neumann":
            # du/dr = 0 en r = R
            j_last = self.n_nodes - 1
            factor = (1.0 - 0.5 / j_last) / (1.0 - 0.25 / j_last)
            lu[-1] = (2.0 * self.D / dr2) * factor * (u[-2] - u[-1])
        else:
            raise RadialDiffusionError(f"Tipo de condición de borde desconocida: {bc_type}")

        return lu

    def step(
        self,
        current_profile: np.ndarray,
        dt_seconds: float,
        bc_type: str = "dirichlet",
        bc_val: float = 0.0
    ) -> np.ndarray:
        """
        Avanza el perfil radial u(r) un paso temporal dt_seconds.
        
        Parámetros:
        - current_profile: Array de longitud n_radial_nodes con los valores u(r) al tiempo t.
        - dt_seconds: Paso temporal dt en segundos (dt >= 0).
        - bc_type: 'dirichlet' (u(R) = bc_val) o 'neumann' (du/dr|_{r=R} = 0).
        - bc_val: Valor escalar de la condición de borde en la superficie.
        
        Retorna:
        - next_profile: Array de longitud n_radial_nodes con los valores u(r) al tiempo t + dt.
        """
        if dt_seconds is None or np.isnan(dt_seconds) or np.isinf(dt_seconds) or dt_seconds < 0.0:
            raise RadialDiffusionError(f"dt_seconds debe ser finito y no negativo: {dt_seconds}")
        
        if dt_seconds == 0.0:
            return current_profile.copy()

        u = np.asarray(current_profile, dtype=np.float64)
        if len(u) != self.n_nodes:
            raise RadialDiffusionError(f"Dimensión de perfil incompatible: esperado {self.n_nodes}, recibido {len(u)}")
        if np.isnan(u).any() or np.isinf(u).any():
            raise RadialDiffusionError("current_profile contiene valores NaN o Inf.")

        bc_type_clean = bc_type.strip().lower()
        if bc_type_clean not in ("dirichlet", "neumann"):
            raise RadialDiffusionError(f"bc_type no soportado: '{bc_type}'. Usar 'dirichlet' o 'neumann'.")

        dr2 = self.dr ** 2
        theta = self.theta

        # =====================================================================
        # CASO 1: FORWARD EULER (theta == 0.0)
        # =====================================================================
        if theta == 0.0:
            # Comprobar estabilidad CFL: dt <= dr^2 / (4 * D)
            cfl_limit = dr2 / (4.0 * self.D)
            if dt_seconds > cfl_limit:
                raise RadialDiffusionError(
                    f"Inestabilidad numérica Forward Euler: dt={dt_seconds:.4e} s excede "
                    f"límite CFL={cfl_limit:.4e} s (dr={self.dr:.4e} m, D={self.D:.4e} m²/s)."
                )
            lu = self.compute_spatial_operator(u, bc_type_clean, bc_val)
            u_next = u + dt_seconds * lu
            if bc_type_clean == "dirichlet":
                u_next[-1] = float(bc_val)
            return u_next

        # =====================================================================
        # CASO 2: THETA-METHOD IMPLÍCITO (0 < theta <= 1.0, e.g. Crank-Nicolson o Backward Euler)
        # Construcción de sistema tridiagonal: A * u^{n+1} = d
        # =====================================================================
        a_diag = np.zeros(self.n_nodes, dtype=np.float64)  # subdiagonal
        b_diag = np.zeros(self.n_nodes, dtype=np.float64)  # diagonal principal
        c_diag = np.zeros(self.n_nodes, dtype=np.float64)  # superdiagonal
        rhs = np.zeros(self.n_nodes, dtype=np.float64)     # lado derecho

        alpha = theta * dt_seconds * self.D / dr2
        beta = (1.0 - theta) * dt_seconds * self.D / dr2

        # 1. Nodo r = 0 (centro cilíndrico)
        # (1 + 4*alpha) * u_0^{n+1} - 4*alpha * u_1^{n+1} = (1 - 4*beta) * u_0^n + 4*beta * u_1^n
        b_diag[0] = 1.0 + 4.0 * alpha
        c_diag[0] = -4.0 * alpha
        rhs[0] = (1.0 - 4.0 * beta) * u[0] + 4.0 * beta * u[1]

        # 2. Nodos interiores j = 1, ..., N-2
        for j in range(1, self.n_nodes - 1):
            fm = 1.0 - 1.0 / (2.0 * j)
            fp = 1.0 + 1.0 / (2.0 * j)

            a_diag[j] = -alpha * fm
            b_diag[j] = 1.0 + 2.0 * alpha
            c_diag[j] = -alpha * fp

            rhs[j] = u[j] + beta * (fm * u[j - 1] - 2.0 * u[j] + fp * u[j + 1])

        # 3. Superficie exterior r = R (nodo N-1)
        if bc_type_clean == "dirichlet":
            # u_{N-1}^{n+1} = bc_val
            a_diag[-1] = 0.0
            b_diag[-1] = 1.0
            c_diag[-1] = 0.0
            rhs[-1] = float(bc_val)
        else:  # neumann: du/dr = 0 en r = R
            j_last = self.n_nodes - 1
            factor = (1.0 - 0.5 / j_last) / (1.0 - 0.25 / j_last)
            gamma_imp = 2.0 * alpha * factor
            gamma_exp = 2.0 * beta * factor

            a_diag[-1] = -gamma_imp
            b_diag[-1] = 1.0 + gamma_imp
            c_diag[-1] = 0.0
            rhs[-1] = u[-1] + gamma_exp * (u[-2] - u[-1])

        # Resolver con TDMA
        u_next = solve_tridiagonal(a_diag, b_diag, c_diag, rhs)
        return u_next

    def compute_average(self, profile: np.ndarray) -> float:
        """
        Calcula el valor medio ponderado por volumen en el cilindro:
            u_bar = (2 / R^2) * int_0^R r u(r) dr
        utilizando la integración exacta por capas concéntricas.
        """
        u = np.asarray(profile, dtype=np.float64)
        if len(u) != self.n_nodes:
            raise RadialDiffusionError(f"Dimensión de perfil incompatible: esperado {self.n_nodes}, recibido {len(u)}")
        return float(np.dot(self.volumetric_weights, u))

    def solve_time_series(
        self,
        initial_profile: np.ndarray,
        dt_seconds: float,
        total_seconds: float,
        boundary_conditions: Union[float, List[float], np.ndarray, Tuple[str, Union[float, List[float], np.ndarray]]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Resuelve la evolución temporal continua del perfil radial sobre múltiples pasos
        de forma altamente optimizada (prefactorización tridiagonal O(N)).
        
        Parámetros:
        - initial_profile: Perfil al tiempo t = 0.
        - dt_seconds: Paso de tiempo constante.
        - total_seconds: Duración total de la simulación.
        - boundary_conditions: Valor escalar, array de BCs o tupla (bc_type, bc_values).
        
        Retorna:
        - time_grid: Array 1D con los instantes de tiempo [0, dt, 2dt, ..., T].
        - profiles_history: Array 2D (n_steps + 1, n_nodes) con la evolución de los perfiles.
        - average_history: Array 1D (n_steps + 1) con los valores medios u_bar(t).
        """
        if total_seconds < 0.0:
            raise RadialDiffusionError(f"total_seconds no puede ser negativo: {total_seconds}")
        if dt_seconds <= 0.0:
            raise RadialDiffusionError(f"dt_seconds debe ser estrictamente positivo: {dt_seconds}")

        n_steps = int(round(total_seconds / dt_seconds))
        time_grid = np.linspace(0.0, total_seconds, n_steps + 1)

        # Parsear BCs
        bc_type = "dirichlet"
        if isinstance(boundary_conditions, tuple):
            bc_type, bc_vals = boundary_conditions
        else:
            bc_vals = boundary_conditions

        if isinstance(bc_vals, (int, float)):
            bc_series = np.full(n_steps, float(bc_vals), dtype=np.float64)
        else:
            bc_series = np.asarray(bc_vals, dtype=np.float64)
            if len(bc_series) < n_steps:
                raise RadialDiffusionError(f"Se requieren {n_steps} valores de BC, suministrados {len(bc_series)}.")

        profiles = np.zeros((n_steps + 1, self.n_nodes), dtype=np.float64)
        averages = np.zeros(n_steps + 1, dtype=np.float64)

        current_u = np.asarray(initial_profile, dtype=np.float64).copy()
        if len(current_u) != self.n_nodes:
            raise RadialDiffusionError(f"Dimensión de perfil incompatible: esperado {self.n_nodes}, recibido {len(current_u)}")
        if np.isnan(current_u).any() or np.isinf(current_u).any():
            raise RadialDiffusionError("initial_profile contiene valores NaN o Inf.")

        profiles[0] = current_u
        averages[0] = self.compute_average(current_u)

        if n_steps == 0:
            return time_grid, profiles, averages

        # =====================================================================
        # PREFACTORIZACIÓN DEL SISTEMA TRIDIAGONAL CONSTANTE
        # =====================================================================
        n = self.n_nodes
        dr2 = self.dr ** 2
        theta = self.theta
        alpha = theta * dt_seconds * self.D / dr2
        beta = (1.0 - theta) * dt_seconds * self.D / dr2

        # Precomputar coeficientes espaciales interiores
        j_idx = np.arange(1, n - 1, dtype=np.float64)
        fm = 1.0 - 1.0 / (2.0 * j_idx)
        fp = 1.0 + 1.0 / (2.0 * j_idx)

        # Precomputar diagonales
        a_diag = np.zeros(n, dtype=np.float64)
        b_diag = np.zeros(n, dtype=np.float64)
        c_diag = np.zeros(n, dtype=np.float64)

        b_diag[0] = 1.0 + 4.0 * alpha
        c_diag[0] = -4.0 * alpha

        a_diag[1:n-1] = -alpha * fm
        b_diag[1:n-1] = 1.0 + 2.0 * alpha
        c_diag[1:n-1] = -alpha * fp

        is_dirichlet = (bc_type.strip().lower() == "dirichlet")
        if is_dirichlet:
            a_diag[-1] = 0.0
            b_diag[-1] = 1.0
            c_diag[-1] = 0.0
        else:
            j_last = n - 1
            factor = (1.0 - 0.5 / j_last) / (1.0 - 0.25 / j_last)
            gamma_imp = 2.0 * alpha * factor
            gamma_exp = 2.0 * beta * factor
            a_diag[-1] = -gamma_imp
            b_diag[-1] = 1.0 + gamma_imp
            c_diag[-1] = 0.0

        # Precomputar sweep de Thomas: c_prime e inv_denom
        c_prime = np.zeros(n, dtype=np.float64)
        inv_denom = np.zeros(n, dtype=np.float64)

        b0 = b_diag[0]
        c_prime[0] = c_diag[0] / b0
        inv_denom[0] = 1.0 / b0

        for j in range(1, n):
            denom = b_diag[j] - a_diag[j] * c_prime[j - 1]
            inv_denom[j] = 1.0 / denom
            if j < n - 1:
                c_prime[j] = c_diag[j] * inv_denom[j]

        # Buffers de trabajo
        rhs = np.zeros(n, dtype=np.float64)
        d_prime = np.zeros(n, dtype=np.float64)
        u_k = current_u.copy()
        v_weights = self.volumetric_weights

        # Bucle temporal ultrarrápido
        for step_idx in range(n_steps):
            bc_v = bc_series[step_idx]

            # Construir RHS
            rhs[0] = (1.0 - 4.0 * beta) * u_k[0] + 4.0 * beta * u_k[1]
            rhs[1:n-1] = u_k[1:n-1] + beta * (fm * u_k[0:n-2] - 2.0 * u_k[1:n-1] + fp * u_k[2:n])
            if is_dirichlet:
                rhs[-1] = bc_v
            else:
                rhs[-1] = u_k[-1] + gamma_exp * (u_k[-2] - u_k[-1])

            # Forward sweep TDMA
            d_prime[0] = rhs[0] * inv_denom[0]
            for j in range(1, n):
                d_prime[j] = (rhs[j] - a_diag[j] * d_prime[j - 1]) * inv_denom[j]

            # Back substitution
            u_k[n - 1] = d_prime[n - 1]
            for j in range(n - 2, -1, -1):
                u_k[j] = d_prime[j] - c_prime[j] * u_k[j + 1]

            profiles[step_idx + 1] = u_k
            averages[step_idx + 1] = np.dot(v_weights, u_k)

        return time_grid, profiles, averages


# =============================================================================
# SOLUCIÓN ANALÍTICA DE REFERENCIA (Carslaw & Jaeger 1959 / Crank 1975)
# =============================================================================
def analytical_bessel_cylinder_diffusion(
    r_coords: np.ndarray,
    t_seconds: float,
    radius_m: float,
    diffusivity_m2_s: float,
    u_initial: float = 1.0,
    u_surface: float = 0.0,
    n_terms: int = 60
) -> np.ndarray:
    """
    Calcula la solución analítica exacta mediante serie de funciones de Bessel
    para la difusión radial en un cilindro homogéneo infinito:
    
        u(r, 0) = u_initial
        u(R, t) = u_surface
        du/dr|_{r=0} = 0
    
    Ecuación analítica:
        (u(r, t) - u_surface) / (u_initial - u_surface) = 
            2 * sum_{m=1}^inf [ J_0(lambda_m * r / R) / (lambda_m * J_1(lambda_m)) * exp(-lambda_m^2 * D * t / R^2) ]
            
    donde lambda_m son los ceros positivos de la función de Bessel J_0(lambda) = 0.
    """
    try:
        from scipy.special import j0, j1, jn_zeros
    except ImportError:
        raise ImportError("scipy es requerido para evaluar la solución analítica de Bessel.")

    if t_seconds == 0.0:
        res = np.full_like(r_coords, u_initial, dtype=np.float64)
        res[np.isclose(r_coords, radius_m)] = u_surface
        return res

    zeros = jn_zeros(0, n_terms)
    r = np.asarray(r_coords, dtype=np.float64)
    r_norm = r / radius_m
    tau = diffusivity_m2_s * t_seconds / (radius_m ** 2)

    series_sum = np.zeros_like(r, dtype=np.float64)
    for lam in zeros:
        j0_val = j0(lam * r_norm)
        j1_val = j1(lam)
        term = (j0_val / (lam * j1_val)) * np.exp(-(lam ** 2) * tau)
        series_sum += term

    dimensionless_u = 2.0 * series_sum
    u_exact = u_surface + (u_initial - u_surface) * dimensionless_u
    return u_exact
