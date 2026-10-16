# Scientific Architecture & Mathematical Formulation

This document outlines the governing equations, numerical methods, and coupling architecture implemented in the scientific core of **Q-NYX Fire** (`src/fire_engine`).

---

## 1. Physical Surface Fire Kernel: Rothermel (1972)

The forward rate of spread of a surface fire ($R$) is determined by Rothermel's semi-empirical model (USDA Forest Service INT-115):

$$R = \frac{I_R \cdot \xi \cdot (1 + \phi_w + \phi_s)}{\rho_b \cdot \epsilon \cdot Q_{ig}}$$

Where:
- $I_R$: **Reaction intensity** [$\text{kJ}/(\text{m}^2 \cdot \text{min})$], the rate of heat release per unit area of the fuel bed:
  $$I_R = \Gamma' \cdot w_n \cdot h \cdot \eta_M \cdot \eta_s$$
  - $\Gamma'$: Optimum reaction velocity [$\text{min}^{-1}$]
  - $w_n$: Net fuel loading [$\text{kg}/\text{m}^2$]
  - $h$: Low heat of combustion [$\text{kJ}/\text{kg}$]
  - $\eta_M$: Moisture damping coefficient:
    $$\eta_M = 1 - 2.59 \left(\frac{M_f}{M_x}\right) + 5.11 \left(\frac{M_f}{M_x}\right)^2 - 3.52 \left(\frac{M_f}{M_x}\right)^3$$
  - $\eta_s$: Mineral damping coefficient ($\eta_s = 0.174 \cdot S_e^{-0.19}$)
- $\xi$: **Propagating flux ratio** [dimensionless], representing the proportion of reaction intensity directed towards unburned fuel:
  $$\xi = \frac{\exp\left[\left(0.792 + 0.681 \cdot \sigma^{0.5}\right) \cdot (\beta + 0.1)\right]}{192 + 0.2595 \cdot \sigma}$$
- $\phi_w$: **Wind multiplier** [dimensionless]:
  $$\phi_w = C \cdot U^B \cdot \left(\frac{\beta}{\beta_{\text{opt}}}\right)^{-E}$$
  with Rothermel wind limit cap ($U \le 0.9 \cdot I_R$).
- $\phi_s$: **Slope multiplier** [dimensionless]:
  $$\phi_s = 5.275 \cdot \beta^{-0.3} \cdot (\tan \Phi)^2$$
- $\rho_b$: Fuel bed bulk density ($\rho_b = w_0 / \delta$).
- $\epsilon$: Effective heating number ($\epsilon = \exp(-138 / \sigma)$).
- $Q_{ig}$: Heat of preignition ($Q_{ig} = 581 + 2594 \cdot M_f$).

---

## 2. Vector Combination of Wind and Slope (Albini 1976 / Finney 1998)

When wind and slope act simultaneously in non-collinear directions, their multipliers are treated as vector quantities:

$$\vec{\phi}_{\text{resultant}} = \vec{\phi}_w + \vec{\phi}_s$$

$$\phi_{\text{resultant}} = \sqrt{\phi_w^2 + \phi_s^2 + 2 \cdot \phi_w \cdot \phi_s \cdot \cos(\theta_w - \theta_s)}$$

$$\tan(\theta_{\text{resultant}} - \theta_w) = \frac{\phi_s \cdot \sin(\theta_s - \theta_w)}{\phi_w + \phi_s \cdot \cos(\theta_s - \theta_w)}$$

Where:
- $\theta_w$: Direction towards which wind blows (Downwind heading = Wind FROM + 180°).
- $\theta_s$: Direction of steepest terrain ascent (Upslope heading = Downslope Aspect + 180°).

---

## 3. Elliptical Wave Expansion (Anderson 1983 / Alexander 1985)

The two-dimensional fire footprint under effective wind is modeled as an ellipse with the ignition point at the rear focus:

1. **Length-to-Breadth Ratio ($L/B$)**:
   $$\frac{L}{B} = 0.936 \cdot \exp(0.2563 \cdot U_{\text{eff}}) + 0.461 \cdot \exp(-0.1548 \cdot U_{\text{eff}}) - 0.397$$
2. **Ellipse Eccentricity ($e$)**:
   $$e = \frac{\sqrt{(L/B)^2 - 1}}{L/B}$$
3. **Directional Rate of Spread ($R(\theta)$)**:
   For any propagation angle $\theta$ relative to the resultant heading $\theta_{\text{resultant}}$:
   $$R(\theta) = R_{\text{head}} \cdot \frac{1 - e}{1 - e \cdot \cos(\theta - \theta_{\text{resultant}})}$$
   - Head ROS: $R(0) = R_{\text{head}}$
   - Back ROS: $R(\pi) = R_{\text{head}} \cdot \frac{1 - e}{1 + e}$
   - Flank ROS: $R(\pm \pi/2) = R_{\text{head}} / (L/B)$

---

## 4. Physical Dead Fuel Moisture: Nelson (2000)

Dead fuel moisture content is computed dynamically by solving the one-dimensional radial moisture diffusion equation inside cylindrical woody fuel sticks:

$$\frac{\partial m}{\partial t} = \frac{1}{r} \frac{\partial}{\partial r} \left( r \cdot D(m, T) \cdot \frac{\partial m}{\partial r} \right)$$

Boundary conditions at the stick surface ($r = R_{\text{stick}}$) couple heat and mass exchange:
- Surface convective heat exchange ($h_c \cdot (T_{\text{air}} - T_{\text{surface}})$)
- Solar radiation absorption ($\alpha \cdot R_{\text{solar}}$)
- Evaporative cooling and condensation latent heat
- Equilibrium moisture content (EMC) sorption isotherms (Nelson 1984 / NFDRS4).

For operational execution, sequential 5-minute pre-conditioned lookups are generated via `NelsonFuelMoistureProvider` with 48 hours of antecedent meteorological conditioning.

---

## 5. Discrete Propagation Engine: 16-Neighbor Stencil Dijkstra

Fire arrival times on the spatial grid are computed using Dijkstra's minimum arrival time algorithm:

1. **Graph Construction**: Each raster cell $(i, j)$ connects to its 16 neighbors on a 2-hop stencil (Stencil-16).
2. **Metric Correction**: Weight of edge $(u, v)$ with displacement $(\Delta x, \Delta y)$ and distance $d = \sqrt{\Delta x^2 + \Delta y^2}$:
   $$\tau(u, v) = \frac{d}{R(\theta_{u \to v})}$$
   where $R(\theta_{u \to v})$ is the directional rate of spread evaluated with fuel, slope, and weather conditions between cells $u$ and $v$.
3. **Numerical Anisotropy**: The 16-neighbor stencil reduces grid metric distortion from $> 8.2\%$ (standard 8-neighbor Moore stencil) to $< 2.8\%$ across all orientations.
