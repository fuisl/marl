# Influence Visualization Math Notes

This note documents how `visualization/graph_influence.py` computes the node map and the smoothed spatial field.

## 1. Jacobian influence between node pairs

For a trained model output (embedding or logit target), the node-pair influence is computed as:

$$
I(v,u) = \sum_i \sum_j \left|\frac{\partial H_v^i}{\partial X_u^j}\right|
$$

where:

- $X_u$ is the input feature vector at node $u$,
- $H_v$ is the chosen model output vector at focal node $v$.

In this repo, this is the L1 aggregation of the Jacobian in `_jacobian_l1_safe(...)`.

## 2. Hop-shell total influence

For hop distance $h$:

$$
T_h(v) = \sum_{u:\rho(v,u)=h} I(v,u)
$$

The code builds exact hop shells and sums influences over each shell.

## 3. Global node influence used by the map

When `analysis.focal_node_index=null`, we compute an all-node global score:

$$
I_{\text{global}}(u) = \frac{1}{|V|}\sum_{v\in V} I(v,u)
$$

This is implemented in `compute_global_influence_rows(...)`.

Interpretation:

- High $I_{\text{global}}(u)$ means node $u$ tends to influence many focal nodes strongly.
- Low does **not** mean no influence; it may be nonzero but small relative to the heavy tail.

## 4. Smoothed spatial influence field

Given node positions $(x_u,y_u)$ and nonnegative node weights $w_u$, the plotted field is a Gaussian kernel mixture on a regular grid:

$$
F(x,y) = \frac{1}{\sum_u w_u}\sum_u w_u\exp\left(-\frac{(x-x_u)^2+(y-y_u)^2}{2\sigma^2}\right)
$$

where:

- $
\sigma = \text{heat\_sigma\_scale} \times \text{map diagonal length}$.

This is implemented in `_compute_smoothed_field(...)`.

### Weight source (`analysis.heat_weight_mode`)

- `raw` (default): $w_u = I_{\text{raw}}(u)$
- `normalized`: $w_u = I_{\text{normalized}}(u)$
- `log_normalized`: $w_u = \text{robust-log-normalized}(I_{\text{raw}}(u))$

For your use case (global influence field), `raw` is the most faithful.

## 5. Why some nodes look faint

If many nodes are visible but only a few are bright, this usually indicates a heavy-tailed influence distribution, not zero influence. Small values can still be nonzero.

Practical controls to reveal weaker regions:

- reduce `analysis.heat_sigma_scale` for sharper local variation,
- use `analysis.heat_weight_mode=normalized` or `log_normalized`,
- lower `analysis.heat_percentile_low`.
