"""
Attention Dynamics on S^{d-1}
==============================

Integrates the continuous-time self-attention ODE on the unit sphere:

  Linear:   dx_k/dt = P^perp_{x_k} ( beta/n     sum_j <Ax_k, x_j> V x_j )
  Softmax:  dx_k/dt = P^perp_{x_k} ( 1/Z_k      sum_j exp(beta <Ax_k, x_j>) V x_j )

where:
  - x_k in S^{d-1}  (unit sphere)
  - P^perp_x y = y - <x, y> x  (projection onto tangent space at x)
  - Z_k = sum_j <Qx_k, Kx_j>          (linear, raw sum)
        or n                             (linear, uniform)
        or sum_j exp(beta <Qx_k, Kx_j>) (softmax, always positive)
  - Q, K, V are d x d real matrices;i A = Q^T K is the combined query-key matrx


Matrix cases (from 2D OA analysis):
  Case 1: V = I,   A random symmetric PD  -> clustering governed by definiteness of A+A^T
  Case 2: A = I,   V symmetric            -> clustering governed by top eigenvalue of V
  Case 4: A = I,   V = block-diag of [[a,b],[-b,-a]] -> Hamiltonian/bifurcation
  Random: A, V both random

Tokens are initialised uniformly on S^{d-1} via normalised Gaussian draws.
ODE solver: RK45 (default), DOP853, or Radau — selectable in sidebar.

More details about ODE solvers and stiffness:
- RK45: Explicit Runge-Kutta method, good for non-stiff problems. Fast but may struggle with stiff dynamics, leading to many function evaluations or failure.
- DOP853: Higher-order explicit method, can be more efficient for smooth problems but still struggles with stiffness.
- Radau: Implicit method designed for stiff problems. More robust when stiffness is present but computationally heavier.

Stiffness can arise if tokens rapidly cluster together, causing large gradients and requiring very small time steps for explicit solvers. If you see warnings about excessive function evaluations or suspect stiffness, try switching to Radau for more reliable integration.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import ortho_group
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Attention Dynamics on Sphere", layout="wide")
st.title("Attention Token Dynamics on $\\mathbb{S}^{d-1}$")
st.markdown(r"""
Integrates the continuous-time self-attention ODE on the unit sphere $\mathbb{S}^{d-1}$:

$$\dot{x}_k = P^\perp_{x_k}\!\left(\frac{1}{Z_k}\sum_{j=1}^n f\!\left(\beta\langle Ax_k, x_j\rangle\right) V x_j\right), \qquad A = Q^\top K$$

with $P^\perp_x y = y - \langle x, y\rangle x$, and $f(s) = s$ (linear) or $f(s) = e^s$ (softmax).
""")

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Simulation controls")

    n_tokens = int(st.number_input("Number of tokens  n", min_value=2, max_value=1000, value=20, step=1))
    d        = int(st.number_input("Token dimension  d", min_value=2, max_value=128, value=4, step=1))
    T        = float(st.number_input("End time  T", min_value=0.1, value=5.0, step=0.5))
    frames   = int(st.slider("Time samples", min_value=100, max_value=5000, value=500, step=50))

    st.markdown("---")
    attn_type   = st.radio("Attention type", ["Softmax  f(s) = exp(s)", "Linear  f(s) = s"])
    use_softmax = attn_type.startswith("Softmax")
    beta        = st.slider("β (inverse temperature)", min_value=0.0, max_value=10.0, value=1.0, step=0.01)
    if not use_softmax:
        st.info("Linear attention: Z_k = n (uniform normalisation).")

    st.markdown("---")
    st.subheader("Matrix case")
    case = st.radio(
        "Choose case",
        ["Case 1: V=I, A arbitrary",
         "Case 2: A=I, V symmetric",
         "Case 3: A=I, V rotation",
         "Case 4: A=I, V Hamiltonian",
         "Random"]
    )
    matrix_seed = int(st.number_input("Matrix random seed", min_value=0, max_value=10**9, value=42, step=1))
    rng_mat     = np.random.default_rng(int(matrix_seed))

    st.markdown("---")

    # ── Case-specific parameters ──────────────────────────────────────────────
    if case == "Case 1: V=I, A arbitrary":
        st.markdown(r"""
**Case 1**: $V = I$, $A$ arbitrary.

The precise conditions (from OA Proposition) are:

- **Full clustering** ($\rho \to 1$): $A$ has at least one eigenvalue with
  non-negative real part. Equivalently: $\mathrm{tr}(A) > 0$, or
  $\mathrm{tr}(A) \leq 0$ and $\det(A) \leq 0$.
- **Partial sync** ($\rho \to \rho_{\mathrm{eq}} \in (0,1)$):
  $\frac{A+A^\top}{2} \prec 0$ (symmetric part of $A$ is negative definite).
        """)
        regime  = st.radio("Regime", [
            "Full clustering: tr(A) > 0",
            "Full clustering: tr(A) ≤ 0 & det(A) ≤ 0",
            "Partial sync: (A+Aᵀ)/2 ≺ 0"
        ])
        a_scale = st.slider("Off-diagonal scale", min_value=0.0, max_value=3.0, value=0.5, step=0.1,
                            help="Scale of skew-symmetric perturbation added to diagonal base.")
        Q_orth  = ortho_group.rvs(d, random_state=int(matrix_seed))
        if regime.startswith("Full clustering: tr(A) > 0"):
            # tr(A) > 0: build A with all positive eigenvalues
            eigvals_A = rng_mat.uniform(0.5, 2.0, size=d)
            A_sym     = Q_orth @ np.diag(eigvals_A) @ Q_orth.T
            perturb   = rng_mat.normal(0.0, 1.0, (d, d))
            skew_pert = (perturb - perturb.T) * float(a_scale) * 0.2
            A  = A_sym + skew_pert
            ev_A = np.linalg.eigvals(A)
            st.success(
                f"tr(A) = {np.trace(A):.3f} > 0, "
                f"det(A) = {np.linalg.det(A):.3f} → **full clustering**"
            )
        elif regime.startswith("Full clustering: tr(A) ≤ 0"):
            # tr(A) <= 0 & det(A) <= 0:
            # In d-dim: build A with mixed eigenvalues — at least one positive,
            # rest sufficiently negative so that tr <= 0 but det <= 0 (product of
            # eigenvalues has sign determined by number of negative ones).
            # Strategy: one large positive eigenvalue, rest negative but smaller in magnitude.
            # This gives tr = pos + sum(neg) <= 0 possible, det = pos * prod(neg) <= 0
            # since odd number of negatives gives det < 0.
            n_neg     = d - 1   # d-1 negative eigenvalues, 1 positive
            pos_val   = rng_mat.uniform(0.3, 0.8)                     # small positive
            neg_vals  = -rng_mat.uniform(pos_val / (n_neg - 0.5),
                                          pos_val / (n_neg - 0.5) * 1.5,
                                          size=n_neg)                  # neg, sum > pos_val
            eigvals_A = np.concatenate([[pos_val], neg_vals])
            A_sym     = Q_orth @ np.diag(eigvals_A) @ Q_orth.T
            perturb   = rng_mat.normal(0.0, 1.0, (d, d))
            skew_pert = (perturb - perturb.T) * float(a_scale) * 0.2
            A         = A_sym + skew_pert
            tr_A  = np.trace(A)
            det_A = np.linalg.det(A)
            ev_A  = np.linalg.eigvals(A)
            has_nonneg = any(v.real >= 0 for v in ev_A)
            tr_ok  = tr_A  <= 0
            det_ok = det_A <= 0
            if tr_ok and det_ok and has_nonneg:
                st.success(
                    f"tr(A) = {tr_A:.3f} ≤ 0,  det(A) = {det_A:.3e} ≤ 0,  "
                    f"A has eigenvalue with Re ≥ 0 → **full clustering**"
                )
            else:
                st.warning(
                    f"tr(A) = {tr_A:.3f},  det(A) = {det_A:.3e} — "
                    f"conditions partially met (try adjusting seed or off-diagonal scale)"
                )
        else:
            # Build A with (A+A^T)/2 negative definite: use negative eigenvalues for sym part
            eigvals_A = rng_mat.uniform(0.5, 2.0, size=d)
            A_sym     = Q_orth @ np.diag(-eigvals_A) @ Q_orth.T   # sym part is NSD
            perturb   = rng_mat.normal(0.0, 1.0, (d, d))
            skew_pert = (perturb - perturb.T) * float(a_scale) * 0.2
            A  = A_sym + skew_pert
            sym_part_ev = np.linalg.eigvalsh(0.5*(A + A.T))
            sym_nd = np.all(sym_part_ev < 0)
            if sym_nd:
                # Compute rho_eq analogue: ratio of Frobenius norms of skew to sym parts
                sym_A = 0.5*(A + A.T)
                skw_A = 0.5*(A - A.T)
                rho_eq_approx = np.linalg.norm(skw_A, "fro") / np.linalg.norm(sym_A, "fro")
                rho_eq_approx = min(rho_eq_approx, 0.999)
                st.warning(
                    f"(A+Aᵀ)/2 ≺ 0 (all sym eigenvalues < 0) → **partial sync**, "
                    f"ρ_eq ≈ {rho_eq_approx:.3f} (2D formula generalised)"
                )
            else:
                st.error("Sym part not fully ND — increase off-diagonal scale or adjust seed")
        V = np.eye(d)
        ev_A_show = np.linalg.eigvals(A)
        ev_strs = [f"{v.real:.3f}{v.imag:+.3f}j" if abs(v.imag)>1e-10 else f"{v.real:.3f}" for v in ev_A_show]
        st.write("A eigenvalues:", "  |  ".join(ev_strs))
        st.write(f"tr(A) = {np.trace(A):.4f},  det(A) = {np.linalg.det(A):.4f}")
        st.write("(A+Aᵀ)/2 eigenvalues:", np.round(np.linalg.eigvalsh(0.5*(A+A.T)), 3).tolist())
    
    elif case == "Case 2: A=I, V symmetric":
        st.markdown(r"""
**Case 2**: $A = I$, $V$ symmetric.

Clustering governed by the **top eigenvalue** $\lambda_1$ of $V$.
Positive $\lambda_1$ → clustering; negative → dispersion.
        """)
        top_eig     = st.slider("Top eigenvalue λ₁ of V", min_value=-5.0, max_value=5.0, value=2.0, step=0.1)
        other_scale = st.slider("Other eigenvalues scale", min_value=0.0, max_value=2.0, value=0.5, step=0.1)
        Q_orth      = ortho_group.rvs(d, random_state=int(matrix_seed))
        # Draw remaining eigenvalues and clamp them so top_eig is always
        # the true spectral maximum, regardless of its sign:
        #   - top_eig >= 0: clamp others to (-|top_eig|*0.9, +|top_eig|*0.9)
        #     so they are strictly smaller in value
        #   - top_eig <  0: clamp others to (-inf, top_eig - eps)
        #     so they are all strictly more negative — all eigenvalues negative,
        #     top_eig is the least negative (= largest), dispersion guaranteed
        eigvals_V = rng_mat.normal(0.0, float(other_scale), size=d)
        if d > 1:
            lam = float(top_eig)
            if lam > 0:
                # Others must be strictly < lam: clamp to (-0.9*lam, 0.9*lam)
                cap = lam * 0.9
                eigvals_V[1:] = np.clip(eigvals_V[1:], -cap, cap)
            elif lam < 0:
                # Others must be strictly < lam (more negative): clamp above to lam - eps
                eigvals_V[1:] = np.clip(eigvals_V[1:], -abs(lam) * 2, lam - 1e-3)
            else:
                # lam == 0: all others negative
                eigvals_V[1:] = -np.abs(eigvals_V[1:]) - 1e-3
        eigvals_V[0] = float(top_eig)
        A = np.eye(d)
        V = Q_orth @ np.diag(eigvals_V) @ Q_orth.T
        actual_top = np.linalg.eigvalsh(V).max()
        if float(top_eig) >= 0:
            st.success(f"Top eigenvalue of V: **{actual_top:.3f}** → clustering expected")
        else:
            st.error(f"Top eigenvalue of V: **{actual_top:.3f}** → dispersion expected")
        st.write("All V eigenvalues:", np.round(np.sort(eigvals_V)[::-1], 3).tolist())

  
    elif case == "Case 3: A=I, V rotation":
        st.markdown(r"""
**Case 3**: $A = I$, $V = aI + bJ$ where $J$ is the block-diagonal skew matrix.

In 2D this is $V = \begin{pmatrix}a & b \\ -b & a\end{pmatrix}$ with $a > 0$.
The OA dynamics simplify to:
$$\dot{\rho} = \tfrac{a\beta}{2}(1-\rho^2)\rho, \qquad \dot{\phi} = \tfrac{b\beta}{2}(\rho^2+3)$$

Since $a > 0$, $\rho \to 1$ exponentially — **always clusters** regardless of $b$.
The parameter $b$ controls **rotation speed** of the cluster phase $\phi$.
For $d > 2$: $V$ is block-diagonal with $\lfloor d/2\rfloor$ identical $2\times 2$ blocks
$\begin{pmatrix}a & b \\ -b & a\end{pmatrix}$, plus $V_{dd} = a$ if $d$ is odd.
        """)
        a_param = st.slider("a  (controls clustering rate)", min_value=0.01, max_value=3.0, value=1.0, step=0.05,
                            help="Must be > 0 for clustering. Larger a = faster convergence.")
        b_param = st.slider("b  (controls rotation)", min_value=-3.0, max_value=3.0, value=0.5, step=0.05,
                            help="b=0: no rotation. b≠0: cluster phase rotates at rate b*beta/2*(rho^2+3).")
        if a_param > 0:
            st.success(f"a = {a_param} > 0 → **clustering guaranteed**, rotation rate ∝ b = {b_param}")
        else:
            st.error("a must be > 0 for clustering guarantee")
        A = np.eye(d)
        # Build V = block-diag of [[a,b],[-b,a]]
        block = np.array([[a_param, b_param], [-b_param, a_param]])
        V = np.zeros((d, d))
        for i in range(0, d - 1, 2):
            V[i:i+2, i:i+2] = block
        if d % 2 == 1:
            V[d-1, d-1] = a_param

    elif case == "Case 4: A=I, V Hamiltonian":
        st.markdown(r"""
**Case 4**: $A = I$, $V$ built from $2\times 2$ Hamiltonian blocks
$\begin{pmatrix}a & b \\ -b & -a\end{pmatrix}$.

- $a > b$: complete clustering
- $a < b$: cyclic / oscillatory (bifurcation)
- $a = b$: bifurcation point

For $d > 2$: $V$ is block-diagonal with $\lfloor d/2 \rfloor$ identical blocks.
        """)
        a_param = st.slider("a", min_value=-3.0, max_value=3.0, value=1.0, step=0.05)
        b_param = st.slider("b", min_value=-3.0, max_value=3.0, value=0.5, step=0.05)
        if a_param > b_param:
            st.success(f"a > b → **clustering regime**")
        elif a_param < b_param:
            st.warning(f"a < b → **cyclic / oscillatory regime**")
        else:
            st.error(f"a = b → **bifurcation point**")
        A     = np.eye(d)
        block = np.array([[a_param, b_param], [-b_param, -a_param]])
        V     = np.zeros((d, d))
        for i in range(0, d - 1, 2):
            V[i:i+2, i:i+2] = block
        if d % 2 == 1:
            V[d-1, d-1] = 0

    else:  # Random
        st.markdown(r"""
**Random**: $A$ and $V$ drawn from $\mathcal{N}(0, \sigma^2/d)$.

Random $A$ has no guaranteed sign structure so clustering is not guaranteed.
The app will show whether $A + A^\top$ is PD (clustering indicator for linear).
        """)
        matrix_scale = st.slider("Entry std-dev σ", min_value=0.1, max_value=3.0, value=1.0, step=0.1)
        scale = float(matrix_scale) / np.sqrt(d)
        A = rng_mat.normal(0.0, scale, (d, d))
        V = rng_mat.normal(0.0, scale, (d, d))
        ApAt     = A + A.T
        ev_ApAt  = np.linalg.eigvalsh(ApAt)
        if np.all(ev_ApAt > 0):
            st.success("A + Aᵀ ≻ 0 → clustering expected (linear)")
        elif np.all(ev_ApAt < 0):
            st.error("A + Aᵀ ≺ 0 → dispersion expected (linear)")
        else:
            st.warning("A + Aᵀ indefinite → mixed behavior (linear)")

    st.markdown("---")
    st.subheader("Token initialisation")
    if "token_seed" not in st.session_state:
        st.session_state.token_seed = int(np.random.SeedSequence().entropy) % (10**9)
    if st.button("Randomise initial tokens"):
        st.session_state.token_seed = int(np.random.SeedSequence().entropy) % (10**9)
    st.write("Token seed:", st.session_state.token_seed)

    st.markdown("---")
    solver_choice = st.selectbox(
        "ODE solver", ["RK45", "DOP853", "Radau"], index=0,
        help="RK45/DOP853: explicit, fast. Radau: implicit, use if stiffness warnings appear."
    )

# ──────────────────────────────────────────────────────────────────────────────
# Show matrices
# ──────────────────────────────────────────────────────────────────────────────
with st.expander("Show A and V matrices and eigenvalues"):
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**A** ({d}×{d})")
        st.dataframe(pd.DataFrame(np.round(A, 4)).astype(float))
        ev = np.linalg.eigvalsh(A) if np.allclose(A, A.T) else np.linalg.eigvals(A)
        ev_strs = [f"{v.real:.3f}{v.imag:+.3f}j" if abs(v.imag) > 1e-10 else f"{v.real:.3f}" for v in ev]
        st.write("Eigenvalues:", "  |  ".join(ev_strs))
    with col2:
        st.write(f"**V** ({d}×{d})")
        st.dataframe(pd.DataFrame(np.round(V, 4)).astype(float))
        ev2 = np.linalg.eigvalsh(V) if np.allclose(V, V.T) else np.linalg.eigvals(V)
        ev_strs2 = [f"{v.real:.3f}{v.imag:+.3f}j" if abs(v.imag) > 1e-10 else f"{v.real:.3f}" for v in ev2]
        st.write("Eigenvalues:", "  |  ".join(ev_strs2))

# ──────────────────────────────────────────────────────────────────────────────
# Initialise tokens on S^{d-1}
# ──────────────────────────────────────────────────────────────────────────────
rng_tok = np.random.default_rng(int(st.session_state.token_seed))
X0_raw  = rng_tok.standard_normal((n_tokens, d))
X0      = X0_raw / np.linalg.norm(X0_raw, axis=1, keepdims=True)
x0_flat = X0.ravel()

# ──────────────────────────────────────────────────────────────────────────────
# ODE right-hand side
# ──────────────────────────────────────────────────────────────────────────────
def make_rhs(A, V, n, d, beta, use_softmax):
    """
    dx_k/dt = P^perp_{x_k} ( 1/Z_k  sum_j f(beta <Ax_k, x_j>) V x_j )

    scores[k,j] = <Ax_k, x_j> = (X @ A.T @ X.T)[k,j]
    f_scores    = beta * scores           (linear,  Z = n)
                = exp(beta * scores)      (softmax, Z = row sum)
    update[k]   = (1/Z_k) sum_j f_scores[k,j] (V x_j)
    dX[k]       = update[k] - <x_k, update[k]> x_k
    """
    AT = A.T
    VT = V.T

    def rhs(t, x_flat):
        X = x_flat.reshape(n, d)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / np.maximum(norms, 1e-12)

        scores = (X @ AT) @ X.T              # (n, n)  scores[k,j] = <Ax_k, x_j>

        if use_softmax:
            s = beta * scores
            s -= s.max(axis=1, keepdims=True)
            f_scores = np.exp(s)
            Z = f_scores.sum(axis=1, keepdims=True)
        else:
            f_scores = beta * scores
            Z = float(n)

        XV     = X @ VT                      # (n, d)  V x_j for each j
        update = (f_scores / Z) @ XV         # (n, d)

        inner = np.sum(X * update, axis=1, keepdims=True)
        dX    = update - inner * X

        return dX.ravel()

    return rhs

rhs_fn = make_rhs(A, V, n_tokens, d, beta, use_softmax)

# ──────────────────────────────────────────────────────────────────────────────
# Integrate
# ──────────────────────────────────────────────────────────────────────────────
t_eval = np.linspace(0.0, T, frames)
label  = "Softmax" if use_softmax else "Linear"

with st.spinner(f"Integrating {n_tokens} tokens × d={d}  |  {case}  |  {label}  |  {solver_choice}..."):
    sol = solve_ivp(
        rhs_fn,
        (0.0, T),
        x0_flat,
        method=solver_choice,
        t_eval=t_eval,
        atol=1e-8,
        rtol=1e-6,
        max_step=T / 200.0,
        dense_output=False,
    )

if not sol.success:
    st.error(f"ODE solver failed: {sol.message}")
    st.stop()

X_traj     = sol.y.reshape(n_tokens, d, frames)
norms_traj = np.linalg.norm(X_traj, axis=1, keepdims=True)
X_traj     = X_traj / np.maximum(norms_traj, 1e-12)

stiff = sol.nfev > 50_000
st.success(
    f"✅ Done — **{sol.nfev}** function evaluations.  "
    + ("⚠️ High eval count — consider Radau." if stiff else "No stiffness issues.")
)
max_norm_dev = np.max(np.abs(np.linalg.norm(X_traj, axis=1) - 1.0))
st.info(f"Max deviation from unit sphere: **{max_norm_dev:.2e}**")

# ──────────────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Token dynamics")

# Clustering metric: <x_i, x_j>^2  (squared cosine similarity)
# This is the correct analogue of the 2D order parameter e^{2i*theta}:
# tokens at theta and theta+pi are antipodal (cos sim = -1) but represent
# the same cluster, so we use cos^2 which equals 1 for both aligned and antipodal.
# In higher dims: <x_i,x_j>^2 = ||x_i x_i^T - x_j x_j^T||^2 proxy (Frobenius).
idx_i, idx_j    = np.triu_indices(n_tokens, k=1)
cos_sim_time    = np.zeros((len(idx_i), frames))   # raw <x_i,x_j>
cos2_sim_time   = np.zeros((len(idx_i), frames))   # squared <x_i,x_j>^2
for t_idx in range(frames):
    Xt = X_traj[:, :, t_idx]
    G  = Xt @ Xt.T                          # Gram matrix, entries = <x_i,x_j>
    cos_sim_time[:,  t_idx] = G[idx_i, idx_j]
    cos2_sim_time[:, t_idx] = G[idx_i, idx_j]**2

mean_cos  = cos_sim_time.mean(axis=0)
std_cos   = cos_sim_time.std(axis=0)
mean_cos2 = cos2_sim_time.mean(axis=0)
std_cos2  = cos2_sim_time.std(axis=0)

fig1, axes = plt.subplots(2, 1, figsize=(15, 12))

ax = axes[0]
# Plot squared cosine similarity (correct clustering metric)
ax.plot(sol.t, mean_cos2, color="tomato", lw=2, label=r"mean $\langle x_i,x_j\rangle^2$")
ax.fill_between(sol.t, mean_cos2 - std_cos2, mean_cos2 + std_cos2,
                alpha=0.2, color="tomato", label="±1 std")
# Also show raw cosine sim as thin dashed for reference
ax.plot(sol.t, mean_cos, color="steelblue", lw=1, linestyle='--',
        alpha=0.6, label=r"mean $\langle x_i,x_j\rangle$")
ax.axhline(1.0, color='green', lw=0.8, linestyle='--', label='full clustering = 1')
ax.axhline(0.0, color='gray',  lw=0.8, linestyle=':')
ax.set_xlabel("t")
ax.set_ylabel(r"$\langle x_i, x_j\rangle^2$")
ax.set_title(f"Clustering metric — {case} - dim {d} - # of tokens: {n_tokens} - {label} attention")
ax.set_xlim(0, T)
ax.set_ylim(-0.05, 1.05)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

dim_means = X_traj.mean(axis=0)
ax2 = axes[1]
cmap = plt.cm.tab10
for dim_idx in range(min(d, 10)):
    ax2.plot(sol.t, dim_means[dim_idx], color=cmap(dim_idx % 10),
             lw=1.5, label=f"dim {dim_idx}")
ax2.set_xlabel("t")
ax2.set_ylabel("mean token value")
ax2.set_title("Per-dimension mean across tokens")
ax2.set_xlim(0, T)
ax2.grid(alpha=0.3)
if d <= 10:
    ax2.legend(fontsize=7, ncol=2)

plt.tight_layout()
st.pyplot(fig1)

# st.markdown(
#     r"""
# #### What this graph shows and why we plot it this way

# **The naive approach — and why it fails.**
# The most natural way to measure whether tokens are clustering is to track the raw
# pairwise cosine similarity $\langle x_i, x_j \rangle$ between every pair of tokens.
# If all tokens collapse to the same point on $\mathbb{S}^{d-1}$, every pairwise cosine
# similarity equals $1$, and the mean would rise to $1$ over time.

# However, this fails for these dynamics. The continuous-time attention ODE does **not**
# push all tokens to the same point — it pushes them to cluster in the sense of the
# second-order parameter $R_2$. In the 2D case ($d=2$, tokens on $\mathbb{S}^1$), the
# order parameter tracked by the OA reduction is

# $$R_2(t) = \frac{1}{n} \sum_{j=1}^n e^{2i\theta_j},$$

# not $R_1 = \frac{1}{n}\sum e^{i\theta_j}$. The factor of $2$ in the exponent means
# the dynamics are $\pi$-periodic in $\theta$: a token at angle $\theta$ and a token at
# $\theta + \pi$ are treated as **identical** by $R_2$, even though as vectors on
# $\mathbb{S}^1$ they are antipodal ($\langle x_i, x_j \rangle = -1$). Full clustering
# in the $R_2$ sense means tokens split into groups at $\theta^*$ and $\theta^* + \pi$,
# so the mean raw cosine similarity is exactly $0$ even when the system is perfectly
# clustered. This is precisely why earlier plots showed a flat line at $0$ regardless
# of parameters.

# **The correct metric.**
# The right analogue of $|R_2| \to 1$ in dimension $d$ is to measure similarity between
# the rank-1 projection matrices $x_i x_i^\top$ rather than between the vectors $x_i$
# themselves. The Frobenius inner product between two such projections is

# $$\langle x_i x_i^\top,\, x_j x_j^\top \rangle_F = \mathrm{tr}(x_i x_i^\top x_j x_j^\top) = \langle x_i, x_j \rangle^2.$$

# This equals $1$ whether $x_i = x_j$ (aligned) or $x_i = -x_j$ (antipodal), and equals
# $0$ when the tokens are orthogonal. It is invariant under the sign flip $x \mapsto -x$
# that the dynamics treat as equivalent.

# **What to look for on the graph.**
# - $\langle x_i, x_j \rangle^2 \to 1$: tokens are clustering (all aligning or forming
#   antipodal pairs), consistent with $|R_2| \to 1$.
# - $\langle x_i, x_j \rangle^2 \approx 1/d$: tokens remain approximately uniformly
#   spread on $\mathbb{S}^{d-1}$ — no clustering.
# - Oscillating $\langle x_i, x_j \rangle^2$: cyclic / Hamiltonian behavior
#   (Case 4 with $a < b$).

# The shaded band is $\pm 1$ standard deviation across all $\binom{n}{2}$ pairs.
# A narrow band near $1$ means tight single-cluster behavior; a wide band means tokens
# have split into multiple distinct clusters. The dashed blue line shows raw cosine
# similarity for reference — note how it stays near $0$ even when the squared version
# is near $1$, which is why we cannot use it as a clustering diagnostic here.
#     """
# )

with st.expander("Pairwise cosine similarity distribution  (t=0 vs t=T)"):
    vals_t0 = cos2_sim_time[:, 0]
    vals_tT = cos2_sim_time[:, -1]

    # Detect spike at 1 (clustering): if >50% of pairs are within 1e-3 of 1
    spike_t0 = np.mean(vals_t0 > 0.999)
    spike_tT = np.mean(vals_tT > 0.999)

    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4), sharey=False)

    for ax_h, vals, t_label, color, spike in [
        (axes2[0], vals_t0, "t=0",   "steelblue", spike_t0),
        (axes2[1], vals_tT, f"t={T}", "tomato",    spike_tT),
    ]:
        # Always use fixed bins over [0,1] so the spike at 1 is visible
        bins = np.linspace(0, 1, 42)   # 41 bins, last bin catches values at 1
        interior = vals[vals < 0.999]
        at_one   = vals[vals >= 0.999]

        if len(interior) > 0:
            ax_h.hist(interior, bins=bins[:-1], alpha=0.7, color=color, label="spread pairs")
        if len(at_one) > 0:
            # Draw the spike at 1 as a separate bar so it is always visible
            bar_width = bins[1] - bins[0]
            ax_h.bar(1.0, len(at_one), width=bar_width, align="edge",
                     color="green", alpha=0.8, label=f"clustered pairs: {spike*100:.1f}%")

        ax_h.axvline(1.0, color="green", lw=1, linestyle="--")
        ax_h.set_xlabel(r"$\langle x_i,x_j\rangle^2$")
        ax_h.set_ylabel("Count")
        ax_h.set_xlim(-0.02, 1.08)
        ax_h.set_title(f"Distribution at {t_label}")
        ax_h.legend(fontsize=8)
        ax_h.grid(alpha=0.3)

    plt.suptitle(r"Squared cosine similarity distribution: $t=0$ vs $t=T$", y=1.02)
    plt.tight_layout()
    st.pyplot(fig2)

#     st.markdown(r"""
# **Why this histogram and what the $\pm 1$ std band means.**

# Each observation in this histogram is one pair of tokens $(x_i, x_j)$, plotted by
# their squared cosine similarity $\langle x_i, x_j
# angle^2 \in [0, 1]$.
# At $t=0$ (blue), tokens are uniformly spread on $\mathbb{S}^{d-1}$ so most pairs
# are nearly orthogonal and the distribution sits near $0$.
# At $t=T$ (red/green), the distribution tells you the clustering structure:

# - **Spike at $1$, no spread**: all pairs score $1$ — tokens have collapsed into a
#   single cluster (or a pair of antipodal clusters), which the ODE treats as identical.
# - **Two peaks (one near $0$, one near $1$)**: there are multiple distinct clusters.
#   Pairs of tokens *within* the same cluster score $approx 1$; pairs *across*
#   different clusters score $< 1$. The gap between the peaks tells you how
#   geometrically separated the clusters are.
# - **Broad distribution near $0$**: no clustering — tokens remain spread out.

# **The $\pm 1$ std band on the time-series plot above** summarises this same
# information over time. Concretely, at each time $t$ we compute
# $\langle x_i, x_j
# angle^2$ for all $binom{n}{2}$ pairs and take their mean and
# standard deviation:

# - **Mean near $1$, std near $0$**: one tight cluster (or antipodal pair) — all
#   pairs are equally clustered.
# - **Mean near $1$, std $> 0$**: multiple clusters — pairs within a cluster score
#   $1$, pairs across clusters score less, pulling the std up. A *wider* band means
#   *more* or *more separated* clusters.
# - **Mean near $0$, std near $0$**: no clustering, tokens uniformly spread.

# So the std is your **multi-cluster detector**: it is near $0$ for both the
# fully-clustered and fully-spread cases, and large only when tokens have
# split into geometrically distinct groups.
#     """)

with st.expander("Cosine similarity matrices  (t=0 and t=T, cluster-sorted)"):
    # Cluster assignment from t=T: project onto dominant direction, sort by sign
    X_final = X_traj[:, :, -1]
    _, _, Vt = np.linalg.svd(X_final, full_matrices=False)
    dominant    = Vt[0]
    projections = X_final @ dominant          # signed projection of each token
    sort_order  = np.argsort(-projections)    # descending: cluster+ first, cluster- last
    n_plus      = int((projections >= 0).sum())

    fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5))
    for ax_idx, (t_idx, t_label) in enumerate([(0, "t=0"), (-1, f"t={T}")]):
        Xt = X_traj[:, :, t_idx]
        # Use raw cosine similarities <x_i, x_j> — informative regardless of attention type
        G  = Xt @ Xt.T                                           # (n,n) Gram matrix
        # Apply cluster-sorted permutation (derived from t=T) to BOTH plots
        score_mat = G[np.ix_(sort_order, sort_order)]

        mat_std  = score_mat.std()
        mat_mean = score_mat.mean()
        uniformity = mat_std / (abs(mat_mean) + 1e-12)

        if uniformity < 1e-3:
            # Essentially uniform — fill with solid red to signal single-cluster collapse.
            # Using a 1-colour custom colormap avoids viridis autoscaling noise artefacts.
            from matplotlib.colors import LinearSegmentedColormap
            red_cmap = LinearSegmentedColormap.from_list("solid_red", ["#8D0226", "#5C001F"])
            d_val = abs(mat_mean) * 0.01 + 1e-9
            im = axes3[ax_idx].imshow(score_mat, cmap=red_cmap, aspect="auto",
                                      vmin=mat_mean - d_val, vmax=mat_mean + d_val)
            axes3[ax_idx].set_title(
                f"<x_i, x_j> ({t_label})\n[★ single cluster — all entries ≈ {mat_mean:.4f}]"
            )    
        else:
            # RdBu_r: red=+1 (same cluster), white=0 (orthogonal), blue=-1 (antipodal)
            im = axes3[ax_idx].imshow(score_mat, cmap="RdBu_r", aspect="auto",
                                      vmin=-1, vmax=1)
            axes3[ax_idx].set_title(f"<x_i, x_j> ({t_label}, cluster-sorted)")
            if n_tokens <= 25:
                for row in range(n_tokens):
                    for col in range(n_tokens):
                        v = score_mat[row, col]
                        axes3[ax_idx].text(col, row, f"{v:.2f}", ha="center", va="center",
                                           fontsize=4.0, color="black" if abs(v) < 0.5 else "white")

        # Draw separator line between the two groups
        if 0 < n_plus < n_tokens:
            axes3[ax_idx].axhline(n_plus - 0.5, color="gold", lw=1.5, linestyle="--")
            axes3[ax_idx].axvline(n_plus - 0.5, color="gold", lw=1.5, linestyle="--")

        axes3[ax_idx].set_xlabel("token j (sorted by cluster)")
        axes3[ax_idx].set_ylabel("token k (sorted by cluster)")
        plt.colorbar(im, ax=axes3[ax_idx])

    # Cluster size annotation
    # if 0 < n_plus < n_tokens:
    #     fig3.text(0.5, -0.02,
    #               f"Gold line separates cluster+ ({n_plus} tokens) | cluster- ({n_tokens-n_plus} tokens). "
    #               f"Sorting derived from t=T positions, applied to both plots.",
    #               ha="center", fontsize=8, style="italic")
    plt.tight_layout()
    st.pyplot(fig3)
    st.caption(
        "Colormap: red = +1 (same direction), white = 0 (orthogonal), blue = -1 (antipodal). "
        "At t=T: red blocks on diagonal = within-cluster similarity, blue off-diagonal = cross-cluster antipodal pairs."
    )


with st.expander("Token positions at t=0 and t=T"):
    col1, col2 = st.columns(2)
    with col1:
        st.write("**t=0**")
        st.dataframe(pd.DataFrame(np.round(X_traj[:, :, 0].astype(float), 4)))
    with col2:
        st.write(f"**t={T}**")
        st.dataframe(pd.DataFrame(np.round(X_traj[:, :, -1].astype(float), 4)))
    st.write(f"Trajectory shape: `(n_tokens={n_tokens}, d={d}, frames={frames})`")
