"""Multi-determinant VMC model for the Be atom."""

from functools import partial
import time

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import flax.linen as nn
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

# Constants and reference energies.
Z_NUC = 4.0
N_ELEC = 4
N_DIM = 3 * N_ELEC

B_MIN, B_MAX = 0.05, 10.0                      # Pade parameter bounds

_i, _j = np.triu_indices(N_ELEC, k=1)          # Six unique electron pairs.
PAIR_I = jnp.array(_i)
PAIR_J = jnp.array(_j)

SPINS = np.array([0, 0, 1, 1])                 # Electrons 0-1 up, 2-3 down.
_parallel = (SPINS[_i] == SPINS[_j])

# Electron-electron cusp coefficients.
A_CUSP = jnp.where(jnp.array(_parallel), 0.25, 0.50)
# Network spin feature: +1 for parallel, -1 for antiparallel.
SPIN_F = jnp.where(jnp.array(_parallel), 1.0, -1.0)

E_HF = -14.5730
E_FN_SD = -14.6570
E_EXACT = -14.6674

COL_W = 3.375        # REVTeX single-column width, inches (246 pt)
FULL_W = 7.00        # REVTeX full text width, inches (510 pt)


# Helpers.
def norm_stable(x):
    return jnp.sqrt(jnp.sum(x ** 2, axis=-1) + 1e-12)


def t_cusp(r):
    """Return a smooth coordinate with zero slope at electron coalescence."""
    return r * r / (1.0 + r)


# Three-body neural Jastrow.
class NeuralJastrow3Body(nn.Module):
    hidden_dim: int = 16
    envelope_alpha: float = 0.3

    @nn.compact
    def __call__(self, feats):
        # The first three features are symmetric under i <-> j.
        x = feats
        for _ in range(2):
            # Keep the network parameters in double precision.
            x = nn.Dense(self.hidden_dim, param_dtype=jnp.float64)(x)
            x = nn.silu(x)
        # Start near the baseline without disabling gradients.
        x = nn.Dense(1, param_dtype=jnp.float64,
                     kernel_init=nn.initializers.normal(1e-2),
                     bias_init=nn.initializers.zeros)(x)

        env = jnp.exp(-self.envelope_alpha * (feats[0] + feats[2]))
        return x.squeeze(-1) * env


def init_params(key, mlp):
    p_mlp = mlp.init(key, jnp.ones((4,), dtype=jnp.float64))
    p_orb = {
        "a2s":    jnp.asarray(0.98, dtype=jnp.float64),
        "a2p":    jnp.asarray(1.00, dtype=jnp.float64),
        "b_par":  jnp.asarray(0.50, dtype=jnp.float64),
        "b_anti": jnp.asarray(0.50, dtype=jnp.float64),
        "cp":     jnp.asarray(0.00, dtype=jnp.float64),   # Start from one determinant.
    }
    return {"mlp": p_mlp, "orb": p_orb}


# Wave function.
def make_log_psi(mlp):

    def log_psi(params, R):
        po = params["orb"]

        # Keep variational exponents in a stable range.
        a1s = Z_NUC                                    # Fixed by the cusp.
        a2s = jnp.clip(po["a2s"], 0.30, 3.00)          # must stay below Z
        a2p = jnp.clip(po["a2p"], 0.30, 3.00)
        b_par = jnp.clip(po["b_par"], B_MIN, B_MAX)
        b_anti = jnp.clip(po["b_anti"], B_MIN, B_MAX)

        r = norm_stable(R)                             # (4,)

        # The 2s prefactor enforces the electron-nucleus cusp.
        # The 2p orbitals are orthogonal by angular symmetry.
        c = 1.0 / (Z_NUC - a2s)
        phi_1s = jnp.exp(-a1s * r)                             # (4,)
        phi_2s = (r - c) * jnp.exp(-a2s * r)                   # (4,)
        phi_2p = R * jnp.exp(-a2p * r)[:, None]                # (4,3) x,y,z

        def det_pair(orb):
            """Return the product of the spin-up and spin-down determinants."""
            d_up = phi_1s[0] * orb[1] - orb[0] * phi_1s[1]
            d_dn = phi_1s[2] * orb[3] - orb[2] * phi_1s[3]
            return d_up * d_dn

        # Equal p-orbital weights preserve rotational symmetry.
        psi_p = (det_pair(phi_2p[:, 0]) + det_pair(phi_2p[:, 1])
                 + det_pair(phi_2p[:, 2]))
        psi_det = det_pair(phi_2s) + po["cp"] * psi_p
        log_det = jnp.log(jnp.abs(psi_det) + 1e-300)

        # Analytic Jastrow with exact electron-electron cusps.
        r_ij = norm_stable(R[PAIR_I] - R[PAIR_J])
        b = jnp.where(SPIN_F > 0.0, b_par, b_anti)
        u_pade = jnp.sum(A_CUSP * r_ij / (1.0 + b * r_ij))

        # Three-body neural correction.
        t_i = t_cusp(r[PAIR_I])
        t_j = t_cusp(r[PAIR_J])
        t_ij = t_cusp(r_ij)
        feats = jnp.stack([t_i + t_j, t_i * t_j, t_ij, SPIN_F], axis=-1)   # (6,4)
        u_nn = jnp.sum(jax.vmap(lambda f: mlp.apply(params["mlp"], f))(feats))

        return log_det + u_pade + u_nn

    return log_psi


# Local energy and kinetic/potential components.
def make_energy_fns(log_psi_fn):

    def kin_pot(params, R):
        x = R.reshape(-1)
        f = lambda z: log_psi_fn(params, z.reshape(N_ELEC, 3))
        grad_fn = jax.grad(f)
        g = grad_fn(x)

        # Compute the Laplacian with JVPs instead of forming a full Hessian.
        def body(acc, i):
            e = jnp.zeros_like(x).at[i].set(1.0)
            return acc + jax.jvp(grad_fn, (x,), (e,))[1][i], None

        lap, _ = jax.lax.scan(body, jnp.array(0.0, dtype=x.dtype),
                              jnp.arange(N_DIM))
        T = -0.5 * (lap + jnp.sum(g ** 2))
        V = (-Z_NUC * jnp.sum(1.0 / norm_stable(R))
             + jnp.sum(1.0 / norm_stable(R[PAIR_I] - R[PAIR_J])))
        return T, V

    def local_energy(params, R):
        T, V = kin_pot(params, R)
        return T + V

    return local_energy, kin_pot


# MCMC with single-electron moves.
def make_mcmc(log_psi_fn):

    def sweeps(key, R, params, step, n_sweeps):
        logp0 = 2.0 * log_psi_fn(params, R)

        def one_move(carry, inp):
            R, logp = carry
            i, k = inp
            k1, k2 = jax.random.split(k)
            R_new = R.at[i].add(step * jax.random.normal(k1, (3,), dtype=R.dtype))
            logp_new = 2.0 * log_psi_fn(params, R_new)
            acc = jnp.log(jax.random.uniform(k2, dtype=R.dtype)) < (logp_new - logp)
            return (jnp.where(acc, R_new, R), jnp.where(acc, logp_new, logp)), acc

        keys = jax.random.split(key, n_sweeps * N_ELEC)
        idx = jnp.tile(jnp.arange(N_ELEC), n_sweeps)
        (R, _), accs = jax.lax.scan(one_move, (R, logp0), (idx, keys))
        return R, jnp.mean(accs)

    @partial(jax.jit, static_argnames=("n_sweeps",))
    def batch_sweeps(keys, R_batch, params, step, n_sweeps):
        R, a = jax.vmap(lambda k, r: sweeps(k, r, params, step, n_sweeps))(keys, R_batch)
        return R, jnp.mean(a)

    return batch_sweeps


def adapt_step(step, acc, target=0.55):
    """Adjust the move length toward the target acceptance rate."""
    if acc > target + 0.05:
        step *= 1.05
    elif acc < target - 0.05:
        step /= 1.05
    return float(np.clip(step, 0.02, 3.0))


# Energy, score matrix, and stochastic reconfiguration.
def make_trainer(log_psi_fn, unravel_fn):
    local_energy, kin_pot = make_energy_fns(log_psi_fn)

    @jax.jit
    def energy_and_scores(params_flat, R_batch):
        params = unravel_fn(params_flat)
        eloc = jax.vmap(lambda r: local_energy(params, r))(R_batch)
        score = lambda r: jax.grad(lambda pf: log_psi_fn(unravel_fn(pf), r))(params_flat)
        return eloc, jax.vmap(score)(R_batch)

    @jax.jit
    def virial(params_flat, R_batch):
        params = unravel_fn(params_flat)
        T, V = jax.vmap(lambda r: kin_pot(params, r))(R_batch)
        return jnp.mean(T), jnp.mean(V)

    @jax.jit
    def sr_update(params_flat, O, eloc, lr, gamma, max_norm):
        # Limit the influence of energetic outliers.
        med = jnp.median(eloc)
        mad = jnp.mean(jnp.abs(eloc - med)) + 1e-9
        e = jnp.clip(eloc, med - 5.0 * mad, med + 5.0 * mad)
        e_c = e - jnp.mean(e)

        O_c = O - jnp.mean(O, axis=0, keepdims=True)
        g = 2.0 * jnp.mean(O_c * e_c[:, None], axis=0)

        S = (O_c.T @ O_c) / O.shape[0]
        # Scale damping by each parameter block's diagonal.
        S_reg = S + jnp.diag(gamma * jnp.diag(S) + 1e-8)
        delta = jnp.linalg.solve(S_reg, g)

        # Apply the trust region to the final update.
        update = lr * delta
        update = update * jnp.minimum(1.0, max_norm / (jnp.linalg.norm(update) + 1e-12))
        return params_flat - update, jnp.linalg.norm(update)   # POST-clip norm

    return energy_and_scores, sr_update, virial


def blocking_error(x):
    """Estimate the error using successive blocking."""
    y = np.asarray(x, dtype=float)
    errs = []
    while len(y) >= 8:
        errs.append(y.std(ddof=1) / np.sqrt(len(y)))
        y = y[:len(y) // 2 * 2].reshape(-1, 2).mean(axis=1)
    return max(errs)


# Figures.
def setup_plot_style():
    """Set typography for the paper figures."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.5,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.borderpad": 0.35,
        "legend.labelspacing": 0.3,
        "legend.handlelength": 1.6,
        "legend.handletextpad": 0.5,
        "lines.linewidth": 0.9,
        "axes.linewidth": 0.6,
        "grid.linewidth": 0.35,
        "grid.alpha": 0.3,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "figure.dpi": 300,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def _save(fig, fname):
    """Write both a vector PDF (for LaTeX) and a raster PNG (for previewing)."""
    stem = fname.rsplit(".", 1)[0]
    fig.savefig(f"{stem}.pdf")
    fig.savefig(f"{stem}.png")
    print(f"[saved] {stem}.pdf / {stem}.png")


def plot_multidet(hist_e, hist_err, hist_var, cp_track=None,
                  fname="multidet_optimisation.pdf"):
    """Plot energy, variance, and the optional mixing coefficient."""
    n = len(hist_e)
    ep = np.arange(1, n + 1)
    e = np.asarray(hist_e)
    err = np.asarray(hist_err)

    if cp_track is None:
        fig, (a1, a2) = plt.subplots(
            2, 1, figsize=(COL_W, 3.9), sharex=True,
            gridspec_kw={"height_ratios": [3, 2], "hspace": 0.08})
        axes = (a1, a2)
    else:
        fig, (a1, a2, a3) = plt.subplots(
            3, 1, figsize=(COL_W, 4.8), sharex=True,
            gridspec_kw={"height_ratios": [3, 2, 1.4], "hspace": 0.08})
        axes = (a1, a2, a3)

    # Energy.
    a1.axhline(E_HF, color="black", ls="--", lw=0.7)
    a1.axhline(E_FN_SD, color="crimson", ls="--", lw=0.7)
    a1.axhline(E_EXACT, color="black", ls=":", lw=0.9)

    a1.fill_between(ep, e - err, e + err, color="#2ca02c", alpha=0.25, lw=0)
    a1.plot(ep, e, color="#2ca02c", lw=0.7, label="(D+)")

    for y, txt, col in ((E_HF, " HF", "black"),
                        (E_FN_SD, " FN-DMC, HF nodes", "crimson"),
                        (E_EXACT, " exact", "black")):
        a1.text(0.985, y, txt, transform=a1.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=6, color=col)

    a1.set_ylabel(r"$E$  ($E_h$)")
    a1.set_ylim(E_EXACT - 0.012, -14.55)
    a1.legend(loc="lower right")
    a1.grid(True)

    # Variance.
    a2.semilogy(ep, hist_var, color="#1f77b4", lw=0.7)
    a2.set_ylabel(r"Var($E_L$)")
    a2.grid(True, which="both")

    # Optional mixing coefficient.
    if cp_track is not None:
        a3.plot(ep, cp_track, color="#7f3fbf", lw=0.8)
        a3.set_ylabel(r"$c_p$")
        a3.grid(True)

    axes[-1].set_xlabel("Epoch")

    _save(fig, fname)


# Training and production pipeline.
def run_pipeline(n_chains=4000,
                 n_epochs=3500,
                 n_sweeps=10,
                 hidden_dim=16,
                 lr0=0.10,
                 gamma=5e-3,
                 max_norm=0.30,
                 seed=42):

    key = jax.random.PRNGKey(seed)
    setup_plot_style()

    mlp = NeuralJastrow3Body(hidden_dim=hidden_dim)
    key, sk = jax.random.split(key)
    params = init_params(sk, mlp)
    params_flat, unravel_fn = ravel_pytree(params)
    n_par = params_flat.size

    print(f"Variational parameters : {n_par}")
    print(f"Walkers                : {n_chains}  (ratio {n_chains/n_par:.1f}x)")
    if n_chains < 4 * n_par:
        print("  WARNING: SR needs roughly 4-10 samples per parameter.")
    print(f"Parameter dtype        : {params_flat.dtype}\n")

    log_psi = make_log_psi(mlp)
    mcmc = make_mcmc(log_psi)
    energy_and_scores, sr_update, virial = make_trainer(log_psi, unravel_fn)

    # Start core electrons near the nucleus and valence electrons farther out.
    key, sk = jax.random.split(key)
    scale = jnp.array([0.4, 0.4, 1.8, 1.8])[None, :, None]
    R = jax.random.normal(sk, (n_chains, N_ELEC, 3), dtype=jnp.float64) * scale

    step = 0.35

    # Burn-in.
    print("=== MCMC burn-in ===")
    t0 = time.time()
    for _ in range(15):
        key, sk = jax.random.split(key)
        p = unravel_fn(params_flat)
        R, acc = mcmc(jax.random.split(sk, n_chains), R, p, step, 20)
        step = adapt_step(step, float(acc))
    print(f"acceptance = {float(acc)*100:.1f}%  step = {step:.3f}  "
          f"({time.time()-t0:.1f}s)\n")

    # Training.
    print("=== Training (multi-determinant + three-body neural Jastrow, SR) ===")
    hist_e, hist_err, hist_var, hist_cp = [], [], [], []
    t0 = time.time()

    for epoch in range(n_epochs):
        key, sk = jax.random.split(key)
        p = unravel_fn(params_flat)
        R, acc = mcmc(jax.random.split(sk, n_chains), R, p, step, n_sweeps)
        step = adapt_step(step, float(acc))

        eloc, O = energy_and_scores(params_flat, R)
        e_mean = float(jnp.mean(eloc))
        e_var = float(jnp.var(eloc))
        e_err = float(jnp.std(eloc) / jnp.sqrt(n_chains))

        hist_e.append(e_mean)
        hist_err.append(e_err)
        hist_var.append(e_var)

        # Decay the learning rate and trust-region radius together.
        d = 0.05 + 0.95 * 0.5 * (1.0 + np.cos(np.pi * epoch / n_epochs))
        params_flat, upd_norm = sr_update(params_flat, O, eloc,
                                          lr0 * d, gamma, max_norm * d)

        po = unravel_fn(params_flat)["orb"]
        hist_cp.append(float(po["cp"]))

        if epoch == 0 or (epoch + 1) % 100 == 0:
            print(f"ep {epoch+1:5d} | E = {e_mean:10.5f} +/- {e_err:.5f} "
                  f"| var = {e_var:6.3f} | |upd| = {float(upd_norm):.4f} "
                  f"| cp = {float(po['cp']):+.3f} a2s = {float(po['a2s']):.3f} "
                  f"| acc = {float(acc)*100:.0f}%")

    print(f"\nTraining in {time.time()-t0:.1f}s")

    # Production sampling.
    print("\n=== Final estimate (40 decorrelated blocks, 150 sweeps apart) ===")
    p = unravel_fn(params_flat)
    blocks, vars_ = [], []
    for _ in range(40):
        key, sk = jax.random.split(key)
        R, _ = mcmc(jax.random.split(sk, n_chains), R, p, step, 150)
        eloc, _ = energy_and_scores(params_flat, R)
        blocks.append(float(jnp.mean(eloc)))
        vars_.append(float(jnp.var(eloc)))

    E = float(np.mean(blocks))
    dE = blocking_error(blocks)
    var = float(np.mean(vars_))
    T, V = virial(params_flat, R)
    T, V = float(T), float(V)

    corr = (E - E_HF) / (E_EXACT - E_HF) * 100.0
    po = unravel_fn(params_flat)["orb"]

    print(f"E_VMC = {E:.5f} +/- {dE:.5f} Eh")
    print(f"Var(E_loc)                     : {var:.4f}")
    print(f"Correlation energy recovered   : {corr:.1f}%")
    print(f"Distance from the exact value  : {E - E_EXACT:+.5f} Eh")
    print(f"Virial ratio V/T               : {V/T:+.4f}   (exact -2)")
    print(f"  <T> = {T:.5f}   <V> = {V:.5f}")
    print("\nOptimised parameters:")
    print(f"  c_p    = {float(po['cp']):+.4f}   (2p^2 weight: moves the nodes)")
    print(f"  a2s    = {float(po['a2s']):.4f}")
    print(f"  a2p    = {float(po['a2p']):.4f}")
    bp = float(np.clip(po["b_par"], B_MIN, B_MAX))
    ba = float(np.clip(po["b_anti"], B_MIN, B_MAX))
    print(f"  b_par  = {bp:.4f}"
          + ("   <-- AT BOUND" if bp >= B_MAX - 1e-6 else ""))
    print(f"  b_anti = {ba:.4f}"
          + ("   <-- AT BOUND" if ba >= B_MAX - 1e-6 else ""))

    # Save diagnostics and results.
    plot_multidet(hist_e, hist_err, hist_var, cp_track=hist_cp)

    np.savez("multidet_results.npz",
             E=E, dE=dE, var=var, T=T, V=V, n_par=n_par,
             cp=float(po["cp"]), a2s=float(po["a2s"]), a2p=float(po["a2p"]),
             b_par=bp, b_anti=ba,
             hist_e=np.array(hist_e), hist_var=np.array(hist_var),
             hist_err=np.array(hist_err), hist_cp=np.array(hist_cp),
             blocks=np.array(blocks))
    print("[saved] multidet_results.npz")

    plt.show()
    return params_flat, unravel_fn


if __name__ == "__main__":
    run_pipeline()