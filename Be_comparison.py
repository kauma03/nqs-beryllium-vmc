"""Compare four fixed-node VMC ansatze for the Be atom."""

from functools import partial
import time

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import flax.linen as nn
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

jax.config.update("jax_enable_x64", True)

# Constants and reference energies.
Z_NUC = 4.0
N_ELEC = 4
N_DIM = 3 * N_ELEC

B_MIN, B_MAX = 0.05, 10.0                         # Pade parameter bounds

_i, _j = np.triu_indices(N_ELEC, k=1)
PAIR_I, PAIR_J = jnp.array(_i), jnp.array(_j)

SPINS = np.array([0, 0, 1, 1])                    # Electrons 0-1 up, 2-3 down.
_par = (SPINS[_i] == SPINS[_j])
A_CUSP = jnp.where(jnp.array(_par), 0.25, 0.50)   # Electron-electron cusps.
SPIN_F = jnp.where(jnp.array(_par), 1.0, -1.0)    # Network spin feature.

E_HF = -14.5730
E_EXACT = -14.6674

MODELS = ("hf", "pade", "nqs", "hyb")
COLORS = {"hf": "#888888", "pade": "#1f77b4", "nqs": "#d62728", "hyb": "#2ca02c"}
LABELS = {"hf": "(A) Determinant only",
          "pade": "(B) Slater-Jastrow (Padé)",
          "nqs": "(C) Slater-NQS  $J_{NN}(r_{ij})$",
          "hyb": "(D) Padé + NN correction"}
# Short labels for compact figures.
SHORT = {"hf": "(A)", "pade": "(B)", "nqs": "(C)", "hyb": "(D)"}

COL_W = 3.375        # REVTeX single-column width, inches (246 pt)
FULL_W = 7.00        # REVTeX full text width, inches (510 pt)


def norm_stable(x):
    return jnp.sqrt(jnp.sum(x ** 2, axis=-1) + 1e-12)


def t_cusp(r):
    """Return a coordinate with zero slope at electron coalescence."""
    return r * r / (1.0 + r)


# Jastrow network.
class JastrowNet(nn.Module):
    """Network shared by all electron pairs."""
    hidden_dim: int = 16

    @nn.compact
    def __call__(self, feats):                    # Features are [distance, spin].
        x = feats
        for _ in range(2):
            # Keep network parameters in double precision.
            x = nn.Dense(self.hidden_dim, param_dtype=jnp.float64)(x)
            x = nn.silu(x)
        x = nn.Dense(1, param_dtype=jnp.float64,
                     kernel_init=nn.initializers.normal(1e-2),
                     bias_init=nn.initializers.zeros)(x)
        return x.squeeze(-1)


# Wave function variants.
def init_params(key, model, mlp):
    orb = {"a2s": jnp.asarray(0.98, dtype=jnp.float64),
           "p1":  jnp.asarray(0.00, dtype=jnp.float64),
           "p2":  jnp.asarray(0.00, dtype=jnp.float64)}
    p = {"orb": orb}
    if model in ("pade", "hyb"):
        p["jas"] = {"b_par":  jnp.asarray(0.50, dtype=jnp.float64),
                    "b_anti": jnp.asarray(0.50, dtype=jnp.float64)}
    if model in ("nqs", "hyb"):
        p["mlp"] = mlp.init(key, jnp.ones((2,), dtype=jnp.float64))
    return p


def pair_feature(model, r_ij):
    """Return the network coordinate for the selected model."""
    if model == "nqs":
        return r_ij / (1.0 + r_ij)
    s = t_cusp(r_ij)
    return s / (1.0 + s)


def make_log_psi(model, mlp=None):

    def log_psi(params, R):
        po = params["orb"]
        a2s = jnp.clip(po["a2s"], 0.30, 3.00)     # Keep the exponent below Z.
        p1 = jnp.clip(po["p1"], -0.15, 1.50)
        p2 = jnp.clip(po["p2"], -0.15, 1.50)

        r = norm_stable(R)
        t = t_cusp(r)

        # Orbital factors preserve the electron-nucleus cusp.
        c = 1.0 / (Z_NUC - a2s)
        phi_1s = jnp.exp(-Z_NUC * r) * (1.0 + p1 * t)
        phi_2s = (r - c) * jnp.exp(-a2s * r) * (1.0 + p2 * t)

        d_up = phi_1s[0] * phi_2s[1] - phi_2s[0] * phi_1s[1]
        d_dn = phi_1s[2] * phi_2s[3] - phi_2s[2] * phi_1s[3]
        log_det = jnp.log(jnp.abs(d_up) + 1e-300) + jnp.log(jnp.abs(d_dn) + 1e-300)

        if model == "hf":
            return log_det

        r_ij = norm_stable(R[PAIR_I] - R[PAIR_J])
        J = 0.0

        if model in ("pade", "hyb"):
            pj = params["jas"]
            b = jnp.where(SPIN_F > 0.0,
                          jnp.clip(pj["b_par"],  B_MIN, B_MAX),
                          jnp.clip(pj["b_anti"], B_MIN, B_MAX))
            J = J + jnp.sum(A_CUSP * r_ij / (1.0 + b * r_ij))

        if model in ("nqs", "hyb"):
            feats = jnp.stack([pair_feature(model, r_ij), SPIN_F], axis=-1)
            J = J + jnp.sum(jax.vmap(lambda f: mlp.apply(params["mlp"], f))(feats))

        return log_det + J

    return log_psi


# Local energy and kinetic/potential components.
def make_energy_fns(log_psi_fn):

    def kin_pot(params, R):
        x = R.reshape(-1)
        f = lambda z: log_psi_fn(params, z.reshape(N_ELEC, 3))
        grad_fn = jax.grad(f)
        g = grad_fn(x)

        def body(acc, i):                          # Accumulate the Laplacian.
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

        def move(carry, inp):
            R, logp = carry
            i, k = inp
            k1, k2 = jax.random.split(k)
            R_new = R.at[i].add(step * jax.random.normal(k1, (3,), dtype=R.dtype))
            logp_new = 2.0 * log_psi_fn(params, R_new)
            acc = jnp.log(jax.random.uniform(k2, dtype=R.dtype)) < (logp_new - logp)
            return (jnp.where(acc, R_new, R), jnp.where(acc, logp_new, logp)), acc

        keys = jax.random.split(key, n_sweeps * N_ELEC)
        idx = jnp.tile(jnp.arange(N_ELEC), n_sweeps)
        (R, _), accs = jax.lax.scan(move, (R, logp0), (idx, keys))
        return R, jnp.mean(accs)

    @partial(jax.jit, static_argnames=("n_sweeps",))
    def batch(keys, R_batch, params, step, n_sweeps):
        R, a = jax.vmap(lambda k, r: sweeps(k, r, params, step, n_sweeps))(keys, R_batch)
        return R, jnp.mean(a)

    return batch


def adapt_step(step, acc, target=0.55):
    if acc > target + 0.05:
        step *= 1.05
    elif acc < target - 0.05:
        step /= 1.05
    return float(np.clip(step, 0.02, 3.0))


# Stochastic reconfiguration.
def make_trainer(log_psi_fn, unravel_fn):
    local_energy, kin_pot = make_energy_fns(log_psi_fn)

    @jax.jit
    def energy_and_scores(pf, R_batch):
        params = unravel_fn(pf)
        eloc = jax.vmap(lambda r: local_energy(params, r))(R_batch)
        score = lambda r: jax.grad(lambda q: log_psi_fn(unravel_fn(q), r))(pf)
        return eloc, jax.vmap(score)(R_batch)

    @jax.jit
    def virial(pf, R_batch):
        params = unravel_fn(pf)
        T, V = jax.vmap(lambda r: kin_pot(params, r))(R_batch)
        return jnp.mean(T), jnp.mean(V)

    @jax.jit
    def sr_update(pf, O, eloc, lr, gamma, max_norm):
        med = jnp.median(eloc)
        mad = jnp.mean(jnp.abs(eloc - med)) + 1e-9
        e = jnp.clip(eloc, med - 5.0 * mad, med + 5.0 * mad)
        e_c = e - jnp.mean(e)

        O_c = O - jnp.mean(O, axis=0, keepdims=True)
        g = 2.0 * jnp.mean(O_c * e_c[:, None], axis=0)

        S = (O_c.T @ O_c) / O.shape[0]
        # Scale damping by the diagonal of the covariance matrix.
        S_reg = S + jnp.diag(gamma * jnp.diag(S) + 1e-8)
        delta = jnp.linalg.solve(S_reg, g)

        update = lr * delta            # Apply the trust region to the final update.
        update = update * jnp.minimum(1.0, max_norm / (jnp.linalg.norm(update) + 1e-12))
        return pf - update, jnp.linalg.norm(update)     # Return the clipped norm.

    return energy_and_scores, sr_update, virial


def blocking_error(x):
    """Estimate the error using successive blocking."""
    y = np.asarray(x, dtype=float)
    errs = []
    while len(y) >= 8:
        errs.append(y.std(ddof=1) / np.sqrt(len(y)))
        y = y[:len(y) // 2 * 2].reshape(-1, 2).mean(axis=1)
    return max(errs)


# Optimisation of one model.
def optimize(model, key, n_chains, n_epochs, lr0, max_norm,
             gamma=3e-3, n_sweeps=10, hidden_dim=16, verbose_every=100):

    mlp = JastrowNet(hidden_dim=hidden_dim) if model in ("nqs", "hyb") else None

    key, sk = jax.random.split(key)
    params = init_params(sk, model, mlp)
    pf, unravel = ravel_pytree(params)

    log_psi = make_log_psi(model, mlp)
    mcmc = make_mcmc(log_psi)
    energy_and_scores, sr_update, virial = make_trainer(log_psi, unravel)

    key, sk = jax.random.split(key)
    scale = jnp.array([0.4, 0.4, 1.8, 1.8])[None, :, None]
    R = jax.random.normal(sk, (n_chains, N_ELEC, 3), dtype=jnp.float64) * scale
    step = 0.35

    print(f"\n{'='*70}\n  {LABELS[model]}   |   {pf.size} parameters\n{'='*70}")

    for _ in range(15):                                    # Burn-in.
        key, sk = jax.random.split(key)
        R, acc = mcmc(jax.random.split(sk, n_chains), R, unravel(pf), step, 20)
        step = adapt_step(step, float(acc))

    hist_e, hist_var = [], []
    t0 = time.time()

    for ep in range(n_epochs):
        key, sk = jax.random.split(key)
        R, acc = mcmc(jax.random.split(sk, n_chains), R, unravel(pf), step, n_sweeps)
        step = adapt_step(step, float(acc))

        eloc, O = energy_and_scores(pf, R)
        hist_e.append(float(jnp.mean(eloc)))
        hist_var.append(float(jnp.var(eloc)))

        # Decay the learning rate and trust-region radius together.
        d = 0.05 + 0.95 * 0.5 * (1.0 + np.cos(np.pi * ep / n_epochs))
        pf, un = sr_update(pf, O, eloc, lr0 * d, gamma, max_norm * d)

        if ep == 0 or (ep + 1) % verbose_every == 0:
            print(f"  ep {ep+1:5d} | E = {hist_e[-1]:10.5f} | "
                  f"var = {hist_var[-1]:7.4f} | |upd| = {float(un):.4f} | "
                  f"acc = {float(acc)*100:.0f}%")

    print(f"  training in {time.time()-t0:.1f}s")

    # Final estimate from decorrelated blocks.
    blocks, vars_ = [], []
    for _ in range(40):
        key, sk = jax.random.split(key)
        R, _ = mcmc(jax.random.split(sk, n_chains), R, unravel(pf), step, 150)
        eloc, _ = energy_and_scores(pf, R)
        blocks.append(float(jnp.mean(eloc)))
        vars_.append(float(jnp.var(eloc)))

    E = float(np.mean(blocks))
    dE = blocking_error(blocks)
    var = float(np.mean(vars_))
    T, V = virial(pf, R)
    T, V = float(T), float(V)
    final = unravel(pf)

    print(f"  E = {E:.5f} +/- {dE:.5f} Eh | var = {var:.4f} | "
          f"V/T = {V/T:+.4f} (exact -2)")
    print(f"  orbitals: a2s = {float(final['orb']['a2s']):.4f}  "
          f"p1 = {float(final['orb']['p1']):+.4f}  "
          f"p2 = {float(final['orb']['p2']):+.4f}")
    if "jas" in final:
        bp = float(np.clip(final["jas"]["b_par"], B_MIN, B_MAX))
        ba = float(np.clip(final["jas"]["b_anti"], B_MIN, B_MAX))
        flag = "  <-- AT BOUND, Jastrow effectively off" if bp >= B_MAX - 1e-6 else ""
        print(f"  Padé: b_par = {bp:.4f}{flag}")
        print(f"        b_anti = {ba:.4f}"
              + ("  <-- AT BOUND" if ba >= B_MAX - 1e-6 else ""))

    return {"model": model, "E": E, "dE": dE, "var": var, "R": R,
            "hist_e": hist_e, "hist_var": hist_var,
            "params": final, "mlp": mlp,
            "T": T, "V": V, "n_par": int(pf.size)}


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
    """Save both PDF and PNG versions of a figure."""
    stem = fname.rsplit(".", 1)[0]
    fig.savefig(f"{stem}.pdf")
    fig.savefig(f"{stem}.png")
    print(f"[saved] {stem}.pdf / {stem}.png")


def plot_convergence(res, fname="comparison_energies.pdf"):
    """Plot energy and variance during optimisation."""
    fig, (a1, a2) = plt.subplots(
        2, 1, figsize=(COL_W, 3.9), sharex=True,
        gridspec_kw={"height_ratios": [3, 2], "hspace": 0.08})

    a1.axhline(E_HF, color="black", ls="--", lw=0.7)
    a1.axhline(E_EXACT, color="black", ls=":", lw=0.9)
    a1.text(0.985, E_HF, " HF", transform=a1.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=6)
    a1.text(0.985, E_EXACT, " exact", transform=a1.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=6)

    for r in res:
        m = r["model"]
        a1.plot(np.arange(1, len(r["hist_e"]) + 1), r["hist_e"],
                color=COLORS[m], lw=0.7,
                label=f"{SHORT[m]}  {r['n_par']} par.")

    a1.set_ylabel(r"$E$  ($E_h$)")
    a1.set_ylim(E_EXACT - 0.015, -14.42)
    a1.legend(loc="upper right", ncol=2, columnspacing=0.8)
    a1.grid(True)

    for r in res:
        a2.semilogy(np.arange(1, len(r["hist_var"]) + 1), r["hist_var"],
                    color=COLORS[r["model"]], lw=0.7)
    a2.set_xlabel("Epoch")
    a2.set_ylabel(r"Var($E_L$)")
    a2.grid(True, which="both")

    _save(fig, fname)


def plot_jastrow(res, fname="comparison_jastrow.pdf"):
    """Plot the learned and analytic Jastrow terms."""
    by = {r["model"]: r for r in res}
    models = [m for m in ("pade", "nqs", "hyb") if m in by]
    if not models:
        return

    rg = jnp.linspace(0.0, 8.0, 400)
    rgn = np.asarray(rg)

    def J_of(entry, spin, rv):
        """Return the model Jastrow term with J(0) = 0."""
        m = entry["model"]
        rvn = np.asarray(rv)
        out = np.zeros_like(rvn)
        if m in ("pade", "hyb"):
            key = "b_par" if spin > 0 else "b_anti"
            b = float(np.clip(entry["params"]["jas"][key], B_MIN, B_MAX))
            A = 0.25 if spin > 0 else 0.50
            out = out + A * rvn / (1.0 + b * rvn)
        if m in ("nqs", "hyb"):
            mlp, pm = entry["mlp"], entry["params"]["mlp"]
            x = pair_feature(m, rv)
            f = jnp.stack([x, jnp.full_like(rv, spin)], axis=-1)
            u = np.asarray(jax.vmap(lambda z: mlp.apply(pm, z))(f))
            out = out + u - u[0]
        return out

    def slope0(entry, spin, h=1e-6):
        v = J_of(entry, spin, jnp.array([0.0, h]))
        return (v[1] - v[0]) / h

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FULL_W, 2.9))
    fig.subplots_adjust(wspace=0.28)

    # Full range with the sampled pair-distance distribution.
    src = by.get("hyb") or by.get("nqs") or by[models[0]]
    rij = np.asarray(jax.vmap(
        lambda r: norm_stable(r[PAIR_I] - r[PAIR_J]))(src["R"])).ravel()
    ah = a1.twinx()
    ah.hist(rij, bins=90, range=(0, 8), density=True,
            color="0.55", alpha=0.22, edgecolor="none")
    ah.set_ylabel(r"sampled $P(r_{ij})$", color="0.4")
    ah.set_ylim(0, None)
    ah.tick_params(axis="y", colors="0.4", labelsize=7)
    ah.spines["right"].set_color("0.4")
    ah.spines["right"].set_linewidth(0.6)

    for m in models:
        for spin, ls in ((-1.0, "-"), (+1.0, "--")):
            a1.plot(rgn, J_of(by[m], spin, rg),
                    color=COLORS[m], ls=ls, lw=1.1)

    # Colour identifies the model; line style identifies the spin channel.
    handles = [Line2D([], [], color=COLORS[m], lw=1.1, label=SHORT[m])
               for m in models]
    handles += [Line2D([], [], color="black", lw=1.1, ls="-", label="antipar."),
                Line2D([], [], color="black", lw=1.1, ls="--", label="parallel")]
    a1.legend(handles=handles, loc="upper left", ncol=2, columnspacing=0.8)

    a1.set_xlabel(r"$r_{ij}$  (bohr)")
    a1.set_ylabel(r"$J(r_{ij})$")
    a1.set_xlim(0, 8)
    a1.grid(True)
    a1.set_zorder(ah.get_zorder() + 1)
    a1.patch.set_visible(False)

    # Coalescence region.
    mz = rgn <= 1.0
    for A in (0.25, 0.50):
        a2.plot(rgn[mz], A * rgn[mz], color="black", ls=":", lw=0.8, zorder=1)
    for m in models:
        for spin, ls in ((-1.0, "-"), (+1.0, "--")):
            a2.plot(rgn[mz], J_of(by[m], spin, rg)[mz],
                    color=COLORS[m], ls=ls, lw=1.1, zorder=2)

    a2.annotate(r"$a_{ij}=1/2$", xy=(0.95, 0.478), fontsize=7,
                ha="right", va="bottom")
    a2.annotate(r"$a_{ij}=1/4$", xy=(0.95, 0.240), fontsize=7,
                ha="right", va="bottom")
    a2.set_xlabel(r"$r_{ij}$  (bohr)")
    a2.set_ylabel(r"$J(r_{ij})$")
    a2.set_xlim(0, 1.0)
    a2.grid(True)

    _save(fig, fname)

    print("\n--- Electron-electron cusp: dJ/dr at r = 0 ---")
    print(f"{'pair':<16}{'exact':>10}" + "".join(f"{m:>12}" for m in models))
    for spin, lab, A in ((+1.0, "parallel", 0.25),
                         (-1.0, "antiparallel", 0.50)):
        row = "".join(f"{slope0(by[m], spin):>12.4f}" for m in models)
        print(f"{lab:<16}{A:>10.4f}{row}")
    print("\n(D) must give exactly 0.2500 / 0.5000: a network acting on t(r)")
    print("cannot contribute any slope at r = 0. A different value means a bug.")


def plot_summary(res, fname="comparison_summary.pdf"):
    """Plot final energies with their estimated errors."""
    fig, ax = plt.subplots(figsize=(COL_W, 2.5))

    names = [SHORT[r["model"]] for r in res]
    E = [r["E"] for r in res]
    dE = [r["dE"] for r in res]

    ax.axhline(E_HF, color="black", ls="--", lw=0.7)
    ax.axhline(E_EXACT, color="black", ls=":", lw=0.9)

    ax.bar(names, E, yerr=dE, capsize=2.5, width=0.6,
           color=[COLORS[r["model"]] for r in res], alpha=0.9,
           error_kw={"elinewidth": 0.7, "capthick": 0.7})

    ax.set_ylim(E_EXACT - 0.012, -14.45)
    ax.set_ylabel(r"$E$  ($E_h$)")

    for i, r in enumerate(res):
        ax.text(i, r["E"] - r["dE"] - 0.0015, f"{r['E']:.4f}",
                ha="center", va="top", fontsize=6.5)

    ax.text(0.985, E_HF, " HF", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=6)
    ax.text(0.985, E_EXACT, " exact", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=6)

    ax.grid(True, axis="y")
    ax.set_axisbelow(True)

    _save(fig, fname)


def save_results(res, fname="comparison_results.npz"):
    """Store everything needed for the write-up without re-running anything."""
    out = {}
    for r in res:
        m = r["model"]
        out[f"{m}_E"] = r["E"]
        out[f"{m}_dE"] = r["dE"]
        out[f"{m}_var"] = r["var"]
        out[f"{m}_T"] = r["T"]
        out[f"{m}_V"] = r["V"]
        out[f"{m}_npar"] = r["n_par"]
        out[f"{m}_hist_e"] = np.array(r["hist_e"])
        out[f"{m}_hist_var"] = np.array(r["hist_var"])
        if "jas" in r["params"]:
            out[f"{m}_b_par"] = float(r["params"]["jas"]["b_par"])
            out[f"{m}_b_anti"] = float(r["params"]["jas"]["b_anti"])
    np.savez(fname, **out)
    print(f"[saved] {fname}")


# Main entry point.
def main(n_chains=4000, seed=0):
    key = jax.random.PRNGKey(seed)
    setup_plot_style()

    setup = {
        "hf":   dict(n_epochs=300,  lr0=0.05, max_norm=0.05, verbose_every=100),
        "pade": dict(n_epochs=600,  lr0=0.05, max_norm=0.05, verbose_every=200),
        "nqs":  dict(n_epochs=1500, lr0=0.10, max_norm=0.30, verbose_every=300),
        "hyb":  dict(n_epochs=1500, lr0=0.10, max_norm=0.30, verbose_every=300),
    }

    res = []
    for model in MODELS:
        key, sk = jax.random.split(key)
        res.append(optimize(model, sk, n_chains=n_chains, **setup[model]))

    # Summary table.
    print(f"\n{'='*80}\n  SUMMARY\n{'='*80}")
    print(f"{'model':<32}{'par.':>6}{'E (Eh)':>13}{'err':>10}"
          f"{'var':>9}{'corr %':>9}{'V/T':>9}")
    print("-" * 80)
    for r in res:
        corr = (r["E"] - E_HF) / (E_EXACT - E_HF) * 100.0
        print(f"{LABELS[r['model']]:<32}{r['n_par']:>6}{r['E']:>13.5f}"
              f"{r['dE']:>10.5f}{r['var']:>9.4f}{corr:>9.1f}"
              f"{r['V']/r['T']:>9.4f}")
    print("-" * 80)
    print(f"{'HF limit (reference)':<32}{'':>6}{E_HF:>13.5f}")
    print(f"{'Exact (reference)':<32}{'':>6}{E_EXACT:>13.5f}")
    print("\nV/T equals -2 for an exact eigenstate, but here a1s is fixed to Z in")
    print("order to impose the e-n cusp: the variational family is not closed")
    print("under the rescaling r -> lambda*r, so the virial theorem cannot be")
    print("satisfied exactly. It measures the missing scaling freedom, not the")
    print("accuracy of the energy.")

    plot_convergence(res)
    plot_jastrow(res)
    plot_summary(res)
    save_results(res)
    plt.show()
    return res


if __name__ == "__main__":
    main()