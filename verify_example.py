#!/usr/bin/env python3
"""
Full numerical verification of the finite-horizon CHUS example:
  beta=1/2, gamma=1, R0=0.3, Rmax=0.6, xi=0.05
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from scipy.integrate import quad, solve_ivp

# ---------------------------------------------------------------------------
# Parameters (paper example)
# ---------------------------------------------------------------------------
BETA = 0.5
GAMMA = 1.0
R0 = 0.3
RMAX = 0.6
XI = 0.05

OUT_DIR = Path(__file__).resolve().parent
PLOTS_DIR = OUT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


def f(x: np.ndarray | float) -> np.ndarray | float:
    x = np.asarray(x, dtype=float)
    return x**BETA * (1.0 - x**GAMMA) ** 2


def fprime_formula(x: float) -> float:
    rho = (BETA / (BETA + 2.0 * GAMMA)) ** (1.0 / GAMMA)
    return (
        -(BETA + 2.0 * GAMMA)
        * x ** (BETA - 1.0)
        * (1.0 - x**GAMMA)
        * (x**GAMMA - rho**GAMMA)
    )


def main() -> None:
    rho = (BETA / (BETA + 2.0 * GAMMA)) ** (1.0 / GAMMA)
    fmax = (2.0 * GAMMA / (BETA + 2.0 * GAMMA)) ** 2 * (
        BETA / (BETA + 2.0 * GAMMA)
    ) ** (BETA / GAMMA)

    print("=" * 60)
    print("CONSTANTS")
    print("=" * 60)
    print(f"rho          = {rho:.16f}")
    print(f"Fmax         = {fmax:.16f}")
    print(f"f(rho)       = {float(f(rho)):.16f}")
    print(f"f(Rmax)      = {float(f(RMAX)):.16f}")
    print(f"xi < f(Rmax) = {XI < float(f(RMAX))}")
    print(f"xi < Fmax    = {XI < fmax}")

    # Derivative factorization check
    xs = np.linspace(0.05, 0.95, 100)
    max_fp_err = 0.0
    for x in xs:
        h = 1e-7
        num = (float(f(x + h)) - float(f(x - h))) / (2.0 * h)
        max_fp_err = max(max_fp_err, abs(num - fprime_formula(float(x))))
    print(f"max |f' num - formula| = {max_fp_err:.3e}")
    assert max_fp_err < 1e-5

    tau_star = (RMAX - R0) / (fmax + XI)
    tau_max, tau_err = quad(lambda s: 1.0 / (float(f(s)) + XI), R0, RMAX, epsabs=1e-14)
    m0 = R0**GAMMA - rho**GAMMA
    phi = (BETA + 2.0 * GAMMA) * RMAX ** (BETA - 1.0) * (1.0 - RMAX**GAMMA) * m0
    u_const = 1.0 / phi

    def E(t):
        t = np.asarray(t, dtype=float)
        return (XI / phi) * (1.0 - np.exp(-phi * t))

    print(f"tau*         = {tau_star:.16f}")
    print(f"tau_max      = {tau_max:.16f}  (quad err {tau_err:.2e})")
    print(f"m0           = {m0:.16f}")
    print(f"Phi          = {phi:.16f}")
    print(f"U=1/Phi      = {u_const:.16f}")
    print(f"xi/Phi       = {XI / phi:.16f}")
    print(f"E(tau*)      = {float(E(tau_star)):.16f}")
    assert tau_star < tau_max
    assert 0 < XI < float(f(RMAX))
    assert rho < R0 < RMAX < 1.0

    # -----------------------------------------------------------------------
    # ODE integration (high accuracy)
    # -----------------------------------------------------------------------
    opts = dict(rtol=1e-11, atol=1e-13, method="DOP853", dense_output=True, max_step=0.01)

    sol_r = solve_ivp(lambda t, y: [float(f(y[0]))], [0.0, tau_max * 1.02], [R0], **opts)
    sol_l = solve_ivp(
        lambda t, y: [float(f(y[0])) - XI], [0.0, tau_max * 1.02], [R0], **opts
    )
    sol_h = solve_ivp(
        lambda t, y: [float(f(y[0])) + XI], [0.0, tau_max * 1.02], [R0], **opts
    )
    sol_b = solve_ivp(
        lambda t, y: [float(f(y[0])) + XI * np.sin(5.0 * t)],
        [0.0, tau_max * 1.02],
        [R0],
        **opts,
    )
    assert sol_r.success and sol_l.success and sol_h.success and sol_b.success

    h_at_tmax = float(sol_h.sol(tau_max)[0])
    print(f"H(tau_max)   = {h_at_tmax:.16f}")
    print(f"|H-Rmax|     = {abs(h_at_tmax - RMAX):.3e}")
    assert abs(h_at_tmax - RMAX) < 1e-8

    # Table at sample times (paper table)
    sample_t = [0.0, 0.2, 0.4, 0.6, 0.8, tau_star]
    print()
    print("=" * 60)
    print("NUMERICAL TABLE (for paper)")
    print("=" * 60)
    header = (
        f"{'t':>12} {'R':>10} {'L':>10} {'H':>10} {'Bosc':>10} "
        f"{'R-L':>10} {'H-R':>10} {'|B-R|':>10} {'E(t)':>10}"
    )
    print(header)
    table_rows = []
    for t in sample_t:
        r = float(sol_r.sol(t)[0])
        l = float(sol_l.sol(t)[0])
        h = float(sol_h.sol(t)[0])
        b = float(sol_b.sol(t)[0])
        row = {
            "t": t,
            "R": r,
            "L": l,
            "H": h,
            "Bosc": b,
            "R_minus_L": r - l,
            "H_minus_R": h - r,
            "abs_B_minus_R": abs(b - r),
            "E": float(E(t)),
        }
        table_rows.append(row)
        print(
            f"{t:12.5f} {r:10.5f} {l:10.5f} {h:10.5f} {b:10.5f} "
            f"{r-l:10.5f} {h-r:10.5f} {abs(b-r):10.5f} {float(E(t)):10.5f}"
        )

    # Dense verification on [0, tau*]
    tt = np.linspace(0.0, tau_star, 50001)
    r = sol_r.sol(tt)[0]
    l = sol_l.sol(tt)[0]
    h = sol_h.sol(tt)[0]
    b = sol_b.sol(tt)[0]
    env = E(tt)

    max_rl = float(np.max(r - l))
    max_hr = float(np.max(h - r))
    max_br = float(np.max(np.abs(b - r)))
    t_max_br = float(tt[int(np.argmax(np.abs(b - r)))])

    print()
    print("=" * 60)
    print("DENSE CHECKS ON [0, tau*]")
    print("=" * 60)
    print(f"max(R-L)     = {max_rl:.10f} at t={tt[int(np.argmax(r-l))]:.6f}")
    print(f"max(H-R)     = {max_hr:.10f} at t={tt[int(np.argmax(h-r))]:.6f}")
    print(f"max|B-R|     = {max_br:.10f} at t={t_max_br:.6f}")
    print(f"E(t_max|B|)  = {float(E(t_max_br)):.10f}")
    print(f"min L        = {float(np.min(l)):.10f}  (barrier rho={rho})")
    print(f"max H        = {float(np.max(h)):.10f}  (Rmax={RMAX})")

    # Assertions
    tol = 1e-9
    assert np.all(r + tol >= l)
    assert np.all(h + tol >= r)
    assert np.all(b + 1e-8 >= l)
    assert np.all(b - 1e-8 <= h)
    assert np.all(l + tol >= rho)
    assert np.all(h <= RMAX + 1e-8)
    assert np.all(r - l <= env + 1e-9)
    assert np.all(h - r <= env + 1e-9)
    assert np.all(np.abs(b - r) <= env + 1e-9)

    print()
    print("ALL AUTOMATED CHECKS PASSED")

    # -----------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plots")
        results = None
    else:
        t_plot = np.linspace(0.0, tau_star, 2001)
        rp = sol_r.sol(t_plot)[0]
        lp = sol_l.sol(t_plot)[0]
        hp = sol_h.sol(t_plot)[0]
        bp = sol_b.sol(t_plot)[0]
        ep = E(t_plot)

        # Trajectories
        fig, ax = plt.subplots(figsize=(8.2, 5.0))
        ax.plot(t_plot, rp, label=r"$R(t)$", lw=2.0)
        ax.plot(t_plot, lp, label=r"$\mathcal{L}(t)$", lw=1.8, ls="--")
        ax.plot(t_plot, hp, label=r"$\mathcal{H}(t)$", lw=1.8, ls="-.")
        ax.plot(t_plot, bp, label=r"$\mathcal{B}_{\mathrm{osc}}(t)$", lw=1.6, ls=":")
        ax.axhline(rho, color="gray", ls=":", lw=1.0, label=r"$\rho$")
        ax.axhline(RMAX, color="gray", ls="--", lw=1.0, label=r"$R_{\max}$")
        ax.set_xlabel(r"$t$")
        ax.set_ylabel("trajectory value")
        ax.set_title(r"Trajectories on the certified interval $[0,\tau^\ast]$")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "trajectories_plot.pdf")
        fig.savefig(PLOTS_DIR / "trajectories_plot.png", dpi=160)
        plt.close(fig)

        # Errors vs envelope
        fig, ax = plt.subplots(figsize=(8.2, 5.0))
        ax.plot(t_plot, rp - lp, label=r"$R-\mathcal{L}$", lw=1.8)
        ax.plot(t_plot, hp - rp, label=r"$\mathcal{H}-R$", lw=1.8, ls="--")
        ax.plot(t_plot, np.abs(bp - rp), label=r"$|\mathcal{B}_{\mathrm{osc}}-R|$", lw=1.6, ls="-.")
        ax.plot(t_plot, ep, label=r"$E(t)=\frac{\xi}{\Phi}(1-e^{-\Phi t})$", lw=2.2, color="k")
        ax.set_xlabel(r"$t$")
        ax.set_ylabel("error")
        ax.set_title(r"Numerical errors and theoretical envelope on $[0,\tau^\ast]$")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "error_envelope_plot.pdf")
        fig.savefig(PLOTS_DIR / "error_envelope_plot.png", dpi=160)
        plt.close(fig)
        print(f"Plots written to {PLOTS_DIR}")

    # Save machine-readable results for the paper
    results = {
        "rho": rho,
        "Fmax": fmax,
        "f_Rmax": float(f(RMAX)),
        "tau_star": tau_star,
        "tau_max": tau_max,
        "m0": m0,
        "Phi": phi,
        "U": u_const,
        "U_xi": XI / phi,
        "E_tau_star": float(E(tau_star)),
        "H_at_tau_max": h_at_tmax,
        "max_R_minus_L": max_rl,
        "max_H_minus_R": max_hr,
        "max_abs_B_minus_R": max_br,
        "t_at_max_abs_B": t_max_br,
        "E_at_max_abs_B": float(E(t_max_br)),
        "table": table_rows,
    }
    out_json = OUT_DIR / "verification_results.json"
    with open(out_json, "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)
    print(f"Results written to {out_json}")

    # Pretty summary for LaTeX rounding (5 decimals as in paper)
    print()
    print("=" * 60)
    print("PAPER-READY ROUNDED VALUES (5 d.p.)")
    print("=" * 60)
    print(f"rho       ≈ {rho:.5f}")
    print(f"Fmax      ≈ {fmax:.5f}")
    print(f"f(Rmax)   ≈ {float(f(RMAX)):.5f}")
    print(f"tau*      ≈ {tau_star:.5f}")
    print(f"tau_max   ≈ {tau_max:.5f}")
    print(f"Phi       ≈ {phi:.5f}")
    print(f"U         ≈ {u_const:.5f}")
    print(f"xi/Phi    ≈ {XI/phi:.5f}")
    print(f"E(tau*)   ≈ {float(E(tau_star)):.5f}")
    print(f"max(R-L)  ≈ {max_rl:.5f}")
    print(f"max(H-R)  ≈ {max_hr:.5f}")
    print(f"max|B-R|  ≈ {max_br:.5f} at t≈{t_max_br:.5f}")
    print(f"E(t*)     ≈ {float(E(t_max_br)):.5f}")


if __name__ == "__main__":
    main()
