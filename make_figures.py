#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_figures.py
===============
High-resolution, high-contrast colour figures for

    "Conditional Hyers-Ulam Stability of a Squared Generalized Richards
     Equation: Finite-Time Blow-Up and a Finite-Horizon Theorem"
     M. Khuddush, J. Z. Lobo, S. Tikare

Produces (in ./plots):

    trajectories_plot.{pdf,png}   Fig. 1  certified-interval trajectories
    error_envelope_plot.{pdf,png} Fig. 2  errors vs. theoretical envelope
    drift_geometry_plot.{pdf,png} Fig. 3  geometry of f, rho, F_max, R_max
    blowup_plot.{pdf,png}         Fig. 4  finite-time blow-up of H

All panels use a light background (never black), saturated colour-blind-safe
hues, 400 dpi rasters and true-vector PDFs.

Run:  python3 make_figures.py
"""

import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch  # noqa: F401  (kept for styling)
from matplotlib.ticker import AutoMinorLocator
from scipy.integrate import quad, solve_ivp

# --------------------------------------------------------------------------
# 0.  Model parameters of the worked example
# --------------------------------------------------------------------------
BETA, GAMMA = 0.5, 1.0
R0, RMAX, XI = 0.3, 0.6, 0.05

RHO = (BETA / (BETA + 2 * GAMMA)) ** (1.0 / GAMMA)
FMAX = (2 * GAMMA / (BETA + 2 * GAMMA)) ** 2 * (BETA / (BETA + 2 * GAMMA)) ** (
    BETA / GAMMA
)


def f(x):
    x = np.maximum(np.asarray(x, dtype=float), 0.0)
    return x**BETA * (1.0 - x**GAMMA) ** 2


PHI = (BETA + 2 * GAMMA) * RMAX ** (BETA - 1) * (1 - RMAX**GAMMA) * (
    R0**GAMMA - RHO**GAMMA
)
TAU_STAR = (RMAX - R0) / (FMAX + XI)
TAU_MAX = quad(lambda s: 1.0 / (f(s) + XI), R0, RMAX, limit=400)[0]


def E(t):
    return XI / PHI * (1.0 - np.exp(-PHI * np.asarray(t, dtype=float)))


# --------------------------------------------------------------------------
# 1.  House style  -- light, saturated, print-safe
# --------------------------------------------------------------------------
INK = "#16213E"        # deep navy, used instead of pure black
PAPER = "#FFFFFF"      # figure background
PANEL = "#FAFBFF"      # axes background: barest blue tint
GRID = "#C9D2E8"

C_R = "#0B4FD8"        # exact solution           - strong blue
C_L = "#00A878"        # lower comparison         - emerald
C_H = "#E8175D"        # upper comparison         - magenta-red
C_B = "#F08A00"        # oscillatory perturbation - amber
C_ENV = "#7B2FF7"      # theoretical envelope     - violet
C_RHO = "#00A0B0"      # barrier level rho        - teal
C_MAX = "#D62246"      # confinement level R_max  - crimson
C_BAND = "#B9D2FF"     # L-H corridor fill
C_SAFE = "#DFF6EC"     # certified-time shading

plt.rcParams.update(
    {
        "figure.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "axes.facecolor": PANEL,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "axes.linewidth": 1.3,
        "axes.titlesize": 15,
        "axes.labelsize": 13.5,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.labelsize": 11.5,
        "ytick.labelsize": 11.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "cm",
        "legend.fontsize": 11.5,
        "legend.framealpha": 0.96,
        "legend.edgecolor": GRID,
        "legend.fancybox": True,
        "lines.solid_capstyle": "round",
        "lines.dash_capstyle": "round",
        "figure.dpi": 130,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.12,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def dress(ax, title=None, xlabel=None, ylabel=None):
    """Common axis cosmetics: soft grid, minor ticks, open frame."""
    ax.grid(True, which="major", color=GRID, lw=0.9, alpha=0.85, zorder=0)
    ax.grid(True, which="minor", color=GRID, lw=0.5, alpha=0.45, zorder=0)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK)
    ax.tick_params(which="both", length=5, width=1.1)
    ax.tick_params(which="minor", length=3, width=0.8)
    if title:
        ax.set_title(title, pad=13, fontweight="bold", color=INK)
    if xlabel:
        ax.set_xlabel(xlabel, labelpad=7)
    if ylabel:
        ax.set_ylabel(ylabel, labelpad=7)
    return ax


def glow(ax, x, y, color, lw=2.9, halo=7.5, **kw):
    """Draw a line with a soft same-hue halo so curves stay legible when
    they run close together (and when printed in greyscale)."""
    ax.plot(x, y, color=color, lw=lw + halo * 0.55, alpha=0.13,
            solid_capstyle="round", zorder=kw.get("zorder", 3) - 1)
    return ax.plot(x, y, color=color, lw=lw, **kw)


def save(fig, stem):
    os.makedirs("plots", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join("plots", f"{stem}.{ext}"))
    plt.close(fig)
    print(f"  wrote plots/{stem}.pdf and plots/{stem}.png")


# --------------------------------------------------------------------------
# 2.  Integrate the four trajectories
# --------------------------------------------------------------------------
def rhs(t, y):
    R, L, H, B = y
    return [f(R), f(L) - XI, f(H) + XI, f(B) + XI * np.sin(5.0 * t)]


sol = solve_ivp(
    rhs, [0.0, TAU_MAX], [R0] * 4,
    method="DOP853", rtol=1e-11, atol=1e-13, dense_output=True,
)

t_star = np.linspace(0.0, TAU_STAR, 4001)
t_full = np.linspace(0.0, TAU_MAX, 4001)
Rs, Ls, Hs, Bs = sol.sol(t_star)
Rf, Lf, Hf, Bf = sol.sol(t_full)


# ==========================================================================
# FIGURE 1 -- trajectories on the certified interval
# ==========================================================================
def figure_trajectories():
    fig, ax = plt.subplots(figsize=(11.2, 6.6))
    dress(
        ax,
        title=r"Certified confinement of the exact and perturbed trajectories"
              "\n"
              r"$R'=R^{1/2}(1-R)^2$,  $R_0=0.3$,  $\xi=0.05$,  "
              r"$R_{\max}=0.6$",
        xlabel=r"$t$",
        ylabel=r"trajectory value",
    )

    # confinement strip [rho, R_max]
    ax.axhspan(RHO, RMAX, color="#EFF4FF", zorder=0)
    # certified-time shading
    ax.axvspan(0, TAU_STAR, color=C_SAFE, alpha=0.75, zorder=0)
    ax.axvspan(TAU_STAR, TAU_MAX, color="#FFF3D6", alpha=0.8, zorder=0)

    # admissible corridor between L and H
    ax.fill_between(t_full, Lf, Hf, color=C_BAND, alpha=0.55, zorder=1,
                    label=r"admissible corridor $[\mathcal{L},\mathcal{H}]$")

    glow(ax, t_full, Hf, C_H, lw=3.0, ls="-",
         label=r"$\mathcal{H}(t)$  (upper, $q\equiv+\xi$)", zorder=6)
    glow(ax, t_full, Bf, C_B, lw=3.0, ls=(0, (6, 2.2)),
         label=r"$\mathcal{B}_{\mathrm{osc}}(t)$  ($q=\xi\sin 5t$)", zorder=5)
    glow(ax, t_full, Rf, C_R, lw=3.4, ls="-",
         label=r"$R(t)$  (exact)", zorder=7)
    glow(ax, t_full, Lf, C_L, lw=3.0, ls=(0, (1.3, 2.2)),
         label=r"$\mathcal{L}(t)$  (lower, $q\equiv-\xi$)", zorder=6)

    # reference levels
    ax.axhline(RHO, color=C_RHO, lw=2.0, ls=(0, (7, 3)), zorder=4)
    ax.axhline(RMAX, color=C_MAX, lw=2.0, ls=(0, (7, 3)), zorder=4)
    ax.axvline(TAU_STAR, color="#0B7A4B", lw=1.8, ls=(0, (2, 2.4)), zorder=4)
    ax.axvline(TAU_MAX, color="#B26B00", lw=1.8, ls=(0, (2, 2.4)), zorder=4)

    ax.text(0.015, RHO + 0.007, r"barrier  $\rho=0.2$", color=C_RHO,
            fontsize=12.5, fontweight="bold", va="bottom")
    ax.text(0.015, RMAX - 0.009, r"confinement  $R_{\max}=0.6$", color=C_MAX,
            fontsize=12.5, fontweight="bold", va="top")

    bbox = dict(boxstyle="round,pad=0.34", fc="white", ec=GRID, lw=1.0,
                alpha=0.95)
    ax.annotate(r"$\tau^{\ast}\approx0.89228$" "\n" r"elementary certificate",
                xy=(TAU_STAR, 0.252), xytext=(TAU_STAR - 0.49, 0.235),
                color="#0B7A4B", fontsize=11.5, fontweight="bold", bbox=bbox,
                arrowprops=dict(arrowstyle="-|>", color="#0B7A4B", lw=1.6))
    ax.annotate(r"$\tau_{\max}\approx1.23406$" "\n"
                r"$\mathcal{H}(\tau_{\max})=R_{\max}$",
                xy=(TAU_MAX, 0.252), xytext=(TAU_MAX - 0.015, 0.352),
                ha="right", color="#B26B00", fontsize=11.5,
                fontweight="bold", bbox=bbox,
                arrowprops=dict(arrowstyle="-|>", color="#B26B00", lw=1.6))

    ax.scatter([TAU_MAX], [RMAX], s=95, color=C_MAX, ec="white", lw=1.8,
               zorder=9)
    ax.scatter([0], [R0], s=95, color=INK, ec="white", lw=1.8, zorder=9)
    ax.annotate(r"$R_0=0.3$", xy=(0, R0), xytext=(0.06, 0.268),
                fontsize=11.5, fontweight="bold", bbox=bbox,
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.5))

    ax.set_xlim(-0.02, TAU_MAX * 1.02)
    ax.set_ylim(0.185, 0.632)
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.135), ncol=3,
                    borderpad=0.7, columnspacing=1.8, handlelength=2.6)
    leg.get_frame().set_facecolor("white")
    save(fig, "trajectories_plot")


# ==========================================================================
# FIGURE 2 -- errors against the theoretical envelope
# ==========================================================================
def figure_envelope():
    """Main axes: zoom on the true errors against E(t).
    Inset: same data on the full scale, showing how far the constant
    ceiling xi/Phi sits above everything."""
    fig, ax = plt.subplots(figsize=(11.4, 6.8))
    dress(
        ax,
        title=r"Numerical deviation versus the certified envelope  "
              r"$E(t)=\frac{\xi}{\Phi}\left(1-e^{-\Phi t}\right)$",
        xlabel=r"$t$",
        ylabel=r"deviation from the exact solution",
    )

    Ev = E(t_star)
    eL, eH, eB = Rs - Ls, Hs - Rs, np.abs(Bs - Rs)

    ax.fill_between(t_star, eH, Ev, color=C_ENV, alpha=0.16, zorder=1,
                    label=r"proved slack  $E(t)-|\mathcal{B}-R|$")
    glow(ax, t_star, Ev, C_ENV, lw=3.8,
         label=r"envelope $E(t)$  (proved bound)", zorder=8)
    glow(ax, t_star, eH, C_H, lw=3.4, ls="-",
         label=r"$\mathcal{H}-R$   (max $0.03592$)", zorder=5)
    glow(ax, t_star, eL, C_L, lw=2.6, ls=(0, (7.5, 5.5)), halo=0,
         label=r"$R-\mathcal{L}$   (max $0.03612$)", zorder=6)
    glow(ax, t_star, eB, C_B, lw=3.0, ls=(0, (6, 2.3)),
         label=r"$|\mathcal{B}_{\mathrm{osc}}-R|$   (max $0.01738$)", zorder=7)

    bbox = dict(boxstyle="round,pad=0.34", fc="white", ec=GRID, lw=1.0,
                alpha=0.96)
    i = int(np.argmax(eB))
    ax.scatter([t_star[i]], [eB[i]], s=100, color=C_B, ec="white", lw=1.9,
               zorder=10)
    ax.annotate(r"$\max|\mathcal{B}_{\mathrm{osc}}-R|\approx0.01738$  at "
                r"$t\approx0.59252$" "\n"
                r"envelope there: $E\approx0.02852$",
                xy=(t_star[i], eB[i]), xytext=(0.335, 0.0055),
                color="#9A5A00", fontsize=11.5, fontweight="bold", bbox=bbox,
                arrowprops=dict(arrowstyle="-|>", color=C_B, lw=1.7))

    ax.scatter([TAU_STAR], [Ev[-1]], s=100, color=C_ENV, ec="white", lw=1.9,
               zorder=10)
    ax.annotate(r"$E(\tau^{\ast})\approx0.04214$",
                xy=(TAU_STAR, Ev[-1]), xytext=(TAU_STAR - 0.30, 0.0455),
                color=C_ENV, fontsize=12, fontweight="bold", bbox=bbox,
                arrowprops=dict(arrowstyle="-|>", color=C_ENV, lw=1.7))

    ax.text(0.36, 0.0335,
            r"$R-\mathcal{L}$ and $\mathcal{H}-R$ almost coincide"
            "\n" r"(they differ by $2\times10^{-4}$ at $\tau^{\ast}$)",
            color="#7A2038", fontsize=11, fontweight="bold", rotation=13.5,
            rotation_mode="anchor")
    ax.set_xlim(-0.012, TAU_STAR * 1.03)
    ax.set_ylim(0, 0.050)

    # ---- inset: full vertical scale, with the constant ceiling ----
    axi = ax.inset_axes([0.055, 0.545, 0.335, 0.395])
    axi.set_facecolor("#FFFFFF")
    axi.fill_between(t_star, 0, Ev, color=C_ENV, alpha=0.18)
    axi.fill_between(t_star, Ev, XI / PHI, color="#FDE7E7", alpha=0.95)
    axi.plot(t_star, Ev, color=C_ENV, lw=2.4)
    axi.plot(t_star, eL, color=C_L, lw=1.9, ls=(0, (1.4, 2.3)))
    axi.plot(t_star, eH, color=C_H, lw=1.9)
    axi.plot(t_star, eB, color=C_B, lw=1.9, ls=(0, (6, 2.3)))
    axi.axhline(XI / PHI, color=C_MAX, lw=2.1, ls=(0, (6, 3)))
    axi.set_xlim(0, TAU_STAR)
    axi.set_ylim(0, XI / PHI * 1.10)
    axi.set_yticks([0, 0.1, 0.2, 0.3, 0.387])
    axi.set_yticklabels(["0", "0.1", "0.2", "0.3", "0.387"], fontsize=9)
    axi.tick_params(labelsize=9, length=3, width=0.9)
    axi.grid(True, color=GRID, lw=0.6, alpha=0.6)
    for sd in ("top", "right"):
        axi.spines[sd].set_visible(False)
    axi.set_title(r"full scale: the ceiling $\mathcal{U}\xi=\xi/\Phi\approx0.38730$"
                  "\n" r"is about $9\times$ the true error",
                  fontsize=10.5, color=C_MAX, fontweight="bold", pad=6)
    axi.annotate("", xy=(0.62, XI / PHI), xytext=(0.62, float(Ev[-1])),
                 arrowprops=dict(arrowstyle="<|-|>", color=C_MAX, lw=1.5))
    axi.text(0.655, 0.21, "unused\nmargin", color=C_MAX, fontsize=9.5,
             fontweight="bold", va="center")

    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.135), ncol=3,
                    borderpad=0.7, columnspacing=1.6, handlelength=2.6)
    leg.get_frame().set_facecolor("white")
    save(fig, "error_envelope_plot")


# ==========================================================================
# FIGURE 3 -- geometry of the squared drift f
# ==========================================================================
def figure_drift():
    fig, ax = plt.subplots(figsize=(11.2, 6.4))
    dress(
        ax,
        title=r"Geometry of the squared drift  $f(x)=x^{\beta}(1-x^{\gamma})^{2}$"
              "\n"
              r"$f$ increases on $[0,\rho]$, decreases on $[\rho,1]$, "
              r"increases on $[1,\infty)$",
        xlabel=r"$x$",
        ylabel=r"$f(x)$",
    )

    x1 = np.linspace(0, RHO, 700)
    x2 = np.linspace(RHO, 1.0, 900)
    x3 = np.linspace(1.0, 1.28, 500)

    ax.fill_between(np.linspace(RHO, RMAX, 400), 0,
                    f(np.linspace(RHO, RMAX, 400)),
                    color="#E3F0FF", zorder=0)
    ax.fill_between(x1, 0, f(x1), color="#E7F8F1", zorder=0)

    glow(ax, x1, f(x1), C_L, lw=3.4, label=r"increasing on $[0,\rho]$",
         zorder=6)
    glow(ax, x2, f(x2), C_R, lw=3.4, label=r"decreasing on $[\rho,1]$",
         zorder=6)
    glow(ax, x3, f(x3), C_H, lw=3.4, label=r"increasing on $[1,\infty)$",
         zorder=6)

    ax.axhline(FMAX, color=C_RHO, lw=2.0, ls=(0, (7, 3)), zorder=4)
    ax.axhline(XI, color=C_B, lw=2.4, ls=(0, (4, 2.4)), zorder=4)
    ax.axhline(f(RMAX), color=C_MAX, lw=2.0, ls=(0, (2, 2.2)), zorder=4)
    ax.axvline(RHO, color=C_RHO, lw=1.7, ls=(0, (2, 2.6)), zorder=3)
    ax.axvline(RMAX, color=C_MAX, lw=1.7, ls=(0, (2, 2.6)), zorder=3)
    ax.axvline(1.0, color="#8A8FA8", lw=1.7, ls=(0, (2, 2.6)), zorder=3)

    ax.fill_between([RHO, RMAX], XI, f(RMAX), color="#FFE9B8", alpha=0.75,
                    zorder=1)
    bbox = dict(boxstyle="round,pad=0.34", fc="white", ec=GRID, lw=1.0,
                alpha=0.96)
    ax.annotate(r"admissibility gap  $f(R_{\max})-\xi\approx0.07394$",
                xy=(0.45, 0.5 * (XI + f(RMAX))), xytext=(0.63, 0.205),
                fontsize=11.5, fontweight="bold", color="#9A5A00", bbox=bbox,
                arrowprops=dict(arrowstyle="-|>", color="#B26B00", lw=1.6))

    ax.scatter([RHO], [FMAX], s=120, color=C_RHO, ec="white", lw=2, zorder=9)
    ax.scatter([RMAX], [f(RMAX)], s=110, color=C_MAX, ec="white", lw=2,
               zorder=9)
    ax.scatter([1.0], [0.0], s=110, color="#8A8FA8", ec="white", lw=2,
               zorder=9)

    ax.annotate(r"$(\rho,\;F_{\max})=(0.2,\;0.28622)$",
                xy=(RHO, FMAX), xytext=(0.265, 0.318), fontsize=11.5,
                fontweight="bold", color=C_RHO, bbox=bbox,
                arrowprops=dict(arrowstyle="-|>", color=C_RHO, lw=1.6))
    ax.annotate(r"$f(R_{\max})\approx0.12394$",
                xy=(RMAX, f(RMAX)), xytext=(0.72, 0.128), fontsize=11.5,
                fontweight="bold", color=C_MAX, bbox=bbox,
                arrowprops=dict(arrowstyle="-|>", color=C_MAX, lw=1.6))
    ax.text(0.012, XI + 0.006, r"$\xi=0.05$", color="#B26B00", fontsize=12.5,
            fontweight="bold", va="bottom")
    ax.text(0.995, 0.030, r"carrying level $x=1$" "\n" r"(double root of $f$)",
            color="#5A6076", fontsize=11.5, fontweight="bold", ha="right")

    ax.set_xlim(0, 1.28)
    ax.set_ylim(0, 0.355)
    leg = ax.legend(loc="upper right", borderpad=0.7, labelspacing=0.55,
                    handlelength=2.6)
    leg.get_frame().set_facecolor("white")
    save(fig, "drift_geometry_plot")


# ==========================================================================
# FIGURE 4 -- finite-time blow-up of the upper comparison solution
# ==========================================================================
def figure_blowup():
    fig, ax = plt.subplots(figsize=(11.2, 6.4))
    dress(
        ax,
        title=r"Finite-time blow-up of $\mathcal{H}'=f(\mathcal{H})+\xi$ "
              r"for every $\xi>0$"
              "\n"
              r"the exact solution $R$ stays global and bounded by the "
              r"carrying level",
        xlabel=r"$t$",
        ylabel=r"solution value  (log scale)",
    )

    xis = [0.20, 0.10, 0.05, 0.02]
    cols = ["#E8175D", "#F0662B", "#B02EC4", "#0B4FD8"]

    CAP = 5.0e5
    Ts = []
    for xi_, col in zip(xis, cols):
        # exact blow-up time, split at s=2 with u = 1/s on the tail
        Th = (quad(lambda s: 1.0 / (f(s) + xi_), R0, 2.0, limit=400)[0]
              + quad(lambda u: 1.0 / (u * u * (f(1.0 / u) + xi_)),
                     1e-12, 0.5, limit=400)[0])
        Ts.append(Th)

        def rhsH(t, y, xi_=xi_):
            return [f(y[0]) + xi_]

        def hit(t, y, xi_=xi_):
            return y[0] - CAP

        hit.terminal, hit.direction = True, 1
        s_ = solve_ivp(rhsH, [0, Th * 1.2], [R0], method="Radau",
                       rtol=1e-10, atol=1e-12, events=hit, dense_output=True,
                       max_step=Th / 400)
        te = s_.t_events[0][0] if len(s_.t_events[0]) else s_.t[-1]
        tt = np.linspace(0, te, 3000)
        glow(ax, tt, s_.sol(tt)[0], col, lw=3.0,
             label=(r"$\mathcal{H}$ with $\xi=%g$   "
                    r"($T_{\mathcal{H}}\approx%.3f$)" % (xi_, Th)),
             zorder=6)
        ax.axvline(Th, color=col, lw=1.5, ls=(0, (2, 3)), alpha=0.85,
                   zorder=3)

    tR = np.linspace(0, max(Ts) * 1.05, 2000)
    sR = solve_ivp(lambda t, y: [f(y[0])], [0, tR[-1]], [R0],
                   method="DOP853", rtol=1e-11, atol=1e-13,
                   dense_output=True)
    glow(ax, tR, sR.sol(tR)[0], C_L, lw=3.6,
         label=r"$R(t)$  exact: global, $R\to1$", zorder=7)

    ax.axhline(1.0, color="#5A6076", lw=1.9, ls=(0, (7, 3)), zorder=4)
    ax.text(13.0, 1.42, r"carrying level  $R=1$", color="#5A6076",
            fontsize=12.5, fontweight="bold", ha="left")

    bbox = dict(boxstyle="round,pad=0.34", fc="white", ec=GRID, lw=1.0,
                alpha=0.96)
    ax.annotate("no matter how small $\\xi$ is,\n"
                "$\mathcal{H}$ escapes in finite time",
                xy=(Ts[-1] * 0.9955, 6.0e2), xytext=(Ts[-1] * 0.36, 1.2e4),
                fontsize=12, fontweight="bold", color="#0B4FD8", bbox=bbox,
                arrowprops=dict(arrowstyle="-|>", color="#0B4FD8", lw=1.7))

    ax.set_yscale("log")
    ax.set_xlim(0, max(Ts) * 1.04)
    ax.set_ylim(0.22, CAP)
    leg = ax.legend(loc="center left", borderpad=0.7, labelspacing=0.5,
                    handlelength=2.6)
    leg.get_frame().set_facecolor("white")
    save(fig, "blowup_plot")
    return dict(zip([f"T_H(xi={x:g})" for x in xis], Ts))


# --------------------------------------------------------------------------
if __name__ == "__main__":
    print("Constants:")
    print(f"  rho      = {RHO:.10f}")
    print(f"  F_max    = {FMAX:.10f}")
    print(f"  f(R_max) = {f(RMAX):.10f}")
    print(f"  Phi      = {PHI:.10f}   U = 1/Phi = {1/PHI:.10f}")
    print(f"  tau*     = {TAU_STAR:.10f}")
    print(f"  tau_max  = {TAU_MAX:.10f}")
    print("Figures:")
    figure_trajectories()
    figure_envelope()
    figure_drift()
    blow = figure_blowup()
    with open("plots/figure_constants.json", "w") as fh:
        json.dump(
            {
                "beta": BETA, "gamma": GAMMA, "R0": R0,
                "Rmax": RMAX, "xi": XI,
                "rho": RHO, "F_max": FMAX, "f_Rmax": float(f(RMAX)),
                "Phi": PHI, "U": 1.0 / PHI, "U_xi": XI / PHI,
                "tau_star": TAU_STAR, "tau_max": TAU_MAX,
                "E_tau_star": float(E(TAU_STAR)),
                "blowup_times": blow,
            },
            fh, indent=2,
        )
    print("  wrote plots/figure_constants.json")
