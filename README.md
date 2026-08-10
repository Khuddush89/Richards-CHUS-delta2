# Conditional Hyers–Ulam Stability of the Squared Generalized Richards Model (δ = 2)

Finite-horizon conditional Hyers–Ulam stability (CHUS) for the reduced
generalized Richards equation with squared saturation factor,

```
R'(t) = R^β(t) (1 − R^γ(t))².
```

M. Khuddush, J. Z. Lobo, S. Tikare.

**This is the revised package.** See `CHANGES.md` for the complete list of
corrections, additions and rebuilt figures.

---

## Contents

| File | Description |
|------|-------------|
| `CHUS.tex` | Main LaTeX manuscript (corrected) |
| `CHUS.pdf` | Compiled PDF — 27 pages |
| `CHANGES.md` | Every edit made, with reasons |
| `ANDERSON_ONITSUKA_ERRATA.md` | Full audit of the published δ = 1 paper (Nonlinear Anal. RWA **89** (2026) 104530) |
| `make_figures.py` | Regenerates all four figures at 400 dpi (PDF + PNG) |
| `verify_example.py` | Independent numerical verification (Python) |
| `verification_results.json` | Machine-readable output of the verification |
| `plots/` | Figures and `figure_constants.json` |
| `julia_finite_horizon_bundle/` | Julia cross-check (`finite_horizon_verification.jl`) and its outputs |
| `example code.mw` | Maple worksheet (optional cross-check) |

## Figures

| Figure | File | Where |
|---|---|---|
| 1. Geometry of the drift `f`, `ρ`, `F_max`, admissibility gap | `plots/drift_geometry_plot.*` | §2 |
| 2. Finite-time blow-up of `H` for four values of `ξ` (log scale) | `plots/blowup_plot.*` | §3 |
| 3. Certified trajectories and the admissible corridor `[L, H]` | `plots/trajectories_plot.*` | §6 |
| 4. Error envelope `E(t)` with full-scale inset | `plots/error_envelope_plot.*` | §6 |

All figures use light backgrounds, saturated colour-blind-safe hues, 400 dpi
rasters (≈3900 px wide) and true-vector PDFs.

## Build

```bash
python3 make_figures.py     # requires numpy, scipy, matplotlib
pdflatex CHUS.tex
pdflatex CHUS.tex           # run twice for cross-references
```

To change the worked example, edit `BETA, GAMMA, R0, RMAX, XI` at the top of
`make_figures.py`; every figure, annotation and derived constant updates
consistently.

## Run the numerical verification

```bash
python3 verify_example.py
```

This script:

1. Checks `ρ`, `F_max`, `Φ`, `τ*`, `τ_max`
2. Integrates `R`, `L`, `H`, `B_osc`
3. Verifies the barrier, the ordering, and the envelope bound `E(t)`
4. Writes `verification_results.json`

### Example parameters

- `β = 1/2`, `γ = 1`, `R₀ = 0.3`, `R_max = 0.6`, `ξ = 0.05`
- `ρ = 0.2`, `F_max ≈ 0.286217`, `f(R_max) ≈ 0.123935`, `Φ ≈ 0.129099`
- `τ* ≈ 0.892282`, `τ_max ≈ 1.234058`, `U = 1/Φ ≈ 7.745967`

Independently reproduced with DOP853 at `rtol = 1e-11`, `atol = 1e-13`:
`H(τ_max) − R_max = 2.2e-16`, `max(R−L) = 0.036119`, `max(H−R) = 0.035915`,
`max|B_osc−R| = 0.017375`, `E(τ*) = 0.042140`. No envelope or ordering
violations on a 20 001-point grid.
