# finite_horizon_verification.jl
#
# One-time installation:
# import Pkg
# Pkg.add(["OrdinaryDiffEq", "QuadGK", "DataFrames", "CSV", "PrettyTables", "Plots"])
#
# Run:
# julia finite_horizon_verification.jl

using OrdinaryDiffEq
using QuadGK
using DataFrames
using CSV
using PrettyTables
using Plots
using Printf

# =========================================================
# 1. PARAMETERS AND ANALYTICAL CONSTANTS
# =========================================================

const β = 0.5
const γ = 1.0
const R₀ = 0.3
const Rmax = 0.6
const ξ = 0.05

f(x) = x^β * (1.0 - x^γ)^2

ρ = (β / (β + 2γ))^(1 / γ)
Fmax = f(ρ)
f_Rmax = f(Rmax)
τstar = (Rmax - R₀) / (Fmax + ξ)

τmax, quadrature_error = quadgk(
    s -> 1.0 / (f(s) + ξ),
    R₀,
    Rmax;
    rtol = 1e-13,
    atol = 1e-13,
)

m₀ = R₀^γ - ρ^γ
Φ = (β + 2γ) * Rmax^(β - 1.0) * (1.0 - Rmax^γ) * m₀
U = 1.0 / Φ
uniform_ceiling = ξ / Φ
E(t) = ξ / Φ * (1.0 - exp(-Φ * t))

@printf("\n============================================\n")
@printf("NUMERICAL CONSTANTS\n")
@printf("============================================\n")
@printf("rho                    = %.16f\n", ρ)
@printf("Fmax                   = %.16f\n", Fmax)
@printf("f(Rmax)                = %.16f\n", f_Rmax)
@printf("f(Rmax) - xi           = %.16f\n", f_Rmax - ξ)
@printf("tau_star               = %.16f\n", τstar)
@printf("tau_max                = %.16f\n", τmax)
@printf("quadrature error       = %.5e\n", quadrature_error)
@printf("m0                     = %.16f\n", m₀)
@printf("Phi                    = %.16f\n", Φ)
@printf("HUS constant U         = %.16f\n", U)
@printf("uniform ceiling xi/Phi = %.16f\n", uniform_ceiling)
@printf("E(tau_star)            = %.16f\n", E(τstar))

@printf("\nHypothesis checks:\n")
@printf("rho < R0 < Rmax < 1    = %s\n", string(ρ < R₀ < Rmax < 1.0))
@printf("0 < xi < f(Rmax)       = %s\n", string(0.0 < ξ < f_Rmax))
@printf("tau_star < tau_max     = %s\n", string(τstar < τmax))

# =========================================================
# 2. COUPLED ODE SYSTEM
# u = [R, L, H, Bosc]
# DP5 is an adaptive Dormand-Prince 5/4 Runge-Kutta method.
# =========================================================

function rhs!(du, u, p, t)
    R, L, H, Bosc = u

    du[1] = f(R)
    du[2] = f(L) - ξ
    du[3] = f(H) + ξ
    du[4] = f(Bosc) + ξ * sin(5.0 * t)

    return nothing
end

u0 = [R₀, R₀, R₀, R₀]
prob = ODEProblem(rhs!, u0, (0.0, τstar))

sol = solve(
    prob,
    DP5();
    reltol = 1e-11,
    abstol = 1e-13,
    dense = true,
    save_everystep = true,
)

# =========================================================
# 3. ENDPOINT VALUES
# =========================================================

u_end = sol(τstar)
R_end, L_end, H_end, B_end = u_end

@printf("\n============================================\n")
@printf("ENDPOINT VALUES AT t = tau_star\n")
@printf("============================================\n")
@printf("R(tau_star)    = %.16f\n", R_end)
@printf("L(tau_star)    = %.16f\n", L_end)
@printf("H(tau_star)    = %.16f\n", H_end)
@printf("Bosc(tau_star) = %.16f\n", B_end)
@printf("R - L          = %.16f\n", R_end - L_end)
@printf("H - R          = %.16f\n", H_end - R_end)
@printf("|Bosc - R|     = %.16f\n", abs(B_end - R_end))
@printf("E(tau_star)    = %.16f\n", E(τstar))

# =========================================================
# 4. HITTING-TIME CHECK
# =========================================================

function rhs_H!(du, u, p, t)
    du[1] = f(u[1]) + ξ
    return nothing
end

prob_H = ODEProblem(rhs_H!, [R₀], (0.0, τmax))
sol_H = solve(
    prob_H,
    DP5();
    reltol = 1e-11,
    abstol = 1e-13,
    dense = true,
)

H_at_τmax = sol_H(τmax)[1]

@printf("\n============================================\n")
@printf("HITTING-TIME CHECK\n")
@printf("============================================\n")
@printf("tau_max                = %.16f\n", τmax)
@printf("H(tau_max)             = %.16f\n", H_at_τmax)
@printf("Rmax                   = %.16f\n", Rmax)
@printf("|H(tau_max)-Rmax|      = %.5e\n", abs(H_at_τmax - Rmax))

# =========================================================
# 5. NUMERICAL TABLE
# =========================================================

sample_times = [0.0, 0.2, 0.4, 0.6, 0.8, τstar]

table_df = DataFrame(
    t = Float64[],
    R = Float64[],
    L = Float64[],
    H = Float64[],
    Bosc = Float64[],
    R_minus_L = Float64[],
    H_minus_R = Float64[],
    abs_Bosc_minus_R = Float64[],
    E = Float64[],
)

for t in sample_times
    R, L, H, Bosc = sol(t)

    push!(
        table_df,
        (
            t,
            R,
            L,
            H,
            Bosc,
            R - L,
            H - R,
            abs(Bosc - R),
            E(t),
        ),
    )
end

println("\n============================================")
println("NUMERICAL TABLE")
println("============================================")

pretty_table(
    table_df;
    formatters = ft_printf("%.6f"),
    header = [
        "t",
        "R(t)",
        "L(t)",
        "H(t)",
        "Bosc(t)",
        "R-L",
        "H-R",
        "|Bosc-R|",
        "E(t)",
    ],
)

CSV.write("numerical_table.csv", table_df)

# Manuscript-ready LaTeX table
open("numerical_table.tex", "w") do io
    println(io, raw"\begin{table}[ht]")
    println(io, raw"\centering")
    println(
        io,
        raw"\caption{Numerical trajectories and theoretical error envelope on $[0,\tau^\ast]$.}",
    )
    println(io, raw"\label{tab:numerical-example-julia}")
    println(io, raw"\resizebox{\textwidth}{!}{%")
    println(io, raw"\begin{tabular}{ccccccccc}")
    println(io, raw"\hline")
    println(
        io,
        raw"$t$ & $R(t)$ & $\mathcal L(t)$ & $\mathcal H(t)$ & " *
        raw"$\mathcal B_{\mathrm{osc}}(t)$ & $R-\mathcal L$ & " *
        raw"$\mathcal H-R$ & $|\mathcal B_{\mathrm{osc}}-R|$ & $E(t)$\\",
    )
    println(io, raw"\hline")

    for (i, row) in enumerate(eachrow(table_df))
        tlabel =
            i == 1 ? raw"$0$" :
            i == nrow(table_df) ?
            "\$\\tau^\\ast\\approx$(@sprintf("%.6f", row.t))\$" :
            "\$" * @sprintf("%.1f", row.t) * "\$"

        values = [
            @sprintf("\$%.6f\$", row.R),
            @sprintf("\$%.6f\$", row.L),
            @sprintf("\$%.6f\$", row.H),
            @sprintf("\$%.6f\$", row.Bosc),
            @sprintf("\$%.6f\$", row.R_minus_L),
            @sprintf("\$%.6f\$", row.H_minus_R),
            @sprintf("\$%.6f\$", row.abs_Bosc_minus_R),
            @sprintf("\$%.6f\$", row.E),
        ]

        println(io, tlabel * " & " * join(values, " & ") * raw"\\")
    end

    println(io, raw"\hline")
    println(io, raw"\end{tabular}}")
    println(io, raw"\end{table}")
end

# =========================================================
# 6. DENSE-GRID VERIFICATION
# =========================================================

tgrid = range(0.0, τstar; length = 20_001)
dense = reduce(hcat, sol.(tgrid))

Rvals = vec(dense[1, :])
Lvals = vec(dense[2, :])
Hvals = vec(dense[3, :])
Bvals = vec(dense[4, :])

lower_error = Rvals .- Lvals
upper_error = Hvals .- Rvals
osc_error = abs.(Bvals .- Rvals)
envelope = E.(tgrid)

i_lower = argmax(lower_error)
i_upper = argmax(upper_error)
i_osc = argmax(osc_error)

max_ordering_violation = maximum(
    max.(
        0.0,
        max.(Lvals .- Bvals, Bvals .- Hvals),
    ),
)

max_envelope_violation = maximum(
    max.(
        0.0,
        max.(
            lower_error .- envelope,
            max.(upper_error .- envelope, osc_error .- envelope),
        ),
    ),
)

@printf("\n============================================\n")
@printf("MAXIMUM ERRORS ON [0,tau_star]\n")
@printf("============================================\n")
@printf(
    "max(R-L)       = %.16f at t = %.16f\n",
    lower_error[i_lower],
    tgrid[i_lower],
)
@printf(
    "max(H-R)       = %.16f at t = %.16f\n",
    upper_error[i_upper],
    tgrid[i_upper],
)
@printf(
    "max|Bosc-R|    = %.16f at t = %.16f\n",
    osc_error[i_osc],
    tgrid[i_osc],
)
@printf("E at osc max   = %.16f\n", envelope[i_osc])
@printf("E(tau_star)    = %.16f\n", E(τstar))
@printf("min L(t)       = %.16f\n", minimum(Lvals))
@printf("max H(t)       = %.16f\n", maximum(Hvals))
@printf("rho            = %.16f\n", ρ)
@printf("Rmax           = %.16f\n", Rmax)
@printf("max ordering violation = %.5e\n", max_ordering_violation)
@printf("max envelope violation = %.5e\n", max_envelope_violation)

tol = 1e-8

barrier_pass = minimum(Lvals) >= ρ - tol
upper_confinement_pass = maximum(Hvals) <= Rmax + tol
ordering_pass = max_ordering_violation <= tol
error_bound_pass = max_envelope_violation <= tol
all_passed =
    barrier_pass &&
    upper_confinement_pass &&
    ordering_pass &&
    error_bound_pass

@printf("\n============================================\n")
@printf("AUTOMATIC VERIFICATION\n")
@printf("============================================\n")
@printf("Barrier L(t) >= rho             : %s\n", string(barrier_pass))
@printf("Upper confinement H(t) <= Rmax  : %s\n", string(upper_confinement_pass))
@printf("Ordering L <= Bosc <= H          : %s\n", string(ordering_pass))
@printf("All errors below E(t)            : %s\n", string(error_bound_pass))
@printf("\nAll checks passed                : %s\n", string(all_passed))

# =========================================================
# 7. SAVE SUMMARY
# =========================================================

open("summary_results.txt", "w") do io
    @printf(io, "rho                    = %.16f\n", ρ)
    @printf(io, "Fmax                   = %.16f\n", Fmax)
    @printf(io, "f(Rmax)                = %.16f\n", f_Rmax)
    @printf(io, "f(Rmax) - xi           = %.16f\n", f_Rmax - ξ)
    @printf(io, "tau_star               = %.16f\n", τstar)
    @printf(io, "tau_max                = %.16f\n", τmax)
    @printf(io, "quadrature error       = %.5e\n", quadrature_error)
    @printf(io, "m0                     = %.16f\n", m₀)
    @printf(io, "Phi                    = %.16f\n", Φ)
    @printf(io, "HUS constant U         = %.16f\n", U)
    @printf(io, "uniform ceiling xi/Phi = %.16f\n", uniform_ceiling)
    @printf(io, "E(tau_star)            = %.16f\n", E(τstar))
    @printf(io, "\nEndpoint values\n")
    @printf(io, "R(tau_star)             = %.16f\n", R_end)
    @printf(io, "L(tau_star)             = %.16f\n", L_end)
    @printf(io, "H(tau_star)             = %.16f\n", H_end)
    @printf(io, "Bosc(tau_star)          = %.16f\n", B_end)
    @printf(io, "\nHitting-time check\n")
    @printf(io, "H(tau_max)              = %.16f\n", H_at_τmax)
    @printf(io, "|H(tau_max)-Rmax|       = %.5e\n", abs(H_at_τmax - Rmax))
    @printf(io, "\nMaximum errors\n")
    @printf(
        io,
        "max(R-L)                = %.16f at t = %.16f\n",
        lower_error[i_lower],
        tgrid[i_lower],
    )
    @printf(
        io,
        "max(H-R)                = %.16f at t = %.16f\n",
        upper_error[i_upper],
        tgrid[i_upper],
    )
    @printf(
        io,
        "max|Bosc-R|             = %.16f at t = %.16f\n",
        osc_error[i_osc],
        tgrid[i_osc],
    )
    @printf(io, "E at oscillatory max    = %.16f\n", envelope[i_osc])
    @printf(io, "\nAll checks passed        = %s\n", string(all_passed))
end

# =========================================================
# 8. PUBLICATION-QUALITY PLOTS
# =========================================================

default(
    linewidth = 2.2,
    framestyle = :box,
    grid = true,
    legend = :best,
    size = (900, 600),
    dpi = 600,
    guidefontsize = 13,
    tickfontsize = 11,
    legendfontsize = 10,
    titlefontsize = 14,
)

p1 = plot(
    tgrid,
    Rvals;
    label = "R(t)",
    xlabel = "t",
    ylabel = "Trajectory value",
    title = "Trajectories on the certified finite horizon",
    linestyle = :solid,
)
plot!(p1, tgrid, Lvals; label = "L(t)", linestyle = :dash)
plot!(p1, tgrid, Hvals; label = "H(t)", linestyle = :dashdot)
plot!(p1, tgrid, Bvals; label = "B_osc(t)", linestyle = :dot)
hline!(p1, [ρ]; label = "rho", linestyle = :dash)
hline!(p1, [Rmax]; label = "R_max", linestyle = :dash)

savefig(p1, "trajectories_plot.pdf")
savefig(p1, "trajectories_plot.png")

p2 = plot(
    tgrid,
    lower_error;
    label = "R(t) - L(t)",
    xlabel = "t",
    ylabel = "Error",
    title = "Numerical errors and theoretical envelope",
    linestyle = :solid,
)
plot!(p2, tgrid, upper_error; label = "H(t) - R(t)", linestyle = :dash)
plot!(
    p2,
    tgrid,
    osc_error;
    label = "|B_osc(t) - R(t)|",
    linestyle = :dashdot,
)
plot!(p2, tgrid, envelope; label = "E(t)", linestyle = :dot, linewidth = 2.8)

savefig(p2, "error_envelope_plot.pdf")
savefig(p2, "error_envelope_plot.png")

println("\nFiles written:")
println("  numerical_table.csv")
println("  numerical_table.tex")
println("  summary_results.txt")
println("  trajectories_plot.pdf")
println("  trajectories_plot.png")
println("  error_envelope_plot.pdf")
println("  error_envelope_plot.png")
