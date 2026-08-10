#!/usr/bin/env julia
# Full numerical verification of the finite-horizon CHUS example
# beta=1/2, gamma=1, R0=0.3, Rmax=0.6, xi=0.05

using DifferentialEquations
using Plots
using JSON
using LinearAlgebra
using QuadGK

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
const BETA = 0.5
const GAMMA = 1.0
const R0 = 0.3
const RMAX = 0.6
const XI = 0.05
const OUT_DIR = ".."   # save results in project root
const PLOTS_DIR = joinpath(OUT_DIR, "plots")
mkpath(PLOTS_DIR)

# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------
f(x) = x^BETA * (1 - x^GAMMA)^2

rho = (BETA / (BETA + 2*GAMMA))^(1/GAMMA)
Fmax = (2*GAMMA / (BETA + 2*GAMMA))^2 * (BETA / (BETA + 2*GAMMA))^(BETA/GAMMA)

tau_star = (RMAX - R0) / (Fmax + XI)
tau_max, _ = quadgk(s -> 1/(f(s) + XI), R0, RMAX, rtol=1e-14)
m0 = R0^GAMMA - rho^GAMMA
Phi = (BETA + 2*GAMMA) * RMAX^(BETA-1) * (1 - RMAX^GAMMA) * m0
E(t) = (XI / Phi) * (1 - exp(-Phi * t))

# ---------------------------------------------------------------------------
# ODE Integration
# ---------------------------------------------------------------------------
tspan = (0.0, tau_max * 1.02)
prob_r = ODEProblem((u,p,t) -> [f(u[1])], [R0], tspan)
prob_l = ODEProblem((u,p,t) -> [f(u[1]) - XI], [R0], tspan)
prob_h = ODEProblem((u,p,t) -> [f(u[1]) + XI], [R0], tspan)
prob_b = ODEProblem((u,p,t) -> [f(u[1]) + XI*sin(5*t)], [R0], tspan)

sol_r = solve(prob_r, Tsit5(), reltol=1e-11, abstol=1e-13, dense=true)
sol_l = solve(prob_l, Tsit5(), reltol=1e-11, abstol=1e-13, dense=true)
sol_h = solve(prob_h, Tsit5(), reltol=1e-11, abstol=1e-13, dense=true)
sol_b = solve(prob_b, Tsit5(), reltol=1e-11, abstol=1e-13, dense=true)

# Check H(tau_max)
h_at_tmax = sol_h(tau_max)[1]
@assert abs(h_at_tmax - RMAX) < 1e-8 "H(tau_max) = $h_at_tmax, expected $RMAX"

# ---------------------------------------------------------------------------
# Sample table
# ---------------------------------------------------------------------------
sample_t = [0.0, 0.2, 0.4, 0.6, 0.8, tau_star]
table_rows = []
println("\n"^2 * "="^60)
println("NUMERICAL TABLE (Julia)")
println("="^60)
println(lpad("t", 12), lpad("R", 10), lpad("L", 10), lpad("H", 10), lpad("Bosc", 10),
        lpad("R-L", 10), lpad("H-R", 10), lpad("|B-R|", 10), lpad("E(t)", 10))

for t in sample_t
    r = sol_r(t)[1]
    l = sol_l(t)[1]
    h = sol_h(t)[1]
    b = sol_b(t)[1]
    row = Dict("t"=>t, "R"=>r, "L"=>l, "H"=>h, "Bosc"=>b,
               "R_minus_L"=>r-l, "H_minus_R"=>h-r, "abs_B_minus_R"=>abs(b-r), "E"=>E(t))
    push!(table_rows, row)
    println(@sprintf("%12.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f",
                     t, r, l, h, b, r-l, h-r, abs(b-r), E(t)))
end

# ---------------------------------------------------------------------------
# Dense checks on [0, tau*]
# ---------------------------------------------------------------------------
tt = range(0, tau_star, length=50001)
r_vals = sol_r.(tt) |> getindex.(1)
l_vals = sol_l.(tt) |> getindex.(1)
h_vals = sol_h.(tt) |> getindex.(1)
b_vals = sol_b.(tt) |> getindex.(1)
env_vals = E.(tt)

max_rl = maximum(r_vals - l_vals)
max_hr = maximum(h_vals - r_vals)
max_br = maximum(abs.(b_vals - r_vals))
t_max_br = tt[argmax(abs.(b_vals - r_vals))]

println("\n"^2 * "="^60)
println("DENSE CHECKS ON [0, tau*] (Julia)")
println("="^60)
println("max(R-L)     = $(max_rl)")
println("max(H-R)     = $(max_hr)")
println("max|B-R|     = $(max_br) at t=$t_max_br")
println("E(t_max|B|)  = $(E(t_max_br))")

# Assertions
@assert all(r_vals .+ 1e-9 .>= l_vals)
@assert all(h_vals .+ 1e-9 .>= r_vals)
@assert all(b_vals .+ 1e-8 .>= l_vals)
@assert all(b_vals .- 1e-8 .<= h_vals)
@assert all(l_vals .+ 1e-9 .>= rho)
@assert all(h_vals .<= RMAX + 1e-8)
@assert all((r_vals - l_vals) .<= env_vals .+ 1e-9)
@assert all((h_vals - r_vals) .<= env_vals .+ 1e-9)
@assert all(abs.(b_vals - r_vals) .<= env_vals .+ 1e-9)

println("\nALL AUTOMATED CHECKS PASSED (Julia)")

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
t_plot = range(0, tau_star, length=2001)
rp = sol_r.(t_plot) |> getindex.(1)
lp = sol_l.(t_plot) |> getindex.(1)
hp = sol_h.(t_plot) |> getindex.(1)
bp = sol_b.(t_plot) |> getindex.(1)
ep = E.(t_plot)

# Trajectories
p1 = plot(t_plot, rp, label="R(t)", lw=2)
plot!(p1, t_plot, lp, label="L(t)", lw=1.8, ls=:dash)
plot!(p1, t_plot, hp, label="H(t)", lw=1.8, ls=:dashdot)
plot!(p1, t_plot, bp, label="B_osc(t)", lw=1.6, ls=:dot)
hline!(p1, [rho], label="rho", color=:gray, ls=:dot)
hline!(p1, [RMAX], label="R_max", color=:gray, ls=:dash)
xlabel!(p1, "t"); ylabel!(p1, "trajectory value")
title!(p1, "Trajectories on [0, tau*] (Julia)")
savefig(p1, joinpath(PLOTS_DIR, "trajectories_plot_julia.pdf"))
savefig(p1, joinpath(PLOTS_DIR, "trajectories_plot_julia.png"))

# Errors vs Envelope
p2 = plot(t_plot, rp - lp, label="R-L", lw=1.8)
plot!(p2, t_plot, hp - rp, label="H-R", lw=1.8, ls=:dash)
plot!(p2, t_plot, abs.(bp - rp), label="|B_osc-R|", lw=1.6, ls=:dashdot)
plot!(p2, t_plot, ep, label="E(t)=ξ/Φ(1-e^{-Φt})", lw=2.2, color=:black)
xlabel!(p2, "t"); ylabel!(p2, "error")
title!(p2, "Errors and envelope (Julia)")
savefig(p2, joinpath(PLOTS_DIR, "error_envelope_plot_julia.pdf"))
savefig(p2, joinpath(PLOTS_DIR, "error_envelope_plot_julia.png"))

println("Plots written to $PLOTS_DIR")

# ---------------------------------------------------------------------------
# Save JSON results
# ---------------------------------------------------------------------------
results = Dict(
    "rho" => rho,
    "Fmax" => Fmax,
    "f_Rmax" => f(RMAX),
    "tau_star" => tau_star,
    "tau_max" => tau_max,
    "m0" => m0,
    "Phi" => Phi,
    "U" => 1/Phi,
    "U_xi" => XI/Phi,
    "E_tau_star" => E(tau_star),
    "H_at_tau_max" => h_at_tmax,
    "max_R_minus_L" => max_rl,
    "max_H_minus_R" => max_hr,
    "max_abs_B_minus_R" => max_br,
    "t_at_max_abs_B" => t_max_br,
    "E_at_max_abs_B" => E(t_max_br),
    "table" => table_rows,
)

open(joinpath(OUT_DIR, "verification_results_julia.json"), "w") do f
    JSON.print(f, results, 2)
end
println("Results written to verification_results_julia.json")