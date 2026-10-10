# Validation and verification plan

| Test | Reference | Current state | Acceptance gate |
|---|---|---|---|
| 1-D slab | exact film + slab resistance | automated | relative error <1e-10 |
| multilayer stack | exact series resistance | automated | relative error <1e-10 |
| parallel-plate radiation | diffuse-gray Stefan-Boltzmann result | automated | relative error <1e-12 |
| clear triple / gas cavity | pyWinCalc 3.3.1, official `CLEAR_3.DAT` | one matched case complete | current U difference 0.038%; expand before claims |
| thin low-e triple | pyWinCalc 3.3.1, public `LOW-E_5.LOF` + `CLEAR_3.DAT` | complete | U difference 0.083%; layer/gap diagnostics recorded |
| uniform 2-D slab | exact 1-D solution | automated | relative error <1e-9; energy <1e-10 |
| 2-D edge bridge | public THERM 8 `Spacer_CMA.thmx` | complete | U error 0.097%; final mesh change 0.60% |
| mesh convergence | h, h/2, h/4 | complete for edge and defect increment | defect increment changes 0.0116% medium-to-fine |
| fixed-topology operator | original 2-D FVM | complete | max heat-flow difference 6.72e-12 |
| reused-preconditioner CPU solve | sparse direct solve | complete, 10,000 cases | max heat-flow difference 8.87e-9; residual <1e-9 |
| batched GPU observables | float64 CPU production oracle | complete, 10,000 cases | every reported observable <2e-4 relative error |
| transient timestep | dt, dt/2, dt/4 | not implemented | expected observed order |
| high-fidelity cavity | published CFD/experiment, not same correlation | pending | quantify bias by Ra/gap regime |

## Energy accounting

The steady 2-D solver reports

`epsilon_E = abs(Q_in-Q_out) / max(abs(Q_in), abs(Q_out))`.

Face fluxes use the same conductances as the matrix. Conservation is necessary
but not sufficient: grid convergence and an independent reference are mandatory.

## pyWinCalc protocol

1. Pin pyWinCalc and record its version/source revision.
2. Use bundled public optical data and record coating order.
3. Match temperatures, films, tilt, height, pressure, gas fractions, and widths.
4. Compare layer temperatures, gap conductance, and total U, not only final U.
5. Treat disagreement as a diagnostic; do not tune properties to force a match.

The completed clear-triple check uses three 3.048 mm generic clear panes, two
12.7 mm gaps with 90% argon / 10% air, 294.15/255.15 K air temperatures, and
prescribed 7.7/25 W m-2 K-1 films. The custom U-value is 1.659393 W m-2 K-1 and
pyWinCalc gives 1.658758 W m-2 K-1 (0.0383% relative difference). This validates
one component case only; it is not ISO certification.

The completed low-e check uses two public 4.7244 mm `LOW-E_5.LOF` panes with
coatings on cavity surfaces 2 and 5, a public `CLEAR_3.DAT` center product
assigned a 0.7 mm thickness, and two 10 mm 90% Ar / 10% air gaps. pyWinCalc
gives 1.000286 W m-2 K-1 and the local stack gives 0.999458 W m-2 K-1. Repeating
with the original 3.048 mm center lite gives 0.997913 and 0.997130 W m-2 K-1.
The 0.7 mm center therefore changes U by only +0.238% in this controlled case.

## Public THERM-compatible edge benchmark

`benchmarks/therm_spacer_fem_benchmark.py` parses the official public
`Spacer_CMA.thmx` geometry, four conductivities, two isothermal-film boundaries,
and stored THERM 8.0.13 result. It does not call THERM. A conforming triangular
finite-element mesh reproduces the stored 45.044960 W m-2 K-1 U-factor as
45.001218 W m-2 K-1 at 61,440 triangles (0.0971% difference). The last refinement
changes U by 0.5965%. This validates conduction and boundary integration for one
public spacer component; it does not validate the structured defect geometry or
surface-temperature post-processing independently.

## Defect-decay verification

The completed campaign uses five narrowing severities (0.2--0.6), three pair
severities, five separations (18--120 mm), and three meshes. It compares a single
exponential, a positive two-exponential sum, and three fixed analytical
coupled-sheet modes. Length selection uses only the near-tail block; the far-tail
block is untouched.

The two-exponential model has the smallest far-block error at all five
severities, but errors remain 54--59% for severities 0.3--0.6 (59% at severity
0.2) because the finite-domain perturbation develops a weak sign-changing return
field. Thus this milestone reports **no globally predictive unique decay law**.
Near-tail fit errors are about 0.9--1.2%. Pair interaction is material at 18--30
mm, falls to order 1e-3 by 50--80 mm, and is not reliably predicted by merely
reusing the fitted tail scale. Full values are under `validation/`.

## Many-query verification

The spatial campaign fixes the random seed, mesh, covariance, cross-gap
correlation, and amplitude distribution. The 10,000-case CPU run is the
float64 scientific oracle. Before production, fixed-CSR assembly is compared
against the original finite-volume operator and reused-preconditioner PCG is
compared against sparse direct solves. The production acceptance checks cover
sample count, residual, iteration count, and retained observables.

The optional GPU route is tested last. It uses the same stored gap fields and is
accepted only when total heat, edge heat fraction, peak flux, and minimum indoor
surface temperature each agree with CPU within 2e-4 relative error. Its looser
float32 residual threshold is explicit and does not weaken the CPU oracle.
