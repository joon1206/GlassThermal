# Spatial uncertainty and many-query performance results

Run date: 8 October 2026.

## Research boundary

The ensemble retains the validated low-e thin-triple and generic spacer section.
Each 0.30 m realization contains two correlated, smooth 10 mm gap fields with an
18 mm correlation length, 0.35 cross-gap correlation, and a uniformly sampled
5--35% bound passed through a hyperbolic tangent. These are synthetic stress-test
fields, not measured manufacturing tolerances or a proposed probabilistic model
for a particular product. Seed `20261008` exactly regenerates all 10,000 fields.

The committed realization table is stored as two lossless XZ parts named
`validation/uq_10000_cpu_results.csv.xz.part01` and `part02`; plotting scripts
transparently join and read them when an uncompressed local campaign output is
absent.

This scope follows three established findings: thin triples can fit conventional
frames, high-performance triple U-factors can be sensitive to gap deflection,
and edge seals/spacers are a separate thermal problem. The contribution tested
here is the combination of spatial deviations with edge coupling and repeated
solution—not nominal thin-triple performance, deflection physics, or spacer
design by itself.

## Scientific result

The 120x36 production mesh contains 4,320 unknowns. Its earlier 30% localized
defect increment differed from the 240x72 result by 0.25%.

| Statistic | Relative heat-flow change |
|---|---:|
| Mean | +1.221% |
| Median | +0.657% |
| 95th percentile | +4.277% |
| 99th percentile | +6.138% |
| Maximum | +8.950% |

The probability of an absolute change above 0.5%, 1%, and 2% is 54.31%, 41.61%,
and 24.26%, respectively, under this synthetic distribution. Exactly 24.47% of
realizations are thermally unchanged to 1e-8 relative because all deviations
remain on the selected cavity correlation's roughly 9--12.5 mm constant-
conductance branch.

That branch also defeats a nominal first-order sensitivity model: its derivative
is zero at the nominal point, while finite spatial fields can cross into the
conductive branch. The linear surrogate underpredicts heat flow by 1.185% on
average and exceeds 1% absolute error in 41.37% of cases. This is the strongest
scientific reason for the many-query nonlinear campaign.

The sampled amplitude bound has Spearman correlation 0.922 with total heat flow.
Mean widths are much weaker summaries (-0.225 and -0.205), confirming that a
single average gap cannot describe the branch-crossing spatial fields.

## Reuse and CPU ensemble measurements

| Path | Cases/s | Decision |
|---|---:|---|
| Original loop assembly + direct | 13.9 | reference only |
| Fixed CSR topology + direct, matched subset | 78.1 | accept |
| Fixed CSR direct, 600 cases | 96.2 | baseline |
| 2-process direct | 59.1 | reject |
| 4-process direct | 45.6 | reject |
| 6-process direct | 33.3 | reject |
| Nominal-factorization PCG | 450.2 | accept |
| CPU production, 10,000 | 387.8 | accepted oracle |

The reusable PCG path takes a mean 2.86 iterations and at most 6. It differs
from direct heat flow by at most 8.9e-9 and has a maximum relative residual of
9.99e-10 over all 10,000 production cases. The process-ensemble loss is a host-
specific Windows spawn/import result, not a general claim against ensembles.

## Batched GPU result

An optional CuPy 14.2 / CUDA 12.9 environment ran matrix-free, sample-specific
Jacobi-preconditioned CG in float32 on an RTX 2080 Ti. A fixed high-amplitude
probe isolates batching from iteration-count differences.

| Batch | Steady cases/s |
|---:|---:|
| 1 | 3.68 |
| 8 | 38.0 |
| 32 | 148 |
| 128 | 610 |
| 512 | 2,185 |
| 1,024 | 2,517 |
| 2,048 | 2,661 |

The real 10,000-case run uses batches of 2,048, finishes in 4.11 s, and sustains
2,435 cases/s: 6.28x faster than the CPU production oracle. Maximum relative
errors versus CPU are 1.01e-4 for total heat, 1.26e-4 for edge fraction, 1.30e-4
for peak flux, and 5.56e-6 for minimum indoor surface temperature. The maximum
float32 residual is 1.98e-4. GPU results therefore pass the declared 2e-4
observable tolerance but are not a substitute for float64 field-level residuals.

## Decision

Topology reuse and nominal-factorization reuse are retained. Windows process
ensembles are dropped for this problem size. GPU batching is retained only for
large ensembles and declared observables; below roughly 128 simultaneous cases,
launch/iteration overhead removes its advantage. The next science task should
replace the synthetic distribution with measured or mechanically generated gap
fields before increasing sample count further.
