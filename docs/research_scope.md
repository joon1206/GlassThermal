# Research scope and decision record

## Research boundary

This project studies how public, synthetic spatial deviations in a thin-glass
triple-pane IGU alter heat flow, edge temperature, and defect additivity. It does
not recreate a rating program, infer factory tolerances, or encode confidential
product construction.

The first physical question is:

> For a localized gap-width perturbation in a gas-filled thin-triple IGU, does a
> conservative 2-D model exhibit predictable spatial decay, and does defect
> distance from the edge, normalized by that decay, predict when an area-weighted
> or additive correction fails?

The computational question is conditional: once verified interaction maps
require at least 10,000 solves, which reuse or ensemble strategy improves
time-to-solution without changing the numerical result beyond tolerance?

## Audit findings (8 October 2026)

- Repository: shared monorepo with unrelated uncommitted projects. This work is
  isolated in `research/thin_igu_defects`; no existing experiment was changed.
- CPU: AMD Ryzen 5 5600X, 6 cores / 12 threads; memory: 15 GiB.
- GPU: NVIDIA RTX 2080 Ti, 11 GiB; driver reports CUDA 13.1 capability.
- WSL Python 3.14 is bare. Windows Python 3.14 has NumPy, SciPy, pandas,
  matplotlib, and pytest. No MPI, PETSc, FEniCSx, CuPy, or system pyWinCalc was
  detected.
- pyWinCalc 3.3.1 is available from PyPI. A local untracked audit copy was used
  for validation; it is not a required runtime dependency.

## Evidence-gated contribution

The phrase "thermal healing length" is not itself novel. Thin-fin theory already
gives exponential decay scales, and Kocer et al. derived an exponential edge
decay length for vacuum insulating glass. Any contribution must be restricted to
the tested combination of localized gap deviations in gas-filled thin triples,
multiple coupled panes, a realistic edge bridge, and a predictive criterion for
defect additivity.

Search absence is not proof of novelty. The combination is classified
`POTENTIALLY_NOVEL`, not claimed novel.

## Promotion criteria

- The 1-D model agrees with pyWinCalc/WINDOW for a matrix of matched public cases.
- The 2-D model agrees with an analytic composite benchmark and a public
  THERM-compatible edge case.
- Mesh and nonlinear tolerances change interaction metrics by less than 1%.
- Single- and multi-exponential/eigenmode decay models are compared using held-out
  error and stable fitting windows.
- The selected scale predicts additivity outside its calibration cases.
- Interaction is larger than numerical and model-form uncertainty.

## Direction selected

Pursue A+C+D in sequence: establish defect interaction physics (A), add synthetic
spatial UQ after deterministic verification (C), then benchmark repeated sparse
solves after the many-query workload exists (D). Transients remain a separate
follow-on. Localized 3-D work is a falsification tool only if 2-D assumptions fail.
