# Defect-aware thermal modelling of thin-glass IGUs

Research code for testing when spatial manufacturing deviations in thin-glass
insulating glazing units (IGUs) can be treated as independent local corrections
and when edge coupling requires a multidimensional model.

The present milestone contains:

- a nonlinear 1-D glazing network with solid conduction, gas-gap conduction /
  natural convection, and diffuse-gray long-wave radiation;
- a conservative structured finite-volume solver for 2-D edge sections;
- parameterized localized gap-width defects using a fixed mesh and an equivalent
  local gap conductance;
- analytic component checks, energy-balance tests, mesh refinement, an
  independent public THERM spacer reproduction, and pyWinCalc checks;
- a five-severity / fifteen-pair separation campaign comparing single-, two-,
  and analytical-mode exponential predictions on held-out data;
- a reproducible 10,000-realization spatial UQ campaign with measured fixed-CSR
  reuse, CPU process ensembles, reusable preconditioning, and batched GPU solves;
- an evidence-gated research and HPC plan.

This is not an ISO 15099 implementation and not a rating tool. The gas model is
an explicit, documented research approximation. ISO/WINDOW/pyWinCalc comparison
is a validation target, not a claim of conformance.

## Quick start

Create an isolated environment with Python 3.11 or newer:

```text
cd thin_igu_defects
python -m venv .venv
python -m pip install -e ".[test]"
python -m pytest
python -m thin_igu.experiments
```

The experiment writes machine-readable CSV files under `results/`. Public,
generic inputs live in code; no proprietary tolerances or geometries are used.

## Reproducing the completed benchmark milestone

The pyWinCalc commands require pyWinCalc 3.3.1 and the public product directory
from the official LBNL repository. The THERM check uses its public
`Spacer_CMA.thmx` file.

```text
python benchmarks/therm_spacer_fem_benchmark.py /path/to/Spacer_CMA.thmx --output-directory validation
python benchmarks/pywincalc_low_e_thin_triple.py /path/to/pyWinCalc/examples/products --output validation/pywincalc_low_e_thin_triple.json
python benchmarks/severity_separation_study.py --output-directory validation
python benchmarks/many_query_cpu.py --samples 10000 --output-directory validation
# In a separate CUDA-capable environment, install the optional pinned GPU stack:
python -m pip install -r requirements-gpu.txt
python benchmarks/many_query_gpu.py --samples 10000 --output-directory validation
python benchmarks/plot_many_query_results.py validation
python benchmarks/plot_research_figures.py validation
python benchmarks/check_milestone.py validation
```

The GPU requirement is deliberately separate: CPU verification and the reference
10,000-case campaign do not require CUDA. The pinned wheel bundles CUDA 12.x
runtime components, but still requires a compatible NVIDIA driver.

## Status

The public edge benchmark, low-e thin-triple component check, deterministic
severity/separation study, 10,000-case spatial campaign, and CPU/GPU crossover
measurements are complete. See `docs/benchmark_results.md` and
`docs/many_query_results.md` for results and limitations. The publication-style
figure set is described in `docs/figures.md`.
