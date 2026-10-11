# Research figure guide

Run `python benchmarks/plot_research_figures.py validation` to regenerate three
publication-oriented figure sets under the project-level `figures/` directory
in both PNG and vector PDF formats.

## Synthetic spatial uncertainty

`uq-scientific-evidence` separates the scientific result from solver timing. It
shows the empirical exceedance curve, the severity-conditioned response, the
narrow-gap branch activation that defeats the nominal linear surrogate, and the
prescribed spatial-field envelope. The probabilities apply only to the stated
synthetic distribution; they are not manufacturing defect rates.

## Decay and separation evidence

`decay-separation-evidence` shows the nonlinear severity penalty, the weak
sign-changing return field, blocked spatial-tail errors for all three requested
model families, measured pair non-additivity, held-out separation prediction
errors, and mesh convergence. Together these panels make the negative result
visible: two exponentials win the blocked spatial-tail comparison, but no model
is a universal predictor of spatial decay and pair interaction.

## Many-query execution evidence

`many-query-execution-evidence` shows the measured sequence from scalar assembly
through topology and preconditioner reuse to the final GPU path. It also shows
why Windows process ensembles were rejected, where GPU batching crosses the CPU
production rate, and the error margin against every declared GPU acceptance
gate. Throughput values are hardware- and environment-specific.
