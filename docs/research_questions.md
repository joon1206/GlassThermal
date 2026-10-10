# Research questions and falsifiable hypotheses

## Primary question

Can the response to a localized gap-width deviation in a gas-filled thin-triple
IGU be reduced to a small set of coupled lateral decay modes, and do those modes
predict when edge and defect penalties are additive?

## Hypotheses

**H1 (modified).** Linearized multilayer sheet equations have several decay
lengths, not necessarily one. A single length is adequate only where the slowest
mode demonstrably dominates.

**H2.** Two defects, or a defect and an edge bridge, are approximately additive
when separation exceeds a threshold in the dominant decay length and severity is
small enough that radiation/cavity nonlinearities remain weak.

**H3.** Additivity fails first for large gap reductions near convection onset,
large emissivity changes, or overlap with the spacer/seal bridge.

**H4.** Thin center glass changes mode amplitudes and fast modes more than the
slowest whole-stack mode. This is tested, not assumed.

**H5.** Fixed topology makes coefficient and symbolic sparse-structure reuse
worthwhile for at least 10,000 samples. Numeric factorization reuse is safe only
for identical matrices or explicitly bounded updates.

## Negative controls

- A uniform gap offset must reproduce the 1-D center solution far from the edge.
- Interaction must vanish with defect amplitude.
- Interaction must approach the discretization floor at large separation.
- The edge perturbation must disappear when spacer and local gap conductivity
  are equal.
- Decay metrics must be insensitive to the distant lateral boundary.

## Metrics

Use energy residual, observed spatial convergence, held-out decay-fit error, and

`I_AB = Q_AB - Q_A - Q_B + Q_0`.

Normalize by

`I_dagger = I_AB / (abs(Q_A-Q_0) + abs(Q_B-Q_0) + epsilon)`

to avoid instability when combined penalties nearly cancel. Include mesh,
property, nonlinear, and correlation uncertainty before interpreting interaction.
