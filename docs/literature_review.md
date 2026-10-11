# Literature and software audit

Search performed through 8 October 2026. Primary papers, official LBNL material,
and institutional reports were preferred. Search absence is treated cautiously.

## Required-topic coverage

| Topic | Audit result | Boundary for this project |
|---|---|---|
| ISO 15099 | Standard baseline; implemented by WINDOW/pyWinCalc | validate, do not reimplement as contribution |
| LBNL WINDOW | center-glass ISO algorithms plus deflection | authoritative Level 0 |
| LBNL THERM | established 2-D frame/edge solver | public edge benchmark needed |
| pyWinCalc | open ISO bindings; gas mixtures and deflection | automated matched-case reference |
| thin-glass triples | established by Hart/LBNL and later field work | defects/edge coupling, not nominal U |
| edge/spacer bridge | established | use accepted psi-style decomposition |
| IGU edge seals | mature review literature; continuing durability work | generic public surrogate only |
| sealed-cavity convection | mature correlations and CFD literature | quantify correlation validity by Ra |
| conjugate conduction/convection/radiation | established | hierarchy, not novelty |
| argon-air concentration | established thermal and measurement studies | uniform gas loss is a baseline defect |
| low-e degradation | established; decoupling from gas loss remains hard | localized patch is a second interaction defect |
| pane deflection / variable gap | established whole-pane effect | localized perturbation is narrower target |
| manufacturing tolerances | standards document dimensional tolerances; sparse thermal propagation work | use synthetic distributions only |
| transient glazing | 3-D transient work dates to 1987 | promote only if defect coupling adds insight |
| 2-D versus 3-D | both established for windows/thermal bridges | 3-D only as a falsification subset |
| localized envelope defects | quantified in opaque facades (Aissani et al.) | glazing-specific interaction remains uncertain |
| glazing/window UQ | Monte Carlo, Morris, and prediction-uncertainty studies exist | spatial fields/local metrics are narrower |
| high-throughput/GPU thermal work | established generally, little IGU-specific evidence found | benchmark only after scale is real |
| parametric sparse PDEs | mature preconditioner/update literature | application crossover, not method invention |
| batched/repeated solves | mature CPU/GPU literature | measure topology/value reuse and crossovers |

## Standards and software

ISO 15099 defines detailed fenestration thermal calculations. LBNL WINDOW
implements its algorithms and adds glazing deflection. THERM solves 2-D frame and
edge heat transfer. The open pyWinCalc bindings expose ISO 15099 glazing
calculations, gas mixtures, and deflection; ordinary glazing systems exclude
frames/dividers except through its component-model workflow. A custom 1-D model
is therefore a verification baseline, not a contribution. A custom 2-D model is
justified for spatial defect fields and repeated-solve experiments not exposed as
a documented pyWinCalc workflow.

## Thin triples and deflection

Hart, Selkowitz, and Curcija (2019) established that a thin center pane can
substantially improve thermal performance while retaining conventional frame and
sash dimensions. LBNL also warns that further center-of-glass improvements give
diminishing whole-window benefit unless frames and spacers improve. Hart et al.
(2012) measured installed double/triple IGU deflection and showed that gap change
can materially affect U-factor, especially for narrow high-performance triples.
Neither gap sensitivity nor whole-pane deflection is novel. Localized spatial
deviation and its interaction with the edge remains the narrower question.

## Edges, seals, and dimensionality

Van Den Bergh et al. (2013) reviewed IGU spacers and edge seals. Two-dimensional
finite-element/finite-volume treatment of window edges is long established in
Loffler and Buck (1997), Wright's modelling work, ISO 10077 practice, and THERM.
Warm-edge condensation effects are also established. Wakili et al. (2021)
measured center and edge performance in commercial triple IGUs and related
deviations to gas mixture and coating emissivity. Spacer sweeps, edge temperature
plots, and scalar gas/emissivity degradation alone are prior art.

## Cavities, gas loss, and low-e degradation

ISO/WINDOW cavity models use temperature-dependent properties and sealed-cavity
convection correlations; gap conductance is not generally inverse-width because
natural convection can activate. Experiments and CFD compare air and argon gaps,
and recent work studies nondestructive argon measurement. Low-e durability and
emissivity change are established. A 2023 IGU degradation review reports that
existing methods struggle to decouple gas loss, coating degradation, and seal
failure. A localized emissivity patch is useful as a second interaction defect,
but "emissivity affects U" is not novel.

## Transients and UQ

Transient 3-D window models date at least to Shapiro, El Diasty, and Fazio (1987),
including lateral heat flow. Thin glass has lower areal heat capacity, but merely
showing faster response is predictable. Promote transient work only if an
edge/defect interaction produces a new regime or observable.

Monte Carlo, Latin hypercube, Morris, and Sobol methods have been applied to
window U-values, glazing, and building-level fenestration; some work uses
pyWinCalc. UQ itself is established. The less-established target is random
spatial defect fields through a verified multidimensional thin-triple model, with
interaction and local condensation outputs rather than only scalar input spread.

## Decay length

Thin-sheet/fin equations already yield spatial decay scales. More directly,
Kocer et al. (2022/2023) derived and assessed exponential edge decay in vacuum
insulating glass with a length proportional to

`sqrt(k_glass * t_glass / effective_surface_conductance)`.

This is direct prior art against discovering a generic glazing healing length. A
gas-filled multilayer IGU gives coupled sheet equations with multiple eigenmodes;
a single exponential need not be valid. The first layer-aligned synthetic run
does fit one selected tail well (log-space R-squared 0.986), but one fit window is
not evidence of a unique physical length. The study must derive modes, compare
models, vary the fitting window/domain, and predict cases outside the fit.

## Repeated solves and HPC

Preconditioner reuse, low-rank updates, and Krylov recycling for changing systems
are established. GPU batched sparse solvers are also active research; published
work emphasizes aggregation to overcome launch/occupancy costs. "Use a GPU" is
not a contribution. The completed 10,000-field campaign therefore reports the
application crossover and reusable structure rather than claiming a new solver:
fixed topology and a reused nominal factorization are effective, Windows process
ensembles lose at this system size, and GPU batching wins only once enough
independent systems are aggregated. The performance result remains subordinate
to the spatial-defect science and its stated observable tolerances.

## Conclusion

No source in this search combined localized spatial gap defects, gas-filled
thin-triple edge coupling, a validated additivity/decay criterion, and many-query
sparse performance. This is `NOT_ESTABLISHED_IN_SEARCH` and only
`POTENTIALLY_NOVEL`; patents, different vocabulary, or proprietary work may be
close prior art.

Core sources and DOIs are recorded in `references.bib`.
