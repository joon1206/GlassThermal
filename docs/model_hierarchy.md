# Model hierarchy

## Level 0 - authoritative center baseline

Use pyWinCalc/WINDOW ISO 15099 calculations with public glazing data. This sets
center U-value, layer temperatures, gap conductances, and gas-mixture references.
No custom result supersedes rating software.

## Level 1 - transparent network

Implemented in `thin_igu.network`. Solids are exact resistances; gas gaps combine
a sealed-cavity equivalent conductance with exact diffuse-gray parallel-plate
radiation. Interface temperatures are solved nonlinearly. The present property
mixing and correlation are an auditable research approximation, not ISO
conformance.

## Level 2 - conservative 2-D edge section

Implemented in `thin_igu.fvm2d` as cell-centred finite volumes. Harmonic face
conductivities preserve material-interface fluxes. Indoor/outdoor films are Robin
boundaries and lateral faces are adiabatic. Gaps use equivalent conductivity for
gas plus linearized radiation; edge cells use a generic spacer/seal surrogate.

A gap deviation is represented on fixed topology by choosing local mesh
conductivity so its through-gap conductance equals that of the desired gap. This
retains lateral glass spreading and a common sparsity pattern, but neglects sloped
interfaces and resolved gas flow. A deformed-geometry or CFD subset must bound
that model-form error.

## Level 3 - transient 2-D

Deferred. Add volumetric heat capacity and backward Euler only after steady edge
validation. Crank-Nicolson is a convergence comparison, not the default.

## Level 4 - localized 3-D / resolved cavity

Deferred falsification tool. Use only if out-of-plane shape or local convection
invalidates Level 2; do not use it as the ensemble engine.

## Level 5 - many-query ensemble

Deferred until Levels 0-2 agree and deterministic metrics are stable. Samples
retain identical topology where possible. Every optimized result is compared to
the serial SciPy reference within a declared temperature/flux tolerance.
