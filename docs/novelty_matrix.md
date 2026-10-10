# Novelty matrix and candidate ranking

Scores run from 1 (weak) to 5 (strong). Novelty means plausibility after this
search, not a claim. Equal weights only force an explicit initial decision.

| Candidate | Classification | Novelty | Depth | Validation | Mfg. | HPC | Publication | Local | Total |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A. interaction / decay physics | POTENTIALLY_NOVEL narrowly; broad healing-length claim is REJECTED_AS_PRIOR_ART | 4 | 5 | 4 | 5 | 3 | 5 | 5 | 31 |
| B. transient thin-glass response | PARTIALLY_ESTABLISHED | 2 | 3 | 3 | 4 | 3 | 3 | 5 | 23 |
| C. manufacturing-tolerance UQ | PARTIALLY_ESTABLISHED; spatial fields less established | 3 | 4 | 3 | 5 | 4 | 4 | 5 | 28 |
| D. repeated sparse solves | ESTABLISHED generally; IGU crossover unknown | 2 | 4 | 5 | 3 | 5 | 3 | 4 | 26 |
| E. localized 3-D edge defects | NOT_ESTABLISHED_IN_SEARCH; difficult validation | 4 | 5 | 2 | 4 | 5 | 4 | 2 | 26 |
| F. staged A+C+D | POTENTIALLY_NOVEL | 4 | 5 | 4 | 5 | 5 | 5 | 4 | **32** |

## Decision

Pursue F as **A first, then C, then D**. This avoids using UQ or acceleration to
amplify an unverified model. E is used only if a 2-D negative control fails. B is
a separate path and should not be mixed into the first result without evidence.

## Claims rejected as prior art

| Claim | Classification | Reason |
|---|---|---|
| Thin triples improve center performance in conventional frames | ESTABLISHED | Hart et al. and LBNL studies. |
| Gap deflection changes triple-pane U-factor | ESTABLISHED | Hart et al. 2012 and WINDOW deflection. |
| Spacers and edge seals are thermal bridges | ESTABLISHED | Decades of 2-D work and review literature. |
| A glazing edge effect can decay exponentially | ESTABLISHED in thin-sheet/VIG settings | Kocer et al. derive it directly. |
| Monte Carlo or Sobol analysis of windows | ESTABLISHED | Multiple window/facade UQ studies. |
| Reusing preconditioners for parametric PDEs | ESTABLISHED | Mature numerical literature. |
| GPU execution is intrinsically faster | REJECTED | Sparse batching is size and hardware dependent. |

The only claim allowed now is: we are testing whether coupled decay modes in a
gas-filled thin-triple edge section predict additivity of localized gap defects.
