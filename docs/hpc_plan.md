# HPC plan: science first, crossover measured

## Trigger and completed decision

The trigger was met by a reproducible 10,000-realization spatial-gap campaign.
Fixed-topology reuse, CPU process ensembles, reusable preconditioning, and GPU
batching were measured in that order. MPI/domain decomposition remains
unjustified for these independent 4,320-unknown systems.

## Reusable structure

With fixed topology, samples share cell connectivity and CSR index arrays.
Defects change some face conductances and therefore matrix values, not sparsity.
A localized defect changes rows near affected cells; the matrix difference is
sparse but is **not assumed low rank**. Numerical rank and update cost are tests.

## Stages

1. Scalar SciPy assembly and sparse direct solve.
2. Reused topology with vectorized face-coefficient assembly.
3. Serial CG (the matrix is symmetric positive definite) with direct-solve error
   checks.
4. Warm starts and frozen/rebuilt preconditioner crossover, grouping samples by
   severity. Do not reuse a numeric factorization when matrix values change.
5. Process ensembles on 1, 2, 4, and 6 physical cores with BLAS threads limited.
6. MPI only where mpi4py and a runtime exist; ensemble decomposition precedes
   domain decomposition.
7. GPU only for batches large enough to amortize transfers and launches. Audit
   CuPy/JAX sparse support before choosing a library.

## Measurements

Record topology preparation, coefficient assembly, matrix construction, solve,
and post-processing separately; also record samples/s, latency, memory, iteration
count, residual, and error relative to the direct reference. Report strong
scaling `S_p=T_1/T_p`, `E_p=S_p/p`, weak ensemble scaling, and crossovers over
cells/system and systems/batch.

Every run records Python/packages, git revision, CPU/GPU, threads, random seed,
mesh, defect distribution, and tolerances.

## Stop rules

- Reject GPU work if the largest scientific batch loses to CPU ensembles.
- Reject preconditioner reuse if setup savings do not exceed extra iterations or
  any result violates tolerance.
- Reject low-rank-update novelty if rank scales with defect area or the method
  loses to fresh factorization over the real campaign.

## Measured outcome (8 October 2026)

- Original loop assembly/direct solve: 13.9 cases/s on the validation subset.
- Reused CSR topology/direct solve: 78.1 cases/s on the same subset.
- One-process reused-topology direct benchmark: 96.2 cases/s over 600 cases.
- Windows 2/4/6-process ensembles were slower: 59.1/45.6/33.3 cases/s. They are
  rejected for this workload because spawn and import costs dominate.
- A nominal sparse factorization reused as the PCG preconditioner reached 450.2
  cases/s on 600 cases, with maximum heat-flow error 8.9e-9 versus direct.
- The 10,000-case CPU production run reached 387.8 cases/s in 25.79 s.
- Matrix-free float32 CuPy batching on an RTX 2080 Ti reached 2,435 cases/s in
  the production run: 4.11 s and 6.28x speedup over CPU production.
- Maximum GPU discrepancy among reported observables was 1.31e-4 relative to
  the CPU oracle; the float32 algebraic residual floor was 1.98e-4.

GPU execution is accepted for these scalar/local observables at the declared
2e-4 tolerance. It is not accepted as a drop-in field-accuracy replacement for
the float64 CPU solver. Batch size 2,048 is the measured optimum among tested
sizes; batches below roughly 128 do not justify this GPU path.
