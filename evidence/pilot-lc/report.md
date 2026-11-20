# Levi-Civita three-body study

The NVIDIA GeForce RTX 5070 study sampled a 256 × 256 initial-condition plane for equal masses released from rest, with a horizon of t = 50.

Jovin CPU projection check `20261119T220403Z-d278d25ad982` passed. The GPU short-time state check had a maximum normalized error of 2.66e-14. On the long-horizon CPU audit, 79 of 86 stable CPU classifications matched the tight GPU run across 96 checked cases.

The full-grid reference retained 98.46% of cells after tolerance and reference checks. Stable here means the two GPU tolerance settings agreed on the label and passed the energy drift screen, with sampled discrepancies from the CPU audit excluded. Unresolved predictions count against agreement on stable reference cells.

| Sampling | Force evaluations | Agreement | Boundary agreement | Coverage | Wall time |
|---|---:|---:|---:|---:|---:|
| Adaptive | 926,804,224 | 98.55% | 86.46% | 100.00% | 16.18 s |
| Uniform at matched budget | 926,713,960 | 96.18% | 72.27% | 99.93% | 16.47 s |

Adaptive disagreed on 0.95% of 846 stable probes in coarse-map interior cells. The preregistered work-savings gate was **not met**.

The real 65,536-neighborhood filter produced identical masks. Median time was 0.1456 ms for fused CUDA and 0.2200 ms for TF32 cuBLAS. Nsight counter status was `counter_permission_denied`. Nsight Systems status was `trace_unavailable`.

The escape labels are finite-threshold observations. The close-approach label stops at pair separation 0.001. Gray cells are unresolved at the tested accuracy.

[Outcome map](outcome_map.svg) · [Detailed results](summary.json) · [CPU cross-checks](cpu_audit.csv) · [Blind probes](interior_probes.csv)

## Validation against the original targets

- Boundary gate **failed** with 5,924 of 6,852 labels agreeing (86.46%, target 90%).
- CPU gate **failed** with 79 of 86 stable labels agreeing (91.86%, target 95% with at least 48 stable).
- On the frozen original boundary, the new adaptive map agreed on 4,687 of 5,608 points (83.58%). The archived map agreed on 5,040 of 5,608 points (89.87%).
- On the original 8 CPU disagreements, 5 now agree. The paired audit replayed the same 96 coordinates and recorded 0 changes to the two CPU labels on replay.

| Initial-condition cell | CPU outcome | Original GPU outcome | Levi-Civita GPU outcome |
|---|---|---|---|
| (171, 114) | Close approach | Body 3 escapes | Body 2 escapes |
| (109, 95) | Close approach | Body 1 escapes | Horizon reached |
| (178, 80) | Close approach | Body 2 escapes | Close approach |
| (173, 103) | Close approach | Body 2 escapes | Close approach |
| (187, 119) | Close approach | Body 1 escapes | Horizon reached |
| (64, 154) | Close approach | Horizon reached | Close approach |
| (129, 205) | Close approach | Body 1 escapes | Close approach |
| (185, 100) | Close approach | Horizon reached | Close approach |

The separate analytic collision fixture crossed the isolated binary in regularized time. The scored trajectories use the original physical-time stop at separation 0.001.

[Paired audit with event times and separations](paired_audit.csv) · CUDA collision fixture
