# Three-body projection and refinement

The NVIDIA GeForce RTX 5070 study sampled a 256 × 256 initial-condition plane for equal masses released from rest, with a horizon of t = 50.

Jovin CPU projection check `20261119T215006Z-bd6dd43deb38` passed. The GPU short-time state check had a maximum normalized error of 2.66e-14. On the long-horizon CPU audit, 78 of 86 stable CPU classifications matched the tight GPU run across 96 checked cases.

The full-grid reference retained 91.55% of cells after tolerance and reference checks. Stable here means the two GPU tolerance settings agreed on the label and passed the energy drift screen, with sampled discrepancies from the CPU audit excluded. Unresolved predictions count against agreement on stable reference cells.

| Sampling | Force evaluations | Agreement | Boundary agreement | Coverage | Wall time |
|---|---:|---:|---:|---:|---:|
| Adaptive | 1,568,242,664 | 99.04% | 89.87% | 100.00% | 14.51 s |
| Uniform at matched budget | 1,568,211,264 | 96.46% | 76.41% | 98.90% | 18.67 s |

Adaptive disagreed on 1.05% of 764 stable probes in coarse-map interior cells. The preregistered work-savings gate was **not met**.

The real 65,536-neighborhood filter produced identical masks. Median time was 0.1679 ms for fused CUDA and 0.2042 ms for TF32 cuBLAS. Nsight counter status was `counter_permission_denied`. Nsight Systems status was `tensor_kernel_observed`.

The escape labels are finite-threshold observations. The close-approach label stops at pair separation 0.001. Gray cells are unresolved at the tested accuracy.

[Outcome map](outcome_map.svg) · [Detailed results](summary.json) · [CPU cross-checks](cpu_audit.csv) · [Blind probes](interior_probes.csv)
