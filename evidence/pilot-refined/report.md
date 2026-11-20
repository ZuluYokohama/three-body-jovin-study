# Levi-Civita boundary refinement and precision reruns

Adaptive boundary gate **passed** at 100.00% (6,852/6,852) against the frozen Levi-Civita reference. The target was 90%.

CPU gate **passed** at 83/86 (96.51%) on the original 96-point audit. The target was 95% with at least 48 stable CPU cases. GPU cases that disagreed between the two strictest settings count as misses.

| Reference boundary | Adaptive | Uniform at matched work |
|---|---:|---:|
| Frozen Levi-Civita | 100.00% (6,852/6,852) | 93.30% (6,393/6,852) |
| Precision rerun | 96.67% (7,192/7,440) | 90.66% (6,745/7,440) |
| Frozen original edge | 96.68% (5,422/5,608) | 90.55% (5,078/5,608) |

Boundary sampling spent 631,429,208 of 641,438,440 available force evaluations. Dense and audit reruns spent 1,211,092,692 of 1,421,087,484. The dense precision pass reran 490 trajectories and resolved 392 additional cells under the agreement rule. The adaptive sampler used its own observations. The dense reference was used for scoring.

The GPU reruns tightened local error controls in FP64. These are classification checks at the original time horizon and separation cutoff. Jovin verified the CPU chart and its analytic fixtures and the run manifest binds the CUDA binary, selection code, settings, archive inputs, and output hashes.

Both percentages are pilot targets. Publication claims need a frozen procedure evaluated on independent cases, convergence and uncertainty results, and a matched-work comparison with a reproducible artifact.

[Outcome map](outcome_map.svg) · [Dense precision checks](dense_precision.csv) · [Summary](summary.json)
