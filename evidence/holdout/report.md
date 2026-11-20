# Held-out Levi-Civita confirmation

We reran the same equal-mass free-fall problem on three shifted grids. All sampled starting points were checked against the pilot archives. Each body was followed to physical time 50, a qualifying escape, or the original close-approach distance of 0.001.

## Accuracy

| Grid | Adaptive boundary | Uniform boundary | GPU agreement with stable CPU | Stable CPU | Confirmed boundary |
|---|---:|---:|---:|---:|---:|
| 1 | 96.62% | 77.25% | 99.67% | 128 | 74 |
| 2 | 96.62% | 94.93% | 96.87% | 127 | 69 |
| 3 | 83.79% | 69.62% | 99.89% | 128 | 72 |

The 90% boundary target **passed** at 91.19% stratum-weighted agreement. Uniform scored 78.55%. The 95% strict GPU agreement target **passed** at 98.81%.

The CPU reference attempted 1873 distinct coordinates. It confirmed 1872 and left 1 unresolved. Unresolved GPU predictions count as misses wherever the independent reference is stable. The accuracy estimates use the frozen sample weights. They describe these grids and the equal-mass free-fall family.

The paired adaptive-minus-uniform 95% spatial-block interval was +1.11 to +27.18 percentage points. The benchmark advantage was **supported**.

## Live GPU timing

Across 21 paired runs, median wall times were 46.86 s for adaptive and 22.39 s for uniform. The median paired uniform/adaptive ratio was 0.479 (95% interval 0.478–0.480). The speedup claim was **unsupported**.

The wall clock includes choosing samples, launching GPU programs, transfers, file handling, and building the map. Force evaluations count calls to the dynamics. Two methods can do similar force work and spend different time on those other operations.

Jovin verified the CPU chart fixtures and the frozen protocol. The independently executed binary128 reference and live CUDA runs carry separate input and output hashes. A label match checks the reported outcome by the specified cutoff and horizon. Event-time differences are recorded separately.

[Scoring data](scored_centers.csv) · [Live timings](timings.csv) · [Frozen protocol](protocol.json) · [Detailed results](summary.json)
