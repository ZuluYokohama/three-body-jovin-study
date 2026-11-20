# Three-body refinement on a GPU

Three equal masses start at rest. Two positions stay fixed at the starting instant, and the third moves across a grid between runs. Each run answers a concrete question. Does a pair pass within $0.001$, does one mass meet the escape criterion, or does the system reach the time limit? The resulting map shows where a small change in the starting position changes the answer.

This study tests whether a sampler that spends more work near those boundaries gives a better map than uniform sampling on the same work budget. Both methods use the same Levi-Civita CUDA integrator. An independent reference integrates the original equations in body coordinates using binary128 arithmetic.

On the three held-out grids, adaptive sampling matched 91.19% of confirmed boundary labels at the same force-work budget, compared with 78.55% for uniform sampling. Grid 3 reached 83.79% on its own. The two strict GPU settings matched 98.81% of stable CPU labels overall. Uniform finished sooner in the paired runs, with a median of 22.39 seconds versus 46.86 seconds for adaptive.

- [Held-out results](evidence/holdout/report.md) and [frozen protocol](evidence/holdout/protocol.json)
- [What the clock, map, and work counts mean](docs/technical-note.md)
- [How to run the study](docs/reproduce.md)
- [Edge pilot](evidence/pilot-edge/report.md), [Levi-Civita pilot](evidence/pilot-lc/report.md), and [refinement pilot](evidence/pilot-refined/report.md)
- [Public viewing and dependency terms](NOTICE.md) and [release file hashes](SHA256SUMS)

The scope is the equal-mass free-fall region and the finite-time rules in the protocol. Jovin binds the inputs and method and verifies scoped CPU checks. GPU trajectories and timings are recorded separately.
