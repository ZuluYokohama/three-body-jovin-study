# Reading the three-body experiment

## Why change clocks?

Near a close pass, the force changes rapidly. A physical-time step that works when the bodies are far apart can skip the fastest part of the encounter. The integrator switches to a coordinate chart for a close pair and keeps the third body's gravity in the equations.

In the close-pair chart, the planar separation is $r=u^2$ and physical time obeys

$$
rac{dt}{ds}=|u|^2.
$$

Here $t$ is the original clock and $s$ is the clock used inside the chart. For example, at separation $|r|=0.004$, a small increment $ds=0.01$ covers approximately $dt=0.00004$ in the original time. The code integrates that clock along the step. It enters at separation 0.05 or below when both separations from the third body are at least four times larger. It can return to the original coordinates at separation 0.075 or when that isolation weakens. [Chart implementation](../cuda/levi_civita.hpp) · [Regularization background](https://arxiv.org/abs/1202.0972)

## What is a boundary on the map?

Consider two adjacent starting points from the pilot. Fix the first two masses at $(0,0)$ and $(1,0)$, and leave all three at rest. At $y=0.102734375$, a third mass starting at $x=0.4828125$ triggers the close-approach stop. Move it one grid cell to $x=0.4859375$ and body 1 meets the escape rule. The labels agree across both pilot accuracy settings. Their event times still vary across settings. [Values for this pair](../evidence/pilot-lc/adjacent-example.csv) · [Pilot map](../evidence/pilot-lc/outcome_map.svg)

That shared edge is a boundary in the sampled map. The holdout uses independently integrated points and their cardinal neighbors to confirm a boundary. Unresolved neighbors can leave a boundary unconfirmed. An inherited map label fills a cell the sampler has yet to integrate directly. In the sparse binary128 panel, gray covers cells outside the scoring stencils as well as unresolved outcomes. The maps use model units and their labels apply only to the prescribed cutoff and time horizon.

## Why count work and measure seconds?

A force evaluation counts one call to the integrator's right-hand side, including calls made in rejected steps. Equal counts can still take different wall time. Batching choices, unequal trajectory lengths inside a GPU launch, data transfers, Python scheduling, and reading and writing data all affect seconds. Counting evaluations matches a numerical work budget. Measuring the complete run answers how much time the user waits.

[Example from one measured pair](../evidence/holdout/timing-example.md) · [All timing pairs](../evidence/holdout/timings.csv)

The accuracy comparison and the speed comparison have separate criteria. The independent CPU reference evaluates outcome labels at two binary128 tolerances. The CUDA timing repeats start from scratch. Jovin checks the frozen procedure and scoped CPU fixtures. The timed CUDA subprocesses run outside that supervised CPU check.
