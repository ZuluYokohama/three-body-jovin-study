# Reproduce the study

## Environment

The GPU runner uses Windows, Python 3.12, NumPy and SciPy, an NVIDIA CUDA toolkit with `nvcc`, and Visual Studio x64 C++ Build Tools. CUDA device 0 was the selected device for the reported runs. The build targets `sm_120`. The independent CPU reference uses WSL Ubuntu 24.04, GNU C++ and libquadmath. Use a Python installation with the dependencies listed in [requirements.txt](../requirements.txt).

Jovin is a separate dependency. Its executable or Python environment and the configured native runtime are required to create new supervised receipts. The public bundle includes receipt summaries and input hashes. The internal native store is kept with the original run.

## Build and replay the pilots

Choose new output directories for replay. The commands below use the folder names expected by the later default arguments. Run them from the repository root in PowerShell.

```powershell
python -m pip install -r requirements.txt
.\build_cuda.ps1 -OutputDir out\bin-lc
.\.jovin\run_cpu.ps1 -OutputDir out\jovin-lc -JovinPython $env:JOVIN_PYTHON
python -B study.py --bin out/bin-lc --jovin out/jovin-lc --out out/study
python -B study.py --coordinates levi-civita --bin out/bin-lc --baseline out/study --jovin out/jovin-lc --out out/study-lc
python -B precision_audit.py --out out/precision-audit-refined
python -B refined_study.py --jovin out/jovin-lc --out out/study-lc-refined
```

Set `JOVIN_PYTHON` to the executable in your installed Jovin environment before launching the CPU verification. Use `python -B study.py --help` and `python -B refined_study.py --help` for replay arguments. A fresh build or run can produce different binary or receipt hashes. Compare physical results and measured timing distributions separately from those byte identifiers.

## Reproduce the holdout

The prepare phase builds the independent oracle and freezes the cohort. Verify the protocol before starting CUDA runs. The execute phase runs the GPU timings, the strict GPU audit, and the independent CPU reference in separate windows, then scores and writes the report. Use a clean output directory and an idle GPU.

~~~powershell
python -B holdout.py prepare --out out/holdout-confirmation --pilot . --bin out/bin-lc
.\verify_holdout.ps1 -Protocol out\holdout-confirmation\protocol.json -OutputDir out\jovin-holdout -JovinPython $env:JOVIN_PYTHON
python -B holdout.py execute --out out/holdout-confirmation --pilot . --bin out/bin-lc --jovin out/jovin-holdout
~~~

The pilot replay above produces the local archives used by the zero-shift check. A clean rebuild should keep one compiled binary for both the refined pilot and the holdout. The runner checks that their recorded binary hashes agree.

The [frozen protocol](../evidence/holdout/protocol.json) records the grid shifts, cohort selection, reference tolerances, and timing rules. The CSVs in [held-out evidence](../evidence/holdout/) let you examine the scored points, confirmed reference labels, and repeated timings without a CUDA build. [SHA256SUMS](../SHA256SUMS) identifies the published files. The published input and receipt views omit local paths and device identifiers. Original digest values in those views identify the local evidence before filtering.
