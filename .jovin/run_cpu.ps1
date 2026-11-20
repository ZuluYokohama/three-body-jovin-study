[CmdletBinding()]
param(
    [string] $OutputDir = (Join-Path (Split-Path -Parent $PSScriptRoot) 'out\jovin-lc'),
    [string] $JovinPython = (Join-Path $env:LOCALAPPDATA 'jovin-harness-dev\venv\Scripts\python.exe'),
    [string] $NativeImage = 'sha256:9e1fa3183ce71070a422e13121a49eab4ada0ce6d0be22a3a57e4439b090ad4a'
)

$ErrorActionPreference = 'Stop'
$study = Split-Path -Parent $PSScriptRoot
$destination = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputDir)
if ((Test-Path -LiteralPath $destination) -and @(Get-ChildItem -LiteralPath $destination -Force).Count -gt 0) {
    throw "Jovin output directory must be new or empty: $destination"
}
if (-not (Test-Path -LiteralPath $JovinPython -PathType Leaf)) {
    throw "Jovin Python was not found: $JovinPython. Pass -JovinPython with the environment containing jovin_harness."
}
if ($NativeImage -notmatch '^sha256:[a-fA-F0-9]{64}$') {
    throw 'NativeImage must be an immutable image SHA-256.'
}

$snapshot = Join-Path $destination 'source'
$snapshotConfig = Join-Path $snapshot '.jovin'
$receiptData = Join-Path $destination 'receipts_data'
New-Item -ItemType Directory -Path $snapshotConfig, $receiptData, (Join-Path $snapshot 'cuda') -Force | Out-Null

$files = [ordered]@{
    'qa/CMakeLists.txt' = 'CMakeLists.txt'
    'qa/verify.cpp' = 'verify.cpp'
    'qa/native_plan.json' = 'native_plan.json'
    'qa/native_ast.json' = 'native_ast.json'
    'qa/native_trace.json' = 'native_trace.json'
    'cuda/levi_civita.hpp' = 'cuda/levi_civita.hpp'
    '.jovin/project.json' = '.jovin/project.json'
}
$inventory = foreach ($file in $files.GetEnumerator()) {
    $sourcePath = Join-Path $study $file.Key
    $targetPath = Join-Path $snapshot $file.Value
    $sourceHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $sourcePath).Hash.ToLowerInvariant()
    Copy-Item -LiteralPath $sourcePath -Destination $targetPath
    $targetHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $targetPath).Hash.ToLowerInvariant()
    if ($sourceHash -cne $targetHash) {
        throw "Source changed during Jovin snapshot: $($file.Key)"
    }
    [ordered]@{ study_path = $file.Key; snapshot_path = $file.Value; sha256 = $targetHash }
}
[ordered]@{
    project_id = 'three-body-levi-civita-cpu'
    scope = 'Portable production chart, independent Newtonian CPU verifier, and analytic binary crossing'
    native_image = $NativeImage
    files = @($inventory)
} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $destination 'scope.json') -Encoding utf8

function Invoke-JovinJson {
    param([string] $Name, [string[]] $Arguments)

    $receiptPath = Join-Path $destination "$Name.json"
    $stderrPath = Join-Path $destination "$Name.stderr.log"
    & $JovinPython -B -m jovin_harness @Arguments 1> $receiptPath 2> $stderrPath
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "Jovin $Name exited with code $exitCode. See $receiptPath and $stderrPath."
    }
    return Get-Content -LiteralPath $receiptPath -Raw | ConvertFrom-Json
}

$previousRuntime = $env:JOVIN_HARNESS_RUNTIME
$previousImage = $env:JOVIN_HARNESS_NATIVE_IMAGE
$previousData = $env:JOVIN_HARNESS_DATA_DIR
try {
    $env:JOVIN_HARNESS_RUNTIME = 'podman'
    $env:JOVIN_HARNESS_NATIVE_IMAGE = $NativeImage
    $env:JOVIN_HARNESS_DATA_DIR = $receiptData

    $status = Invoke-JovinJson 'native_status' @('native', 'status')
    if (-not $status.ready) {
        throw "Jovin's supervised native runner is unavailable. See $destination\native_status.json."
    }

    $baseline = Invoke-JovinJson 'baseline' @('project', 'baseline', '--root', $snapshot)
    $ast = Invoke-JovinJson 'ast' @('ast', 'check', '--root', $snapshot,
        '--baseline-receipt', $baseline.receipt_id, '--plan', 'native_ast.json')
    $trace = Invoke-JovinJson 'trace' @('trace', 'check', '--root', $snapshot,
        '--baseline-receipt', $baseline.receipt_id, '--ast-receipt', $ast.receipt_id,
        '--plan', 'native_trace.json')
    $project = Invoke-JovinJson 'project' @('project', 'check', '--root', $snapshot,
        '--baseline-receipt', $baseline.receipt_id, '--ast-receipt', $ast.receipt_id,
        '--trace-receipt', $trace.receipt_id, '--native-receipt', $trace.native_receipt_id)
    $executionId = $trace.native_result.stage_receipts[-1]
    $execution = Invoke-JovinJson 'execution' @('receipts', 'show', $executionId)
    $parityCheck = @($execution.result.execution.steps | Where-Object { $_.id -eq 'projection_parity' })
    if ($parityCheck.Count -ne 1 -or $parityCheck[0].exit_code -ne 0) {
        throw 'Expected one successful projection_parity check in the Jovin runtime receipt.'
    }
    $parityMetrics = $parityCheck[0].stdout_tail | ConvertFrom-Json
    if (-not $parityMetrics.passed -or -not $parityMetrics.levi_civita.passed) {
        throw 'The Jovin runtime receipt did not report passing projection and Levi-Civita metrics.'
    }
    [ordered]@{
        execution_receipt = $executionId
        source_manifest_sha256 = $project.source_manifest_sha256
        metrics = $parityMetrics
    } | ConvertTo-Json -Depth 7 | Set-Content -LiteralPath (Join-Path $destination 'metrics.json') -Encoding utf8
    $chain = Invoke-JovinJson 'chain' @('receipts', 'verify')

    [ordered]@{
        baseline_receipt = $baseline.receipt_id
        ast_receipt = $ast.receipt_id
        native_receipt = $trace.native_receipt_id
        trace_receipt = $trace.receipt_id
        project_receipt = $project.receipt_id
        chain = $chain
    } | ConvertTo-Json -Depth 10
} finally {
    $env:JOVIN_HARNESS_RUNTIME = $previousRuntime
    $env:JOVIN_HARNESS_NATIVE_IMAGE = $previousImage
    $env:JOVIN_HARNESS_DATA_DIR = $previousData
}
