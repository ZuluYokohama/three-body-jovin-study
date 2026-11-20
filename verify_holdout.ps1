[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Protocol,
    [string]$OutputDir = (Join-Path $PSScriptRoot 'out\jovin-holdout'),
    [string]$JovinPython = (Join-Path $env:LOCALAPPDATA 'jovin-harness-dev\venv\Scripts\python.exe'),
    [string]$NativeImage = 'sha256:9e1fa3183ce71070a422e13121a49eab4ada0ce6d0be22a3a57e4439b090ad4a'
)

$ErrorActionPreference = 'Stop'
$protocolPath = (Resolve-Path -LiteralPath $Protocol).Path
$target = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputDir)
if (Test-Path -LiteralPath $target) { throw 'Use a new Jovin output directory.' }
if (-not (Test-Path -LiteralPath $JovinPython -PathType Leaf)) { throw 'Supply -JovinPython for the installed Jovin runtime.' }
$contract = Get-Content -LiteralPath $protocolPath -Raw | ConvertFrom-Json
$snapshot = Join-Path $target 'source'
New-Item -ItemType Directory -Path (Join-Path $snapshot '.jovin'), (Join-Path $snapshot 'cuda') | Out-Null
$pythonFiles = @('study.py','reference.py','refined_study.py','precision_audit.py','boundary_refine.py',
    'lc_checks.py','holdout.py','holdout_gpu.py','holdout_oracle.py')
$native = [ordered]@{'qa/CMakeLists.txt'='CMakeLists.txt'; 'qa/verify.cpp'='verify.cpp';
    'qa/native_plan.json'='native_plan.json'; 'qa/native_ast.json'='native_ast.json';
    'qa/native_trace.json'='native_trace.json'; 'cuda/levi_civita.hpp'='cuda/levi_civita.hpp';
    'holdout_oracle.cpp'='holdout_oracle.cpp'}
foreach ($file in $pythonFiles) { $native[$file] = $file }
foreach ($file in $native.Keys) {
    $original = Join-Path $PSScriptRoot $file
    $observed = (Get-FileHash -LiteralPath $original -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($observed -cne $contract.source_sha256.PSObject.Properties[$file].Value) {
        throw "Frozen source hash mismatch: $file"
    }
    Copy-Item -LiteralPath $original -Destination (Join-Path $snapshot $native[$file])
}
Copy-Item -LiteralPath $protocolPath -Destination (Join-Path $snapshot 'protocol.json')
$config = Get-Content -LiteralPath (Join-Path $PSScriptRoot '.jovin\project.json') -Raw | ConvertFrom-Json
$config.project_id = 'three-body-held-out-confirmation'
$config.objective = 'Verify the Levi-Civita CPU fixtures and bind the held-out cohort, method, reference source, and matched-work timing protocol.'
$config.partitions += @{id='python_protocol'; language='python'; paths=$pythonFiles; depends_on=@()}
$config.partitions += @{id='holdout_manifest'; language='build'; paths=@('protocol.json','holdout_oracle.cpp'); depends_on=@('python_protocol')}
$config.required_gates += 'python_ast'
$config | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $snapshot '.jovin\project.json') -Encoding utf8

function Invoke-Receipt([string]$Name, [string[]]$Arguments) {
    $receipt = Join-Path $target "$Name.json"
    & $JovinPython -B -m jovin_harness @Arguments 1> $receipt 2> (Join-Path $target "$Name.stderr.log")
    if ($LASTEXITCODE -ne 0) { throw "Jovin $Name failed. See $receipt." }
    Get-Content -LiteralPath $receipt -Raw | ConvertFrom-Json
}
$before = @{runtime=$env:JOVIN_HARNESS_RUNTIME; image=$env:JOVIN_HARNESS_NATIVE_IMAGE; data=$env:JOVIN_HARNESS_DATA_DIR}
try {
    $env:JOVIN_HARNESS_RUNTIME = 'podman'
    $env:JOVIN_HARNESS_NATIVE_IMAGE = $NativeImage
    $env:JOVIN_HARNESS_DATA_DIR = Join-Path $target 'receipts_data'
    $status = Invoke-Receipt native_status @('native','status')
    if (-not $status.ready) { throw 'Jovin native runner is unavailable.' }
    $baseline = Invoke-Receipt baseline @('project','baseline','--root',$snapshot)
    $ast = Invoke-Receipt ast @('ast','check','--root',$snapshot,'--baseline-receipt',$baseline.receipt_id,'--plan','native_ast.json')
    $trace = Invoke-Receipt trace @('trace','check','--root',$snapshot,'--baseline-receipt',$baseline.receipt_id,'--ast-receipt',$ast.receipt_id,'--plan','native_trace.json')
    $project = Invoke-Receipt project @('project','check','--root',$snapshot,'--baseline-receipt',$baseline.receipt_id,'--ast-receipt',$ast.receipt_id,'--trace-receipt',$trace.receipt_id,'--native-receipt',$trace.native_receipt_id)
    $chain = Invoke-Receipt chain @('receipts','verify')
    if (-not $project.passed -or -not $chain.valid) { throw 'Jovin project or receipt chain failed.' }
    Write-Output "Verified held-out protocol $($project.receipt_id) with $($chain.count) receipts."
} finally {
    $env:JOVIN_HARNESS_RUNTIME = $before.runtime
    $env:JOVIN_HARNESS_NATIVE_IMAGE = $before.image
    $env:JOVIN_HARNESS_DATA_DIR = $before.data
}

