param([string]$OutputDir = (Join-Path $PSScriptRoot 'out\extended-precision'))

$ErrorActionPreference = 'Stop'
$destination = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputDir)
if (Test-Path -LiteralPath $destination) { throw "Output directory already exists: $destination" }
New-Item -ItemType Directory -Path $destination | Out-Null
$distro = 'Ubuntu-24.04'
$wslRoot = & wsl.exe -d $distro --exec wslpath -a $PSScriptRoot
$wslOut = & wsl.exe -d $distro --exec wslpath -a $destination
$executable = "$wslOut/ext_precision"
& wsl.exe -d $distro --cd $wslRoot --exec g++ -std=gnu++17 -O3 -Wall -Wextra ext_precision.cpp -lquadmath -o $executable
if ($LASTEXITCODE) { throw "binary128 build failed with exit $LASTEXITCODE" }
$selfCheck = & wsl.exe -d $distro --exec $executable --self-test
if ($LASTEXITCODE) { throw "analytic equilateral check failed with exit $LASTEXITCODE" }
$audit = Join-Path $PSScriptRoot 'out\study-lc\paired_audit.csv'
$cases = Import-Csv -LiteralPath $audit |
    Where-Object { $_.gpu_standard -ne $_.gpu_tight } |
    Sort-Object @{Expression={[double]$_.gpu_time}}, @{Expression={[int]$_.j}}, @{Expression={[int]$_.i}} |
    Select-Object -First 2 i,j,x,y,gpu_standard,gpu_tight,gpu_time
$checks = @(foreach ($case in $cases) {
    foreach ($tol in '1e-20','1e-24') {
        $clock = [System.Diagnostics.Stopwatch]::StartNew()
        $stdout = & wsl.exe -d $distro --exec timeout 45s $executable $case.x $case.y '50' $tol
        $code = $LASTEXITCODE
        $clock.Stop()
        if ($code -notin @(0,1,124)) { throw "binary128 integration failed with exit $code" }
        [ordered]@{
            i = [int]$case.i; j = [int]$case.j; x = $case.x; y = $case.y;
            gpu_standard = [int]$case.gpu_standard; gpu_tight = [int]$case.gpu_tight;
            gpu_tight_time = [double]$case.gpu_time; local_tolerance = $tol;
            exit_code = $code; wall_seconds = $clock.Elapsed.TotalSeconds;
            result = if ($stdout) { $stdout | ConvertFrom-Json } else { $null }
        }
    }
})
$receipt = [ordered]@{
    label_use = 'Out-of-band numerical diagnostic. Does not change either original score.';
    selection = 'Two shortest tight-GPU event times among archived label disagreements. Ties by row then column.';
    audit_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $audit).Hash.ToLowerInvariant();
    source_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $PSScriptRoot 'ext_precision.cpp')).Hash.ToLowerInvariant();
    executable_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $destination 'ext_precision')).Hash.ToLowerInvariant();
    analytic_check = $selfCheck | ConvertFrom-Json;
    horizon = 50; checks = $checks
}
$receipt | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $destination 'diagnostic.json') -Encoding utf8
$receipt | ConvertTo-Json -Depth 6
