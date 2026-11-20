param([string]$OutputDir = (Join-Path $PSScriptRoot 'out\bin'))

$ErrorActionPreference = 'Stop'
$nvcc = (Get-Command nvcc -ErrorAction Stop).Source
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$install = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (!$install) { throw 'Visual Studio x64 C++ Build Tools could not be located.' }
$vcvars = Join-Path $install 'VC\Auxiliary\Build\vcvars64.bat'
$output = (New-Item -ItemType Directory -Path $OutputDir -Force).FullName
$solver = Join-Path $PSScriptRoot 'cuda\solver.cu'
$filter = Join-Path $PSScriptRoot 'cuda\filter.cu'
$command = 'call "{0}" && "{1}" -O3 -std=c++17 -arch=sm_120 "{2}" -o "{3}\solver.exe" && "{1}" -O3 -std=c++17 -arch=sm_120 "{4}" -lcublas -o "{3}\filter.exe"' -f $vcvars, $nvcc, $solver, $output, $filter
Push-Location -LiteralPath $output
try {
    & cmd.exe /d /c $command
    if ($LASTEXITCODE -ne 0) { throw "CUDA build exited $LASTEXITCODE." }
} finally { Pop-Location }
