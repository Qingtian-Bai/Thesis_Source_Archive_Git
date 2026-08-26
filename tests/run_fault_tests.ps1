param(
    [string]$PythonExecutable = "D:\python 3.13.7\python.exe"
)

$releaseRoot = Split-Path -Parent $PSScriptRoot
$testScript = Join-Path $PSScriptRoot "test_fault_tolerance.py"

Push-Location $releaseRoot
try {
    & $PythonExecutable $testScript
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
