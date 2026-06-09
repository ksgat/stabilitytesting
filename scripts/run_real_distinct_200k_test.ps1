$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $repo

& cmd.exe /c "scripts\run_real_distinct_200k_test.cmd"
exit $LASTEXITCODE
