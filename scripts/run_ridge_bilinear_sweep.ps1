$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $repo

$outDir = Join-Path $repo "experiments\ridge_bilinear_sweep"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$logPath = Join-Path $outDir "run.log"
$python = Join-Path $repo ".venv\Scripts\python.exe"
$script = Join-Path $repo "scripts\16_ridge_bilinear_sweep.py"

$runArgs = @(
    $script,
    "--embedding-path", "experiments\tenk_minilm_candidate\outputs\embeddings\sentence_transformers__all_MiniLM_L6_v2_n10000_tok160.npy",
    "--out-dir", "experiments\ridge_bilinear_sweep",
    "--n-docs", "10000",
    "--train-docs", "8000",
    "--eval-queries", "1000",
    "--anchor-counts", "256", "384", "512", "768", "1024",
    "--ridge-regs", "0.001", "0.003", "0.01", "0.03", "0.1", "0.3", "1.0",
    "--transforms", "row_zscore", "raw_l2",
    "--targets", "raw",
    "--pools", "10", "25", "50", "100", "250",
    "--time-budget-minutes", "115"
)

"Starting ridge bilinear sweep at $(Get-Date -Format s)" | Tee-Object -FilePath $logPath
"& $python $($runArgs -join ' ')" | Tee-Object -FilePath $logPath -Append
& $python @runArgs 2>&1 | Tee-Object -FilePath $logPath -Append
"Finished ridge bilinear sweep at $(Get-Date -Format s)" | Tee-Object -FilePath $logPath -Append
