@echo off
setlocal
cd /d "%~dp0\.."

set OUTDIR=experiments\relation_router_rerank_cross_encoder
set LOG=%OUTDIR%\run.log
set PY=.venv\Scripts\python.exe

if not exist "%OUTDIR%" mkdir "%OUTDIR%"

set LOKY_MAX_CPU_COUNT=8
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1

echo Starting relation_router_rerank_test cross-encoder at %DATE% %TIME% > "%LOG%"
"%PY%" scripts\22_relation_router_rerank_test.py ^
  --doc-count 100000 ^
  --eval-queries 100 ^
  --out-dir "%OUTDIR%" ^
  --anchor-count 1024 ^
  --anchor-candidate-docs 20000 ^
  --tfidf-max-features 30000 ^
  --ivf-clusters 512 ^
  --ivf-probes 4 ^
  --ef-construction 200 ^
  --raw-low-ef 24 ^
  --relation-ef 128 ^
  --insert-sample 200 ^
  --reranker cross_encoder ^
  --cross-encoder-batch-size 32 ^
  --cross-encoder-max-length 192 >> "%LOG%" 2>&1
set EXITCODE=%ERRORLEVEL%
echo Finished relation_router_rerank_test cross-encoder exit=%EXITCODE% at %DATE% %TIME% >> "%LOG%"
exit /b %EXITCODE%
