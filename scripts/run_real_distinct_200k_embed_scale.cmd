@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0\.."

set OUTDIR=experiments\real_distinct_hf_code
set LOG=%OUTDIR%\embed_scale.log
set PY=.venv\Scripts\python.exe
set CORPUS=experiments\real_distinct_hf_code\data\hf_code_x_glue_python_distinct_200000.jsonl

if not exist "%OUTDIR%" mkdir "%OUTDIR%"
if not exist "%CORPUS%" (
  echo Missing corpus %CORPUS% > "%LOG%"
  exit /b 1
)

echo Starting real distinct 200k embed/scale at %DATE% %TIME% > "%LOG%"

echo START embed_200k >> "%LOG%"
"%PY%" scripts\20_embed_jsonl_corpus.py ^
  --corpus "%CORPUS%" ^
  --out-embeddings experiments\real_distinct_hf_code\embeddings\hf_code_x_glue_python_distinct_200000_minilm.npy ^
  --batch-size 128 ^
  --max-tokens 128 ^
  --resume >> "%LOG%" 2>&1
if errorlevel 1 (
  echo embed_200k failed with exit code %ERRORLEVEL% >> "%LOG%"
  exit /b %ERRORLEVEL%
)
echo END embed_200k >> "%LOG%"

echo START scale_200k >> "%LOG%"
"%PY%" scripts\17_scale_relation_index.py ^
  --embedding-path experiments\real_distinct_hf_code\embeddings\hf_code_x_glue_python_distinct_200000_minilm.npy ^
  --out-dir experiments\real_distinct_hf_code\relation_index_scale ^
  --sizes 1000 3000 10000 100000 200000 ^
  --anchor-count 1024 ^
  --anchor-candidate-docs 20000 ^
  --ridge-reg 0.03 ^
  --eval-queries 500 ^
  --mechanics-queries 500 ^
  --pools 10 25 ^
  --batch-size 20000 ^
  --score-batch-size 64 >> "%LOG%" 2>&1
if errorlevel 1 (
  echo scale_200k failed with exit code %ERRORLEVEL% >> "%LOG%"
  exit /b %ERRORLEVEL%
)
echo END scale_200k >> "%LOG%"

echo Finished real distinct 200k embed/scale at %DATE% %TIME% >> "%LOG%"
exit /b 0
