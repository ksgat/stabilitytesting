@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0\.."

set OUTDIR=experiments\real_distinct_hf_code
set LOG=%OUTDIR%\run.log
set PY=.venv\Scripts\python.exe

if not exist "%OUTDIR%" mkdir "%OUTDIR%"

echo Starting real distinct 200k relation test at %DATE% %TIME% > "%LOG%"

echo START build_distinct_corpus >> "%LOG%"
"%PY%" scripts\19_build_distinct_hf_code_corpus.py ^
  --files python/train-00000-of-00002.parquet python/train-00001-of-00002.parquet ^
  --limits 100000 200000 ^
  --out-dir experiments\real_distinct_hf_code\data ^
  --min-code-chars 40 ^
  --batch-size 8192 >> "%LOG%" 2>&1
if errorlevel 1 (
  echo build_distinct_corpus failed with exit code %ERRORLEVEL% >> "%LOG%"
  exit /b %ERRORLEVEL%
)
echo END build_distinct_corpus >> "%LOG%"

echo START embed_200k >> "%LOG%"
"%PY%" scripts\20_embed_jsonl_corpus.py ^
  --corpus experiments\real_distinct_hf_code\data\hf_code_x_glue_python_distinct_200000.jsonl ^
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

echo Finished real distinct 200k relation test at %DATE% %TIME% >> "%LOG%"
exit /b 0
