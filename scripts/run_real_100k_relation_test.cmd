@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0\.."

set OUTDIR=experiments\real_100k_chunks
set LOG=%OUTDIR%\run.log
set PY=.venv\Scripts\python.exe

if not exist "%OUTDIR%" mkdir "%OUTDIR%"

echo Starting real 100k relation test at %DATE% %TIME% > "%LOG%"

echo START embed_100k >> "%LOG%"
"%PY%" scripts\18_real_chunk_corpus_embed.py ^
  --source-corpus experiments\tenk_minilm_candidate\data\processed\hf_google_code_x_glue_ct_code_to_text_n10000_seed7.jsonl ^
  --out-corpus experiments\real_100k_chunks\data\code_chunks_100k.jsonl ^
  --out-embeddings experiments\real_100k_chunks\embeddings\code_chunks_100k_minilm.npy ^
  --target-chunks 100000 ^
  --chunk-chars 220 ^
  --stride-chars 30 ^
  --min-chars 40 ^
  --batch-size 64 ^
  --max-tokens 128 ^
  --resume >> "%LOG%" 2>&1
if errorlevel 1 (
  echo embed_100k failed with exit code %ERRORLEVEL% >> "%LOG%"
  exit /b %ERRORLEVEL%
)
echo END embed_100k >> "%LOG%"

echo START scale_100k >> "%LOG%"
"%PY%" scripts\17_scale_relation_index.py ^
  --embedding-path experiments\real_100k_chunks\embeddings\code_chunks_100k_minilm.npy ^
  --out-dir experiments\real_100k_chunks\relation_index_scale ^
  --sizes 1000 3000 10000 100000 ^
  --anchor-count 1024 ^
  --anchor-candidate-docs 20000 ^
  --ridge-reg 0.03 ^
  --eval-queries 500 ^
  --mechanics-queries 500 ^
  --pools 10 25 ^
  --batch-size 20000 ^
  --score-batch-size 64 >> "%LOG%" 2>&1
if errorlevel 1 (
  echo scale_100k failed with exit code %ERRORLEVEL% >> "%LOG%"
  exit /b %ERRORLEVEL%
)
echo END scale_100k >> "%LOG%"

echo Finished real 100k relation test at %DATE% %TIME% >> "%LOG%"
exit /b 0
