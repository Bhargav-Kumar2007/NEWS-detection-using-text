@echo off
REM Windows helper.  Usage: run.bat <command> [extra args passed to the Python script]
setlocal EnableDelayedExpansion
cd /d "%~dp0"

set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"

set "CMD=%~1"
set "REST="
for /f "tokens=1,* delims= " %%a in ("%*") do set "REST=%%b"

if "%CMD%"=="" goto usage
if /i "%CMD%"=="setup" goto setup
if /i "%CMD%"=="download" ( "%PY%" download_data.py %REST% & goto end )
if /i "%CMD%"=="train-small" ( "%PY%" train.py --mode small %REST% & goto end )
if /i "%CMD%"=="train-full" ( "%PY%" train.py --mode full %REST% & goto end )
if /i "%CMD%"=="evaluate" ( "%PY%" evaluate.py %REST% & goto end )
if /i "%CMD%"=="predict" ( "%PY%" predict.py %REST% & goto end )
if /i "%CMD%"=="test" ( "%PY%" tests\test_smoke.py & goto end )
goto usage

:setup
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
REM Optional GPU wheel index: set TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 before running setup
if defined TORCH_INDEX_URL .venv\Scripts\python.exe -m pip install torch --index-url %TORCH_INDEX_URL%
.venv\Scripts\python.exe -m pip install -r requirements.txt
goto end

:usage
echo Usage: run.bat {setup^|download^|train-small^|train-full^|evaluate^|predict^|test} [args]
echo Examples:
echo   run.bat setup
echo   run.bat download
echo   run.bat train-small --model textcnn
echo   run.bat evaluate --run-dir outputs\bilstm_small
echo   run.bat predict --text "Scientists discover a cat that can speak fluent French"

:end
endlocal
