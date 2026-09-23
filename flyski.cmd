@echo off
rem Runs flyski_sim.<module> with the flybody conda env python (no conda activate needed).
rem (cmd.exe cannot read UTF-8 Korean comments, so this file stays ASCII.)
rem   flyski play --stage speed --controller snowplow
rem   flyski carving_test --gif
setlocal
set "PY=%USERPROFILE%\miniforge3\envs\flybody\python.exe"
if not exist "%PY%" set "PY=python"
set "PYTHONPATH=%~dp0;%PYTHONPATH%"
set "PYTHONIOENCODING=utf-8"
"%PY%" -W ignore -m flyski_sim.%*
