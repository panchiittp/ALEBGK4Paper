@echo off
rem ===========================================================================
rem build.bat - Configure and build on Windows.
rem
rem   Scripts\build.bat          CPU backend
rem   Scripts\build.bat cuda     GPU backend
rem
rem Run from a "x64 Native Tools Command Prompt for VS", or any shell where
rem cl.exe and cmake are on PATH.
rem ===========================================================================
setlocal
cd /d "%~dp0.."

set OPTS=
if /I "%1"=="cuda" set OPTS=-DALEBGK_CUDA=ON

cmake -B build %OPTS%
if errorlevel 1 exit /b 1
cmake --build build --config Release -j
if errorlevel 1 exit /b 1

echo.
echo Built: build\bin\Release\alebgk.exe
echo Run with no arguments to list the cases.
endlocal
