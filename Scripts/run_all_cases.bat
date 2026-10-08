@echo off
rem ===========================================================================
rem run_all_cases.bat - Run the six cavity cases, or collect cost estimates.
rem
rem   Scripts\run_all_cases.bat                 short runs, for checking the build
rem   Scripts\run_all_cases.bat production      full-length runs
rem   Scripts\run_all_cases.bat estimate        resolution sweep, serial
rem   Scripts\run_all_cases.bat estimate 16     the same sweep on 16 MPI ranks
rem
rem estimate mode advances two time steps per configuration and projects the
rem whole run from the second (the first carries warm-up), then stops.  A sweep
rem therefore costs minutes rather than months, and prints a summary table of
rem projected wall times at the end.
rem
rem   Sweep:  cavity2d  Nx = 400, 600, 800, 1200, 1600
rem           cavity3d  Nx = 20, 40, 60, 80
rem
rem Environment overrides:
rem   NV2D=20   NV3D=20     velocity points per direction
rem   NX2D="400 800"        override the 2-D resolution list
rem   NX3D="20 40"          override the 3-D resolution list
rem   MPIEXEC=<full path to mpiexec.exe>
rem ===========================================================================
setlocal enabledelayedexpansion
cd /d "%~dp0.."

set "MODE=%~1"
if "%MODE%"=="" set "MODE=smoke"
set "NP=%~2"

rem --- locate the solver -----------------------------------------------------
set "BIN="
if defined NP (
    call :find_bin build_mpi build
) else (
    call :find_bin build build_mpi
)
if not defined BIN (
    if defined NP (
        echo MPI solver not built.  Run:  Scripts\build.bat mpi
    ) else (
        echo Solver not built.  Run:  Scripts\build.bat
    )
    exit /b 1
)

if /I "%MODE%"=="estimate" goto :estimate
goto :normal

rem =============================== ESTIMATE ==================================
:estimate
if not defined NV2D set "NV2D=20"
if not defined NV3D set "NV3D=20"
if not defined NX2D set "NX2D=400 600 800 1200 1600"
if not defined NX3D set "NX3D=20 40 60 80"

set "LAUNCH="
if defined NP (
    echo %NP%| findstr /r "^[1-9][0-9]*$" >nul
    if errorlevel 1 (
        echo Rank count must be a positive integer, got '%NP%'.
        exit /b 1
    )
    if not defined MPIEXEC call :find_mpiexec
    if not defined MPIEXEC (
        echo No mpiexec found.  Set MPIEXEC to its full path, or run
        echo the oneAPI setvars.bat first.
        exit /b 1
    )
    set "LAUNCH="!MPIEXEC!" -n %NP%"
    echo Launcher: !MPIEXEC! -n %NP%
)

set "LOGDIR=%CD%\output\estimates"
if not exist "%LOGDIR%" mkdir "%LOGDIR%"
for /f "tokens=* usebackq" %%T in (`powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"`) do set "STAMP=%%T"
set "SUMMARY=%LOGDIR%\summary_%STAMP%.txt"
type nul > "%SUMMARY%"

echo ==============================================
echo   Estimate sweep (two steps per configuration)
echo   Solver: %BIN%
if defined NP echo   Ranks:  %NP%
echo   Logs:   %LOGDIR%
echo ==============================================

for %%N in (%NX2D%) do call :run_one cavity2d 2 %%N %NV2D%
for %%N in (%NX3D%) do call :run_one cavity3d 3 %%N %NV3D%

echo.
echo ================= SUMMARY ====================
type "%SUMMARY%"
echo ==============================================
echo Saved: %SUMMARY%
endlocal
exit /b 0

rem --- one configuration:  %1 case  %2 dim  %3 Nx  %4 Nv ---------------------
:run_one
set "CASE=%~1"
set "DIM=%~2"
set "NX=%~3"
set "NV=%~4"
set "TAG=%CASE%_Nx%NX%_Nv%NV%"
set "LOGTAG=%TAG%"
if defined NP set "LOGTAG=%TAG%_np%NP%"
set "LOG=%LOGDIR%\%LOGTAG%_%STAMP%.log"

rem Per-rank footprint of the replicated cloud, in GiB: g plus the transport
rem buffer, 2*Nv^2 doubles per particle in 2-D and Nv^3 in 3-D.
for /f "tokens=* usebackq" %%M in (`powershell -NoProfile -Command ^
  "$d=%DIM%; $nx=%NX%; $nv=%NV%; if($d -eq 2){$b=32.0*$nx*$nx*$nv*$nv}else{$b=16.0*[double]$nx*$nx*$nx*$nv*$nv*$nv}; '{0:N2}' -f ($b/1GB)"`) do set "MEM=%%M"

echo.
echo --- %CASE%  Nx=%NX%  Nv=%NV%  (cloud %MEM% GiB per replicated rank) ---
if defined NP (
    rem The startup check in main.cpp sizes one rank against the whole node,
    rem so a replicated run can pass it and still be OOM-killed.
    for /f "tokens=* usebackq" %%A in (`powershell -NoProfile -Command ^
      "'{0:N1}' -f ([double]'%MEM%'.Replace(',','') * %NP%)"`) do (
        echo     replicated: %%A GiB across %NP% ranks on one node
    )
)

%LAUNCH% "%BIN%" %CASE% --Nx %NX% --Nv %NV% --estimate 0 > "%LOG%" 2>&1
if errorlevel 1 goto :run_failed

set "LINE="
for /f "tokens=* usebackq delims=" %%L in (`findstr /c:"[ESTIMATE]" "%LOG%"`) do (
    if not defined LINE set "LINE=%%L"
)
if not defined LINE (
    echo     no [ESTIMATE] line produced; see %LOG%
    >>"%SUMMARY%" echo %TAG%  Nv=%NV%  no estimate line ^(see %LOGTAG%_%STAMP%.log^)
    exit /b 0
)
echo     !LINE!
>>"%SUMMARY%" echo %TAG%  !LINE!
exit /b 0

:run_failed
rem Non-zero exit: out of memory, ineligible decomposition, or a launcher
rem mismatch.  Keep sweeping rather than aborting.
echo     FAILED - see %LOG%
for /f "tokens=* usebackq delims=" %%L in (`powershell -NoProfile -Command ^
  "Get-Content -LiteralPath '%LOG%' -Tail 3"`) do echo       ^| %%L
>>"%SUMMARY%" echo %TAG%  Nv=%NV%  FAILED ^(see %LOGTAG%_%STAMP%.log^)
exit /b 0

rem =========================== SMOKE / PRODUCTION ============================
:normal
for %%C in (cavity2d cavity3d cavity2d-square cavity2d-circle cavity3d-sphere cavity3d-cube) do (
    echo.
    echo === %%C ===
    if /I "%MODE%"=="production" (
        rem Case defaults are the production configuration.  The 3-D body cases
        rem checkpoint every 2000 steps: a run interrupted partway resumes with
        rem ALEBGK_RESTART=^<outdir^>\restart.ckpt rather than starting over.
        if "%%C"=="cavity3d-sphere" (
            set "ALEBGK_CHECKPOINT_EVERY=2000" & "%BIN%" %%C
        ) else if "%%C"=="cavity3d-cube" (
            set "ALEBGK_CHECKPOINT_EVERY=2000" & "%BIN%" %%C
        ) else (
            "%BIN%" %%C
        )
    ) else (
        rem Coarse grid, a handful of steps: enough to confirm the case builds,
        rem embeds its body and takes a step.
        if "%%C"=="cavity2d" (
            "%BIN%" %%C --Nx 40 --tfinal 1e-12
        ) else if "%%C"=="cavity3d" (
            "%BIN%" %%C --Nx 15 --tfinal 1e-10
        ) else if "%%C"=="cavity2d-square" (
            "%BIN%" %%C --Nx 30 --tfinal 1.5e-10
        ) else if "%%C"=="cavity2d-circle" (
            "%BIN%" %%C --Nx 30 --tfinal 1.5e-10
        ) else (
            "%BIN%" %%C --Nx 15 --tfinal 1.5e-10
        )
    )
)

echo.
echo All six cases completed.  Output is under output\.
endlocal
exit /b 0

rem ---------------------------------------------------------------------------
:find_bin
for %%D in (%*) do (
    if not defined BIN if exist "%CD%\%%D\bin\Release\alebgk.exe" set "BIN=%CD%\%%D\bin\Release\alebgk.exe"
    if not defined BIN if exist "%CD%\%%D\bin\alebgk.exe"         set "BIN=%CD%\%%D\bin\alebgk.exe"
)
exit /b 0

rem Quoted so the parentheses in "Program Files (x86)" are not batch syntax.
:find_mpiexec
if defined I_MPI_ROOT if exist "%I_MPI_ROOT%\bin\mpiexec.exe" set "MPIEXEC=%I_MPI_ROOT%\bin\mpiexec.exe"
if not defined MPIEXEC (
    for %%D in (
        "C:\Program Files (x86)\Intel\oneAPI\mpi\latest\bin\mpiexec.exe"
        "C:\Program Files\Intel\oneAPI\mpi\latest\bin\mpiexec.exe"
        "C:\Program Files\Microsoft MPI\Bin\mpiexec.exe"
    ) do if not defined MPIEXEC if exist %%D set "MPIEXEC=%%~D"
)
exit /b 0
