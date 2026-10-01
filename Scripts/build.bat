@echo off
rem ===========================================================================
rem build.bat - Configure and build on Windows.
rem
rem   Scripts\build.bat              CPU backend    -> build\
rem   Scripts\build.bat cuda         GPU backend    -> build_cuda\
rem   Scripts\build.bat mpi          MPI backend    -> build_mpi\
rem   Scripts\build.bat mpi intel    MPI, force Intel MPI
rem   Scripts\build.bat mpi ms       MPI, force MS-MPI
rem
rem "mpi" with no second argument uses Intel MPI when oneAPI is installed and
rem falls back to MS-MPI otherwise.  Intel MPI needs I_MPI_ROOT; if it is not
rem already set this script probes the usual oneAPI locations, so running the
rem oneAPI setvars.bat beforehand is optional.
rem
rem Each backend builds in its own directory.  CMake options are sticky once
rem cached, so sharing one directory would silently keep a previous backend
rem enabled.
rem
rem Run from a "x64 Native Tools Command Prompt for VS", or any shell where
rem cl.exe and cmake are on PATH.
rem ===========================================================================
setlocal
cd /d "%~dp0.."

set "BUILDDIR=build"
set "OPTS=-DALEBGK_CUDA=OFF -DALEBGK_MPI=OFF -DALEBGK_INTEL_MPI=OFF"
set "BACKEND=serial CPU"
set "ISMPI="

if /I "%~1"=="cuda" goto :cuda
if /I "%~1"=="mpi"  goto :mpi
if "%~1"=="" goto :configure
echo Unknown backend '%~1'.  Use: cuda ^| mpi ^| mpi intel ^| mpi ms
exit /b 1

:cuda
set "BUILDDIR=build_cuda"
set "OPTS=-DALEBGK_CUDA=ON -DALEBGK_MPI=OFF -DALEBGK_INTEL_MPI=OFF"
set "BACKEND=CUDA"
goto :configure

:mpi
set "BUILDDIR=build_mpi"
set "ISMPI=1"
if /I "%~2"=="ms"    goto :mpi_ms
if /I "%~2"=="msmpi" goto :mpi_ms
if /I "%~2"=="intel" goto :mpi_intel
rem Auto-select: Intel MPI when it is available, MS-MPI otherwise.
if defined I_MPI_ROOT goto :mpi_intel
call :find_impi
if defined I_MPI_ROOT goto :mpi_intel
goto :mpi_ms

:mpi_intel
if not defined I_MPI_ROOT call :find_impi
if not defined I_MPI_ROOT goto :no_impi
if not exist "%I_MPI_ROOT%\include\mpi.h" goto :no_impi
set "OPTS=-DALEBGK_MPI=ON -DALEBGK_INTEL_MPI=ON -DALEBGK_CUDA=OFF"
set "BACKEND=MPI (Intel MPI)"
echo Intel MPI: "%I_MPI_ROOT%"
goto :configure

:mpi_ms
set "OPTS=-DALEBGK_MPI=ON -DALEBGK_INTEL_MPI=OFF -DALEBGK_CUDA=OFF"
set "BACKEND=MPI (MS-MPI)"
echo MS-MPI: located by CMake FindMPI.
goto :configure

:no_impi
echo.
echo Intel MPI was requested but I_MPI_ROOT is not set and oneAPI was not
echo found in the usual locations.  Either:
echo   - run the oneAPI setvars.bat first, or
echo   - set I_MPI_ROOT to the mpi\latest directory, or
echo   - build against MS-MPI instead:  Scripts\build.bat mpi ms
exit /b 1

:configure
echo.
echo === %BACKEND% -^> %BUILDDIR%\ ===
cmake -B "%BUILDDIR%" %OPTS%
if errorlevel 1 exit /b 1
cmake --build "%BUILDDIR%" --config Release -j
if errorlevel 1 exit /b 1

echo.
echo Built: %BUILDDIR%\bin\Release\alebgk.exe
if defined ISMPI goto :mpi_hint
echo Run with no arguments to list the cases.
goto :done

:mpi_hint
echo.
echo Launch with the matching mpiexec - Microsoft MPI and Intel MPI are not
echo interchangeable, and the wrong launcher yields N independent 1-rank runs
echo rather than an error.  For Intel MPI:
echo.
echo   "%%I_MPI_ROOT%%\bin\mpiexec.exe" -n 8 %BUILDDIR%\bin\Release\alebgk.exe cavity2d --Nx 1200 --Nv 20
echo.
echo Add --estimate 0 to project the full run from the first steps and exit.

:done
endlocal
exit /b 0

rem ---------------------------------------------------------------------------
rem Probe the usual oneAPI install locations.  Quoted so the parentheses in
rem "Program Files (x86)" are not read as batch syntax.
rem ---------------------------------------------------------------------------
:find_impi
for %%D in (
    "C:\Program Files (x86)\Intel\oneAPI\mpi\latest"
    "C:\Program Files\Intel\oneAPI\mpi\latest"
) do if exist "%%~D\include\mpi.h" set "I_MPI_ROOT=%%~D"
exit /b 0
