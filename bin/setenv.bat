@echo off

REM This program sets the path to add the current directory. This way, scripts can be called from everywhere.

cls

set path=%path%;%CD%

echo. Path was set to include current directory.
echo.