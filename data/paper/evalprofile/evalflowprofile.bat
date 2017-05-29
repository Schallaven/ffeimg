@ECHO OFF
REM Test batch for evaluating the flow profile of a chip
REM ====================================================

REM First call all evaluation files for the individual flow measurements
call evalch1.bat
call evalch2.bat
call evalch3.bat
call evalch4.bat
call evalch5.bat

REM Now extract the time-position information out of these files (will create the csv-files)
timeposition.py

REM Finally, combine and render the flow
combineandrenderflow.py


