@ECHO OFF
REM Test batch for evaluating a series of files
REM ===========================================

REM First, delete all previous result files
cd mix1
del *.result.png > nul
cd ..

REM Delete old evaluation file
del evaluating.log > nul

REM copy pngdata from file #1 (separation zone) to everything else, insert additional features (inlets, outlets, physical dimensions), and then process the files

FOR %%I IN (mix1/*.FFE.png) DO (
 echo. 
 pngdata.py --input-file=mix1/out0001.FFE.png --output-file=mix1/%%I --command=copy
 pngdata.py --input-file=mix1/%%I --command=add < features.txt
 findfeatures.py --input-file=mix1/%%I --thresh-binary=25 --sepzone-ratio=0.90 --skip-flowmarkers
 findtrajectories.py --input-file=mix1/%%I --gradient=0.025 --threshbin=21 --threshpen=0.90 --zone-border=50 --useinlet=2 --maxoverlap=0.90 --channel=green --minpoints=15 --threshpen=0.25
 rendertrajectories.py --input-file=mix1/%%I --output-file=mix1/%%~nI.result.png
 echo.
)

REM Copy evaluation log into folder
copy evaluating.log mix1


