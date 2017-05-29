@ECHO OFF
REM Test batch for evaluating a series of files
REM ===========================================

REM First, delete all previous result files
cd flowch3
del *.result.png > nul
cd ..

REM copy pngdata from file #1 (separation zone) to everything else, insert additional features (inlets, outlets, physical dimensions), and then process the files

FOR %%I IN (flowch3/*.FFE.png) DO (
 echo. 
 pngdata.py --input-file=flowch3/out0001.FFE.png --output-file=flowch3/%%I --command=copy
 pngdata.py --input-file=flowch3/%%I --command=add < features.txt
 findfeatures.py --input-file=flowch3/%%I --thresh-binary=20 --sepzone-ratio=0.90 --skip-flowmarkers
 findtrajectories.py --input-file=flowch3/%%I --gradient=0.04 --threshpen=0.80 --threshbin=150 --zone-border=50
 rendertrajectories.py --input-file=flowch3/%%I --output-file=flowch3/%%~nI.result.png
 echo.
)


