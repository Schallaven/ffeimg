@ECHO OFF
REM Test batch for evaluating a series of files
REM ===========================================

REM First, delete all previous result files
cd flowch5
del *.result.png > nul
cd ..

REM copy pngdata from file #1 (separation zone) to everything else, insert additional features (inlets, outlets, physical dimensions), and then process the files

FOR %%I IN (flowch5/*.FFE.png) DO (
 echo. 
 pngdata.py --input-file=flowch5/out0001.FFE.png --output-file=flowch5/%%I --command=copy
 pngdata.py --input-file=flowch5/%%I --command=add < features.txt
 findfeatures.py --input-file=flowch5/%%I --thresh-binary=10 --sepzone-ratio=0.90 --skip-flowmarkers
 findtrajectories.py --input-file=flowch5/%%I --gradient=0.04 --threshpen=0.90 --threshbin=100 --zone-border=50
 rendertrajectories.py --input-file=flowch5/%%I --output-file=flowch5/%%~nI.result.png
 echo.
)


