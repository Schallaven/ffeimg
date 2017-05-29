@ECHO OFF
REM Test batch for evaluating a series of files
REM ===========================================

REM First, delete all previous result files
cd flowch1
del *.result.png > nul
cd ..

REM copy pngdata from file #5 (separation zone) to everything else, insert additional features (inlets, outlets, physical dimensions), and then process the files

FOR %%I IN (flowch1/*.FFE.png) DO (
 echo. 
 pngdata.py --input-file=flowch1/out0005.FFE.png --output-file=flowch1/%%I --command=copy
 pngdata.py --input-file=flowch1/%%I --command=add < features.txt
 findfeatures.py --input-file=flowch1/%%I --thresh-binary=16 --sepzone-ratio=0.90 --skip-flowmarkers
 findtrajectories.py --input-file=flowch1/%%I --gradient=0.04 --threshpen=0.80 --threshbin=100 --zone-border=50
 rendertrajectories.py --input-file=flowch1/%%I --output-file=flowch1/%%~nI.result.png
 echo.
)


