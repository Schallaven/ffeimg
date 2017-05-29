@ECHO OFF
REM Test batch for evaluating a series of files
REM ===========================================

REM First, delete all previous result files
cd mix2
del *.result.png > nul
cd ..

REM copy pngdata from file #1 (separation zone) to everything else, insert additional features (inlets, outlets, physical dimensions), and then process the files

FOR %%I IN (mix2/*.FFE.png) DO (
 echo. 
 pngdata.py --input-file=mix2/out0001.FFE.png --output-file=mix2/%%I --command=copy
 pngdata.py --input-file=mix2/%%I --command=add < features.txt
 findfeatures.py --input-file=mix2/%%I --thresh-binary=25 --sepzone-ratio=0.90 --skip-flowmarkers
 findtrajectories.py --input-file=mix2/%%I --gradient=0.05 --threshbin=20 --threshpen=0.80 --zone-border=50
 rendertrajectories.py --input-file=mix2/%%I --output-file=mix2/%%~nI.result.png
 echo.
)


