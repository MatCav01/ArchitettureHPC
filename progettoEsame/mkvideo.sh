#!/bin/bash

if test $# -ne 3 -a $# -ne 5
then
    echo "Usage: $0 <startiter> <enditer> <stepiter> [<GLX> <GLY>]"
    exit 0
fi

if [[ $# -eq 3 ]];
then
    GLX=512
    GLY=512
else
    GLX=$4
    GLY=$5
fi

module load gnuplot

echo "[mkvideo.sh]: running mkframe.gpl $1 $2 $3 $GLX $GLY..."

cd video

gnuplot -c ../mkframe.gpl $1 $2 $3 $GLX $GLY

convert -delay 50 -loop 0 frame-*_adaptive.png video_adaptive.gif

convert -delay 50 -loop 0 frame-*_fixed.png video_fixed.gif