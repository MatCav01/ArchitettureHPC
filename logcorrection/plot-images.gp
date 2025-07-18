#!/usr/bin/env gnuplot

set terminal png
set view map
set style function pm3d

set palette defined (0.00 'dark-blue', 0.25 'dark-green', \
                     0.50 'orange', 0.75 'dark-red')

set cbrange [:]

set datafile separator whitespace

set output "moon.png"
splot[0:512][0:512] 'moon.in' binary format="%float" array=512x512 with pm3d

set output "moon-corr.png"
splot[0:512][0:512] 'moon.out' binary format="%float" array=512x512 with pm3d