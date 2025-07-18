#!/usr/bin/gnuplot
set terminal postscript eps enhanced "Verdana-Bold" 14 color size 15,10.5
# the above generates an eps file, use convert to make a png:
#   convert -flatten file.eps file.png
# as alternate you can use terminal png

set size 1,0.75 # set aspect ratios
set autoscale # scale axes automatically
unset label # remove any previous labels
set ytic auto # set ytics automatically
set grid # show grid

set title "Derivative"
set xlabel "X"
set ylabel "Y"
set out "derivative.eps"

f(x) = cos(x)

plot [][] \
    f(x) w l, \
    "results.out" u 1:2 t "" w p

!convert -flatten derivative.eps derivative.png
!/bin/rm derivative.eps