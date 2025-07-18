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

# some style definitions
set style line 1 lt 0 pt 3 ps 1.3 lw 4 linecolor rgb "red"
set style line 2 lt 0 pt 4 ps 1.3 lw 4 linecolor rgb "blue"
set style line 3 lt 0 pt 5 ps 1.3 lw 4 linecolor rgb "#FF00FF" # purple
set style line 4 lt 0 pt 6 ps 1.3 lw 4 linecolor rgb "orange"

#------------------------------------
# Time
set title "Time"
set xlabel "Number of Threads"
set ylabel "Time [sec]"
set out "time.eps"

plot [][] \
    "data.dat" u 2:4 t "" ls 3 w lp

# covert eps to png
!convert -flatten time.eps time.png
# delete eps file
!/bin/rm time.eps

#------------------------------------
# Speedup T1/Tn
set title "Relative Speedup"
set xlabel "Number of Threads"
set ylabel "Speedup"
set out "speedup.eps"

A=system("head -n 1 data.dat | awk {'print $4'}")

i(x)=x

plot [][1:] \
    i(x) t "ideal speedup" ls 1, \
    "data.dat" u 2:(A/$4) t "" ls 3 w lp

!convert -flatten speedup.eps speedup.png
!/bin/rm speedup.eps

#------------------------------------
# Efficiency; T1/(n*Tn)
set title "Relative Efficiency"
set xlabel "Number of Threads"
set ylabel "Efficiency"
set out "efficiency.eps"

A=system("head -n 1 data.dat | awk {'print $4'}")

i(x) = 1

plot [][0:1.2] \
    i(x) t "ideal efficiency" ls 1, \
    "data.dat" u 2:(A/($4*$2)) t "" ls 3 w lp

!convert -flatten efficiency.eps efficiency.png
!/bin/rm efficiency.eps