#!/bin/bash

cd fortran
ftnchek -portability=all -nopretty -columns=131 *.f
gfortran -O3 -ffixed-line-length-132 -march=native -ffast-math *.f
./a.out
