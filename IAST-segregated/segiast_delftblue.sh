#!/bin/sh
#
#SBATCH --job-name="segiast_600K"
#SBATCH --partition=compute
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G

module load 2022r2
module load py-numpy
module load py-scipy
module load py-matplotlib
cd iast_wrapper/python
python3 autosegiast.py  
