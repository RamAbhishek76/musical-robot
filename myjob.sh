#!/bin/bash

#SBATCH -c 1
#SBATCH -p general
#SBATCH -G a30:1
#SBATCH -N 1
#SBATCH --mem=24G
#SBATCH -t 0-08:00

python ops_custom.py --dataset $1 --segment $2
