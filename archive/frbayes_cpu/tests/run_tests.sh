#!/bin/bash
#SBATCH -J test_models
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH -p compute

# Load required modules
module load python/3.8

# Set the PYTHONPATH
export PYTHONPATH=/home/sakl2/rds/hpc-work/development/frbayes2/frbayes:$PYTHONPATH

# Run the test script
python tests/test_models.py 