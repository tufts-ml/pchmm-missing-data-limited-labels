#!/usr/bin/env bash
#SBATCH -n 1                         # Number of cores
#SBATCH -t 0-24:00                   # Runtime in D-HH:MM
#SBATCH -p doshi-velez               # Partition to submit to
#SBATCH --mem-per-cpu $XHOST_MEM_MB  # Memory (in MB) per cpu
#SBATCH -o $XHOST_LOG_DIR/%A.%a.out
#SBATCH -e $XHOST_LOG_DIR/%A.%a.err
#SBATCH --array=$XHOST_FIRSTTASK-$XHOST_NTASKS
#SBATCH --export=ALL

bash $XHOST_BASH_EXE
