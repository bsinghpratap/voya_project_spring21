#!/bin/bash
#
#SBATCH --job-name=VOYA_5_predict
#SBATCH --output=/mnt/nfs/scratch1/hshukla/prediction_results/output_%j.txt
#SBATCH -e /mnt/nfs/scratch1/hshukla/prediction_results/error_%j.txt
#SBATCH --partition=2080ti-long
#SBATCH --gres=gpu:1
#SBATCH --mem=250GB
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1

python3 repickle.py
