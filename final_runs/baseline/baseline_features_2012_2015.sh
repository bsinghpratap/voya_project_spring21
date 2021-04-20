#!/bin/bash
#
#SBATCH --job-name=VOYA_harshul_ws1
#SBATCH --output=/mnt/nfs/scratch1/hshukla/final_results/output_%j.txt
#SBATCH -e /mnt/nfs/scratch1/hshukla/final_results/error_%j.txt
#SBATCH --partition=1080ti-long
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=100GB
#SBATCH --ntasks-per-node=4

JOB_TYPE='baseline'
SCRIPT_DIR='/mnt/nfs/work1/696ds-s21/hshukla/voya_project_spring21/'
INPUT='/mnt/nfs/scratch1/hshukla/final_results/data_baseline.pkl'
OUTPUT='/mnt/nfs/scratch1/hshukla/final_results/'
START=2012
END=2015
cd $SCRIPT_DIR
python3 Processing.py --job_type $JOB_TYPE --input $INPUT --output_file $OUTPUT --start_year $START --end_year $END --pickled
