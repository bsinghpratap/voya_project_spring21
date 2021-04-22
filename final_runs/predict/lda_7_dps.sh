#!/bin/bash
#
#SBATCH --job-name=VOYA_harshul_ws1
#SBATCH --output=/mnt/nfs/scratch1/hshukla/final_predictions/output_%j.txt
#SBATCH -e /mnt/nfs/scratch1/hshukla/final_predictions/error_%j.txt
#SBATCH --partition=1080ti-long
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=150GB
#SBATCH --ntasks-per-node=8

JOB_TYPE='lda'
TARGET='is_dps_cut'
WINDOW_SIZE=7
SCRIPT_DIR='/mnt/nfs/work1/696ds-s21/hshukla/voya_project_spring21/'
INPUT='/mnt/nfs/scratch1/hshukla/final_results/'
OUTPUT='/mnt/nfs/scratch1/hshukla/final_predictions/'
cd $SCRIPT_DIR
python3 Predict.py --job_type $JOB_TYPE --target $TARGET --input_folder $INPUT --output_folder $OUTPUT --window_size WINDOW_SIZE
