#!/bin/bash
#
#SBATCH --job-name=VOYA_harshul
#SBATCH --output=/mnt/nfs/scratch1/hshukla/final_results/output_%j.txt
#SBATCH -e /mnt/nfs/scratch1/hshukla/final_results/error_%j.txt
#SBATCH --partition=1080ti-long
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=50GB
#SBATCH --ntasks-per-node=8

JOB_TYPE='preprocess_data'
SCRIPT_DIR='/mnt/nfs/work1/696ds-s21/hshukla/voya_project_spring21/'
INPUT='/mnt/nfs/scratch1/hshukla/sentence_results/joined_data.csv'
OUTPUT='/mnt/nfs/scratch1/hshukla/final_results/data_baseline.pkl'
PPT='vanilla_lda'
WS=7
WO=3
cd $SCRIPT_DIR
python3 Processing.py --job_type $JOB_TYPE --input $INPUT --output_file $OUTPUT --preprocess_type $PPT --window_size $WS --window_overlap $WO
