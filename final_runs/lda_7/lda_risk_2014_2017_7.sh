#!/bin/bash
#
#SBATCH --job-name=VOYA_harshul_ws1
#SBATCH --output=/mnt/nfs/scratch1/hshukla/final_results/output_%j.txt
#SBATCH -e /mnt/nfs/scratch1/hshukla/final_results/error_%j.txt
#SBATCH --partition=1080ti-long
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=350GB
#SBATCH --ntasks-per-node=8

JOB_TYPE='sentence_lda'
SCRIPT_DIR='/mnt/nfs/work1/696ds-s21/hshukla/voya_project_spring21/'
INPUT='/mnt/nfs/scratch1/hshukla/final_results/data_sen_7_3.pkl'
OUTPUT='/mnt/nfs/scratch1/hshukla/final_results/'
WS=7
START=2014
END=2017
LABEL="item1a_risk"
cd $SCRIPT_DIR
python3 Preprocessing.py --job_type $JOB_TYPE --input $INPUT --output_file $OUTPUT --window_size $WS --start_year $START --end_year $END --label $LABEL --pickled
