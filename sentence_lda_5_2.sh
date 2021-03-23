#!/bin/bash
#
#SBATCH --job-name=VOYA_harshul_ws1
#SBATCH --output=/mnt/nfs/scratch1/hshukla/sentence_results/output_%j.txt
#SBATCH -e /mnt/nfs/scratch1/hshukla/sentence_results/error_%j.txt
#SBATCH --partition=1080ti-long
#SBATCH --gres=gpu:1
#SBATCH --mem=350GB
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1

JOB_TYPE='sentence_lda'
SCRIPT_DIR='/mnt/nfs/work1/696ds-s21/hshukla/voya_project_spring21/'
INPUT='/mnt/nfs/scratch1/hshukla/sentence_results/df_sen_5_2.pkl'
OUTPUT='/mnt/nfs/scratch1/hshukla/sentence_results/'
WS=5
START=2012
END=2015
LABEL="item7_mda"
cd $SCRIPT_DIR
python3 Preprocessing.py --job_type $JOB_TYPE --input $INPUT --output_file $OUTPUT --window_size $WS --start_year $START --end_year $END --pickled --label $LABEL
