#!/bin/bash
#
#SBATCH --job-name=VOYA_harshul
#SBATCH --output=/mnt/nfs/scratch1/hshukla/sentence_results/output_%j.txt
#SBATCH -e /mnt/nfs/scratch1/hshukla/sentence_results/error_%j.txt
#SBATCH --partition=1080ti-short
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=0-4:00:00
#SBATCH --mem=40GB
#SBATCH --ntasks-per-node=8

echo "here"

JOB_TYPE='preprocess_data'
SCRIPT_DIR='/mnt/nfs/work1/696ds-s21/hshukla/voya_project_spring21/'
INPUT='/mnt/nfs/scratch1/hshukla/sentence_results/joined_data.csv'
OUTPUT='/mnt/nfs/scratch1/hshukla/sentence_results/data_sen_1_1.csv'
PPT='sentence_lda'
WS=1
WO=1
cd $SCRIPT_DIR
python3 Preprocessing.py --job_type $JOB_TYPE --input $INPUT --output_file $OUTPUT --preprocess_type $PPT --window_size $WS --window_overlap $WO
