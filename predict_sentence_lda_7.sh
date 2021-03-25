#!/bin/bash
#
#SBATCH --job-name=VOYA_7_predict
#SBATCH --output=/mnt/nfs/scratch1/hshukla/prediction_results/output_%j.txt
#SBATCH -e /mnt/nfs/scratch1/hshukla/prediction_results/error_%j.txt
#SBATCH --partition=2080ti-long
#SBATCH --gres=gpu:1
#SBATCH --mem=150GB
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1

JOB_TYPE='sentence_lda'
LDA_RISK='/mnt/nfs/scratch1/hshukla/sentence_results/sen_lda_item1a_risk_7.model'
LDA_MDA='/mnt/nfs/scratch1/hshukla/sentence_results/sen_lda_item7_mda_7.model'
SCRIPT_DIR='/mnt/nfs/work1/696ds-s21/hshukla/voya_project_spring21/'
INPUT='/mnt/nfs/scratch1/hshukla/sentence_results/df_sen_7_3.pkl'
OUTPUT='/mnt/nfs/scratch1/hshukla/prediction_results/'
WS=7
START=2012
END=2015
PREDICT=2016
cd $SCRIPT_DIR
python3 Predict.py --job_type $JOB_TYPE --input_file $INPUT --output_folder $OUTPUT --lda_risk $LDA_RISK --lda_mda $LDA_MDA --window_size $WS --start_year $START --end_year $END --predict_year $PREDICT --pickled --corpus_filter
