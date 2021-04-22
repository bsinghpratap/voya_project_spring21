JOB_TYPE='lda'
TARGET='is_dps_cut'
WINDOW_SIZE=1
SCRIPT_DIR='/mnt/nfs/work1/696ds-s21/hshukla/voya_project_spring21/'
INPUT='/mnt/nfs/scratch1/hshukla/final_results/'
OUTPUT='/mnt/nfs/scratch1/hshukla/final_predictions/'
cd $SCRIPT_DIR
python3 Predict.py --job_type $JOB_TYPE --target $TARGET --input_folder $INPUT --output_folder $OUTPUT --window_size $WINDOW_SIZE