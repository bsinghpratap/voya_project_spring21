JOB_TYPE='preprocess_data'
SCRIPT_DIR='/mnt/nfs/work1/696ds-s21/hshukla/voya_project_spring21/'
INPUT='/mnt/nfs/scratch1/hshukla/sentence_results/joined_data.csv'
OUTPUT='/mnt/nfs/scratch1/hshukla/sentence_results/data_sen_5_2.csv'
PPT='sentence_lda'
WS=5
WO=2
cd $SCRIPT_DIR
python3 Preprocessing.py --job_type $JOB_TYPE --input $INPUT --output_file $OUTPUT --preprocess_type $PPT --window_size $WS --window_overlap $WO
