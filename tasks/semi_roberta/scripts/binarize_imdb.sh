# modify paths here
CODE_PATH="[YOUR_CODE_PATH]"
USER_DIR=${CODE_PATH}/preprocess_readers
ROBERTA_PATH="[VANILLA_ROBERTA_LARGE_PATH]"
DATA_DIR="[IMDB_DATA_PATH]"
READER_TYPE="semi_roberta_tokenize"

# sample code
#CODE_PATH="/home/sunzijun/sstc"
#USER_DIR=${CODE_PATH}/preprocess_readers
#ROBERTA_PATH="/data/nfsdata2/sunzijun/loop/roberta-base"
#DATA_DIR="/data/nfsdata2/sunzijun/loop/experiments/train_25k/teacher_data/all_data"
#READER_TYPE="semi_roberta_tokenize"

# export local variable
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=${CODE_PATH}

DATA_BIN=${DATA_DIR}/bin
for phase in 'train' 'test'; do
    INFILE=${DATA_DIR}/${phase}.txt;
    OFILE_PREFIX=${phase};
    shannon-preprocess \
        --input-file ${INFILE} \
        --output-file ${OFILE_PREFIX} \
        --destdir ${DATA_BIN} \
        --user-dir ${USER_DIR} \
        --reader-type ${READER_TYPE} \
        --roberta_base ${ROBERTA_PATH} \
        --max_len 512 \
        --workers 40 \
        --echo
done;