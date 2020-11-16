# modify paths here
CODE_PATH="[YOUR_CODE_PATH]"
USER_DIR=${CODE_PATH}/preprocess_readers
ROBERTA_PATH="[VANILLA_ROBERTA_LARGE_PATH]"
DATA_DIR="[REVIEWS_PATH]"
READER_TYPE="roberta_tokenize"

# sample code
#CODE_PATH="/home/sunzijun/sstc"
#USER_DIR=${CODE_PATH}/preprocess_readers
#ROBERTA_PATH="/data/nfsdata2/sunzijun/loop/roberta-base"
#DATA_DIR="/data/nfsdata2/sunzijun/loop/imdb/sunzijun"
#READER_TYPE="roberta_tokenize"

# split data into train and dev
head -n 100 ${DATA_DIR}/reviews.txt > ${DATA_DIR}/dev.txt
tail -n +100 ${DATA_DIR}/reviews.txt > ${DATA_DIR}/train.txt

# export local variable
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=${CODE_PATH}

# binarize data
DATA_BIN=${DATA_DIR}/bin
for phase in 'dev' 'train'; do
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
        --workers 16 \
        --echo
done;