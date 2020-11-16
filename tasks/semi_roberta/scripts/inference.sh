python inference.py \
--checkpoint_path [ROOT_SAVE_PATH]/train_student_checkpoint/***.ckpt \
--roberta_path [ROBERT_APTH] \
--batch_size=10 \
--gpus=0,1,2,3 \
--imdb_data_path [IMDB_DATA_PATH]/bin \
--precision 16