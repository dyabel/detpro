
#For training the DetPro, we need to save the embeddings of proposals embedded by the clip image encoder, correponding labels and ious with gt boxes. 
./tools/dist_train.sh  configs/lvis/prompt_save_train.py 8 --work-dir workdirs/prompt_save_train
./tools/dist_train.sh  configs/lvis/prompt_save_val.py 8 --work-dir workdirs/prompt_save_val