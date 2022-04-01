
#For training the DetPro, we need to save the embeddings of proposals embedded by the clip image encoder, correponding labels and ious with gt boxes. 
CUDA_VISIBLE_DEVICES=6,7 ./tools/dist_train.sh  configs/lvis/prompt_save_train.py 2 --work-dir workdirs/prompt_save_train
CUDA_VISIBLE_DEVICES=6,7 ./tools/dist_train.sh  configs/lvis/prompt_save_val.py 2 --work-dir workdirs/prompt_save_val
python prompt/gather.py data/lvis_clip_image_proposal_embedding/train train_data.pth
python prompt/gather.py data/lvis_clip_image_proposal_embedding/val val_data.pth