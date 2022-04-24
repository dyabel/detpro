
#save the embeddings of precomputed proposals encoded by the clip image encoder for reuse，this will take a long time.
CUDA_VISIBLE_DEVICES=6,7 ./tools/dist_train.sh  configs/lvis/detpro_ens_20e.py 2 --work-dir workdirs/collect_data --cfg-options model.roi_head.load_feature=False total_epochs=1
#compress the embeddings into a zip file
zip -r data/lvis_clip_image_embedding.zip data/lvis_clip_image_embedding/*
#For training the DetPro, we need to save the embeddings of proposals embedded by the clip image encoder, correponding labels and ious with gt boxes. 
# CUDA_VISIBLE_DEVICES=6,7 ./tools/dist_train.sh  configs/lvis/prompt_save_train.py 2 --work-dir workdirs/prompt_save_train
CUDA_VISIBLE_DEVICES=6,7 ./tools/dist_train.sh  configs/lvis/prompt_save_train_reuse.py 2 --work-dir workdirs/prompt_save_train
CUDA_VISIBLE_DEVICES=6,7 ./tools/dist_train.sh  configs/lvis/prompt_save_val.py 2 --work-dir workdirs/prompt_save_val
python prompt/gather.py data/lvis_clip_image_proposal_embedding/train train_data.pth
python prompt/gather.py data/lvis_clip_image_proposal_embedding/val val_data.pth