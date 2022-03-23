#save the embeddings of precomputed proposals encoded by the clip image encoder for reuse，this will take a long time.
./tools/dist_train.sh  configs/lvis/detpro_ens_20e.py 8 --work-dir workdirs/vild_ens_20e_fg_bg_5_10_end --cfg-options model.roi_head.prompt_path=checkpoints/exp/fg_bg_5_10_end_ens.pth model.roi_head.load_feature=False totol_epochs=1
#compress the embeddings into a zip file
zip -r lvis_clip_image_embedding.zip data/lvis_clip_image_embedding/*
./tools/dist_train.sh  configs/lvis/detpro_ens_20e.py 8 --work-dir workdirs/vild_ens_20e_fg_bg_5_10_end --cfg-options model.roi_head.prompt_path=checkpoints/exp/fg_bg_5_10_end_ens.pth model.roi_head.load_feature=True