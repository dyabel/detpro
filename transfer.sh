#Generate class embedding for tranfer Datasets(take objects365 as example)
python gen_cls_embedding.py checkpoints/exp/fg_bg_5_5_6_endepoch6_prompt.pth checkpoints/obj365/fg_bg_5_5_6_obj365 obj365
python gen_cls_embedding.py checkpoints/exp/fg_bg_5_6_7_endepoch6_prompt.pth checkpoints/obj365/fg_bg_5_5_6_obj365 obj365
python gen_cls_embedding.py checkpoints/exp/fg_bg_5_7_8_endepoch6_prompt.pth checkpoints/obj365/fg_bg_5_5_6_obj365 obj365
python gen_cls_embedding.py checkpoints/exp/fg_bg_5_8_9_endepoch6_prompt.pth checkpoints/obj365/fg_bg_5_5_6_obj365 obj365
python gen_cls_embedding.py checkpoints/exp/fg_bg_5_9_10_endepoch6_prompt.pth checkpoints/obj365/fg_bg_5_5_6_obj365 obj365
python run.py test data/lvis_clip_image_proposal_embedding/train data/lvis_clip_image_proposal_embedding/val checkpoints/obj365 fg_bg_5_10_obj365 soft 0.0 0.5 0.5 checkpoints/obj365/fg_bg_5_5_6_obj365.pth checkpoints/obj365/fg_bg_5_6_7_obj365.pth checkpoints/obj365/fg_bg_5_7_8_obj365.pth checkpoints/obj365/fg_bg_5_8_9_obj365.pth checkpoints/obj365/fg_bg_5_9_10_obj365.pth
# Transfer to other datasets
./tools/dist_test.sh  configs/transfer/transfer_voc.py workdirs/vild_ens_2x_neg5_ens/epoch_24.pth 8 --eval mAP --cfg-options model.roi_head.load_feature=False model.roi_head.prompt_path=checkpoints/voc/fg_bg_6_10_voc_neg30_ens.pth model.roi_head.fixed_lambda=0.6
./tools/dist_test.sh  configs/transfer/transfer_coco.py workdirs/vild_ens_2x_neg5_ens/epoch_24.pth 8 --eval bbox --cfg-options model.roi_head.load_feature=False model.roi_head.prompt_path=checkpoints/coco/fg_bg_6_10_voc_neg30_ens.pth model.roi_head.fixed_lambda=0.6
./tools/dist_test.sh  configs/transfer/transfer_objects365.py workdirs/vild_ens_2x_neg5_ens/epoch_20.pth 8 --eval bbox --cfg-options model.roi_head.load_feature=False model.roi_head.prompt_path=checkpoints/obj365/fg_bg_6_10_obj365_neg30_ens.pth model.roi_head.fixed_lambda=0.6