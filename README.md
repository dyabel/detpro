# Prepare data
Download dataset according to [LVIS](https://www.lvisdataset.org/) [VOC](http://host.robots.ox.ac.uk/pascal/VOC/) [COCO](https://cocodataset.org/#home) [Objects365](https://www.objects365.org/overview.html)
# Installation
install mmdet2.7
```shell
pip install -r requirements/build.txt
pip install -e .
pip install instaboostfast
pip install git+https://github.com/openai/CLIP.git
pip uninstall pycocotools -y
pip uninstall mmpycocotools -y
pip install mmpycocotools
pip install git+https://github.com/lvis-dataset/lvis-api.git
pip install mmcv-full==1.2.5 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html


```
# Prepare proposal embedding, label, iou for prompt training
```shell
./tools/dist_train.sh  configs/lvis/prompt_save_train.py 8 --work-dir workdirs/prompt_save_train
./tools/dist_train.sh  configs/lvis/prompt_save_train.py 8 --work-dir workdirs/prompt_save_train
```
# Train prompt
```shell
cd prompt
python run.py train data/lvis_clip_image_proposal_embedding/train data/lvis_clip_image_proposal_embedding/val checkpoints/exp4 fg_bg_5_5_6_end soft 0.5 0.5 0.6 8 end
python run.py train data/lvis_clip_image_proposal_embedding/train data/lvis_clip_image_proposal_embedding/val checkpoints/exp4 fg_bg_5_6_7_end soft 0.5 0.6 0.7 8 end
python run.py train data/lvis_clip_image_proposal_embedding/train data/lvis_clip_image_proposal_embedding/val checkpoints/exp4 fg_bg_5_7_8_end soft 0.5 0.7 0.8 8 end
python run.py train data/lvis_clip_image_proposal_embedding/train data/lvis_clip_image_proposal_embedding/val checkpoints/exp4 fg_bg_5_8_9_end soft 0.5 0.8 0.9 8 end
python run.py train data/lvis_clip_image_proposal_embedding/train data/lvis_clip_image_proposal_embedding/val checkpoints/exp4 fg_bg_5_9_10_end soft 0.5 0.9 1.1 8 end
python run.py test data/lvis_clip_image_proposal_embedding/train data/lvis_clip_image_proposal_embedding/val checkpoints/exp4 fg_bg_5_10_end soft 0.0 0.5 0.5 checkpoints/exp4/fg_bg_5_5_6_endepoch6.pth checkpoints/exp4/fg_bg_5_6_7_endepoch6.pth checkpoints/exp4/fg_bg_5_7_8_endepoch6.pth checkpoints/exp4/fg_bg_5_8_9_endepoch6.pth checkpoints/exp4/fg_bg_5_9_10_endepoch6.pth
./tools/dist_train.sh  configs/lvis/azure_ens.py 8 --work-dir workdirs/vild_ens_1x_fg_bg_5_10_end --cfg-options model.roi_head.prompt_path=checkpoints/exp4/fg_bg_5_10_end_ens.pth

```
# Train OVOD detector
```
#save clip image embedding
./tools/dist_train.sh  configs/lvis/azure_ens_2x.py 8 --work-dir workdirs/vild_ens_2x_fg_bg_5_10_end --cfg-options model.roi_head.prompt_path=checkpoints/exp4/fg_bg_5_10_end_ens.pth model.roi_head.load_feature=False totol_epochs=1
#zip clip image embedding
zip -r lvis_clip_image_embedding.zip data/lvis_clip_image_embedding/*
./tools/dist_train.sh  configs/lvis/azure_ens_2x.py 8 --work-dir workdirs/vild_ens_2x_fg_bg_5_10_end --cfg-options model.roi_head.prompt_path=checkpoints/exp4/fg_bg_5_10_end_ens.pth model.roi_head.load_feature=True
```
# Generate class embedding for tranfer Datasets(take objects365 as example)
```
python gen_cls_embedding.py checkpoints/exp3/fg_bg_5_5_6_endepoch6_prompt.pth checkpoints/obj365/fg_bg_5_5_6_obj365 obj365
python gen_cls_embedding.py checkpoints/exp3/fg_bg_5_6_7_endepoch6_prompt.pth checkpoints/obj365/fg_bg_5_5_6_obj365 obj365
python gen_cls_embedding.py checkpoints/exp3/fg_bg_5_7_8_endepoch6_prompt.pth checkpoints/obj365/fg_bg_5_5_6_obj365 obj365
python gen_cls_embedding.py checkpoints/exp3/fg_bg_5_8_9_endepoch6_prompt.pth checkpoints/obj365/fg_bg_5_5_6_obj365 obj365
python gen_cls_embedding.py checkpoints/exp3/fg_bg_5_9_10_endepoch6_prompt.pth checkpoints/obj365/fg_bg_5_5_6_obj365 obj365
python run.py test data/lvis_clip_image_proposal_embedding/train data/lvis_clip_image_proposal_embedding/val checkpoints/obj365 fg_bg_5_10_obj365 soft 0.0 0.5 0.5 checkpoints/obj365/fg_bg_5_5_6_obj365.pth checkpoints/obj365/fg_bg_5_6_7_obj365.pth checkpoints/obj365/fg_bg_5_7_8_obj365.pth checkpoints/obj365/fg_bg_5_8_9_obj365.pth checkpoints/obj365/fg_bg_5_9_10_obj365.pth
```
# Transfer to other datasets
```
 ./tools/dist_test.sh  configs/transfer/transfer_voc.py workdirs/vild_ens_2x_neg5_ens/epoch_24.pth 8 --eval mAP --cfg-options model.roi_head.load_feature=False model.roi_head.prompt_path=checkpoints/voc/fg_bg_6_10_voc_neg30_ens.pth model.roi_head.fixed_lambda=0.6
./tools/dist_test.sh  configs/transfer/transfer_coco.py workdirs/vild_ens_2x_neg5_ens/epoch_24.pth 8 --eval bbox --cfg-options model.roi_head.load_feature=False model.roi_head.prompt_path=checkpoints/coco/fg_bg_6_10_voc_neg30_ens.pth model.roi_head.fixed_lambda=0.6
./tools/dist_test.sh  configs/transfer/transfer_objects365.py workdirs/vild_ens_2x_neg5_ens/epoch_20.pth 8 --eval bbox --cfg-options model.roi_head.load_feature=False model.roi_head.prompt_path=checkpoints/obj365/fg_bg_6_10_obj365_neg30_ens.pth model.roi_head.fixed_lambda=0.6
```
