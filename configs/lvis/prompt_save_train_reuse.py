_base_ = ['./mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1_pretrain.py']

checkpoint_config = dict(interval=2,create_symlink=False)
load_from = 'data/current_mmdetection_Head.pth'

total_epochs = 1

model = dict(roi_head=dict(type='StandardRoIHeadColReuse',save_feature_dir='./data/lvis_clip_image_proposal_embedding/train'))
# model = dict(roi_head=dict(save_feature_dir='data/LVIS_prompt_train/train'))
