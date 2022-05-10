_base_ = ['./mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1_pretrain.py']

checkpoint_config = dict(interval=2,create_symlink=False)
load_from = 'data/current_mmdetection_Head.pth'

total_epochs = 1

model = dict(roi_head=dict(
    type='StandardRoIHeadCol',
    save_feature_dir='./data/lvis_clip_image_proposal_embedding/val'))

data_root = 'data/lvis_v1/'
data = dict(
    train=dict(
        dataset=dict(
            type='LVISV1Dataset_ALLCLS',
            ann_file=data_root + 'annotations/lvis_v1_val.json',
            proposal_file=data_root + 'proposals/rpn_r101_fpn_lvis_val.pkl')))