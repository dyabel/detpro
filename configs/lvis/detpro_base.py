_base_ = ['./mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1_pretrain.py']

optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.000025)
evaluation = dict(interval=4,metric=['bbox', 'segm'])
checkpoint_config = dict(interval=2, create_symlink=False)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2)

model = dict(
    roi_head=dict(
        mask_head=dict(class_agnostic=True)
        ))
load_from = 'data/current_mmdetection_Head.pth'
