_base_ = ['./mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1_pretrain.py']
lr_config = dict(step=[8,10,11])
total_epochs = 12
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.000025)
evaluation = dict(interval=2,metric=['bbox', 'segm'])
checkpoint_config = dict(interval=1, create_symlink=False)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2)

model = dict(
    roi_head=dict(
        type='StandardRoIHeadTEXTPrompt',
        mask_head=dict(class_agnostic=True)))
load_from = 'data/current_mmdetection_Head.pth'
# fp16 = dict(loss_scale=512.)
