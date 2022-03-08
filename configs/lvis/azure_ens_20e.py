_base_ = 'mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1_pretrain_ens.py'
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.000025)
lr_config = dict(step=[16,19])
total_epochs = 20
evaluation = dict(interval=2,metric=['bbox', 'segm'])
resume_from = "workdirs/vild_ens_new_1x/epoch_8.pth"
checkpoint_config = dict(interval=2, create_symlink=False)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2
    )