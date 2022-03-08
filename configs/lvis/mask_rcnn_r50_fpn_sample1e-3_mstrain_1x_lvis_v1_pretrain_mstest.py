_base_ = [
    'mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1_pretrain.py'
]
# test_cfg_aug = dict(
#     rpn=dict(
#         nms_across_levels=False,
#         nms_pre=1000,
#         nms_post=1000,
#         max_num=1000,
#         nms_thr=0.7,
#         min_bbox_size=0),
#     rcnn=dict(
#         score_thr=0.0001,
#         nms=dict(type='nms', iou_threshold=0.5),
#         max_per_img=300,
#         mask_thr_binary=0.5))
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadProposals', num_max_proposals=None),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(1333, 600), (1333, 640), (1333,700), (1333,800), (3000,900), (3000,1000), (3000,1100)],
        # img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                #    (1333, 768), (1333, 800)],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img','img_no_normalize']),
            dict(type='Collect', keys=['img','img_no_normalize'])
        ])
]
checkpoint_config = dict(interval=1,create_symlink=False)
load_from = 'current_mmdetection_Head.pth'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    test=dict(pipeline=test_pipeline))
# fp16 = dict(loss_scale=512.)