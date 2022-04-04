_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.000025)
model = dict(
    pretrained='open-mmlab://detectron2/resnet50_caffe',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=True,
        style='caffe'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        num_outs=5),
    roi_head=dict(
    bbox_head=dict(
        type='Shared4Conv1FCBBoxHead',
        in_channels=256,
        ensemble=True,
        fc_out_channels=1024,
        roi_feat_size=7,
        with_cls=False,
        num_classes=80,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0., 0., 0., 0.],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        reg_class_agnostic=True,
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)
    ), 
     mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            roi_feat_size=7,
            class_agnostic=True,
            num_classes=80,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))
    ))
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=1000,
        nms_post=1000,
        max_num=1000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.01,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100,
        mask_thr_binary=0.5))
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadProposals', num_max_proposals=None),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                   (1333, 768), (1333, 800)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img','proposals', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='LoadProposals', num_max_proposals=None),
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
            dict(type='ImageToTensor', keys=['img']),
            # dict(type='ToTensor', keys=['proposals']),
            # dict(
                # type='ToDataContainer',
                # fields=[dict(key='proposals', stack=False)]),
            # dict(type='Collect', keys=['img', 'proposals']),
            dict(type='Collect', keys=['img']),
        ])
]
load_from = 'data/current_mmdetection_Head.pth'
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(dataset=dict(pipeline=train_pipeline)),
    test=dict(pipeline=test_pipeline))
# fp16 = dict(loss_scale=512.)