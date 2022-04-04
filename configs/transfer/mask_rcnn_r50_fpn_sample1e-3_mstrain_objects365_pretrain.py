_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/obj365_detection.py',
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
        ensemble=False,
        fc_out_channels=1024,
        roi_feat_size=7,
        with_cls=False,
        num_classes=365,
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
            num_classes=365,
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
        score_thr=0.003,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=300,
        mask_thr_binary=0.5))
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
load_from = 'data/current_mmdetection_Head.pth'
# data = dict(train=dict(dataset=dict(pipeline=train_pipeline)),
            # )
# fp16 = dict(loss_scale=512.)