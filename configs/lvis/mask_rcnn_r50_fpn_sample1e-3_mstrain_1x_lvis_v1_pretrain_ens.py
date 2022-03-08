_base_ = 'mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1_pretrain.py'
model = dict(roi_head=dict(bbox_head=dict(ensemble=True)))
evaluation = dict(interval=1,metric=['bbox', 'segm'])