_base_ = 'coco_detection.py'
dataset_type = 'Objects365Dataset'
data_root = 'data/objects365/'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='ClassBalancedDataset',
        oversample_thr=1e-3,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/objects365_train.json',  # _Tiny
            img_prefix=data_root + 'train/')),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/objects365_val.json',
        img_prefix=data_root + 'val/'),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/objects365_val.json',
        img_prefix=data_root + 'val/'))
evaluation = dict(metric='bbox')
# train=dict(
#     type=dataset_type,
#     ann_file=data_root + 'annotations/objects365_train.json',#_Tiny
#     img_prefix=data_root + '/train/'),
# train=dict(
#     type='ClassBalancedDataset',
#     oversample_thr=1e-3,
#     dataset=dict(
#         type=dataset_type,
#         ann_file=data_root + 'annotations/objects365_Tiny_train.json',
#         img_prefix=data_root + '/train/')),
# train = dict(
#     type=dataset_type,
#     ann_file=data_root + 'annotations/instances_train2017.json',
#     img_prefix=data_root + '/train2017/'),