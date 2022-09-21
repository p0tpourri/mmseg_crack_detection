# dataset settings
dataset_type = 'CrackDataset'
# data_root = 'C://Users\\hyunseok.lee\\Documents\\workspace\\cracks\\dataset2'
data_root = '../../../../../workspace/cracks/dataset2'
# vid_data_root = 'C://Users//hyunseok.lee//Documents//workspace//cracks//드론촬영사진'
vid_data_root = '../../../../../workspace/cracks/드론촬영사진'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (1080, 1080)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False), # background class 추가시 False
    dict(type='Resize', img_scale=(3000, 4000), ratio_range=(1.4, 1.6)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75, ignore_index=0), # background crop 안되게 추가
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomRotate', prob=0.5, degree=90),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

train_pipeline2 = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False), 
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75, ignore_index=0), 
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomRotate', prob=0.5, degree=90),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),]

test_pipeline = [
    
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2160, 3840),
        # img_ratios=[0.75, 1.0, 1.25],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=[
      ## 1. still image dataset ##
        # dict(
        # type=dataset_type,
        # data_root=data_root,
        # img_dir='images\\train',
        # ann_dir='annotations\\train_w36',
        # # split='dataset2\\splits\\train.txt',
        # pipeline=train_pipeline),

      ## 2. video image dataset ##
        dict(
            type=dataset_type,
            data_root=vid_data_root,
            img_dir='images',
            ann_dir='annotations\\mask6',
            pipeline=train_pipeline2)
        ],

    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/test',
        ann_dir='annotations/test66',
        pipeline=test_pipeline),

    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/test',
        ann_dir='annotations/test66',
        pipeline=test_pipeline))