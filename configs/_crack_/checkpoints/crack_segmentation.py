norm_cfg = dict(type='SyncBN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=512,
        embed_dims=128,
        patch_size=4,
        window_size=12,
        mlp_ratio=4,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        strides=(4, 2, 2, 2),
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        use_abs_pos_embed=False,
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='LN', requires_grad=True),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'C://Users//hyunseok.lee//Documents//workspace//mmsegmentation//work_dirs//_upernet_swin_base_patch4_window12_768_noresize//iter_160000.pth'
        )),
    decode_head=dict(
        type='UPerHead',
        in_channels=[128, 256, 512, 1024],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=[
            dict(
                type='TverskyLoss',
                loss_name='loss_tversky',
                alpha=0.3,
                beta=0.7,
                smooth=1,
                class_weight=[0.01, 1],
                loss_weight=1.0)
        ],
        sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=512,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=[
            dict(
                type='TverskyLoss',
                loss_name='loss_tversky',
                alpha=0.2,
                beta=0.8,
                smooth=1,
                class_weight=[0.01, 1],
                loss_weight=0.3)
        ],
        sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
dataset_type = 'CrackDataset'
data_root = 'C://Users\hyunseok.lee\Documents\workspace\cracks\dataset2'
vid_data_root = 'C://Users//hyunseok.lee//Documents//workspace//cracks//드론촬영사진'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (1080, 1080)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=(3000, 4000), ratio_range=(1.4, 1.6)),
    dict(
        type='RandomCrop',
        crop_size=(1080, 1080),
        cat_max_ratio=0.75,
        ignore_index=0),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomRotate', prob=0.5, degree=90),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(1080, 1080), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
train_pipeline2 = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(
        type='RandomCrop',
        crop_size=(1080, 1080),
        cat_max_ratio=0.75,
        ignore_index=0),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomRotate', prob=0.5, degree=90),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(1080, 1080), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(3000, 4000),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=[
        dict(
            type='CrackDataset',
            data_root=
            'C://Users//hyunseok.lee//Documents//workspace//cracks//드론촬영사진',
            img_dir='images',
            ann_dir='annotations\mask6',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', reduce_zero_label=False),
                dict(
                    type='RandomCrop',
                    crop_size=(1080, 1080),
                    cat_max_ratio=0.75,
                    ignore_index=0),
                dict(type='RandomFlip', prob=0.5),
                dict(type='RandomRotate', prob=0.5, degree=90),
                dict(type='PhotoMetricDistortion'),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(
                    type='Pad', size=(1080, 1080), pad_val=0, seg_pad_val=255),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_semantic_seg'])
            ])
    ],
    val=dict(
        type='CrackDataset',
        data_root='C://Users\hyunseok.lee\Documents\workspace\cracks\dataset2',
        img_dir='images/test',
        ann_dir='annotations/test66',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(3000, 4000),
                flip=False,
                transforms=[
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='CrackDataset',
        data_root='C://Users\hyunseok.lee\Documents\workspace\cracks\dataset2',
        img_dir='images/test',
        ann_dir='annotations/test66',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(3000, 4000),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'C://Users//hyunseok.lee//Documents//workspace//mmsegmentation//work_dirs//_upernet_swin_base_patch4_window12_768_noresize//iter_160000.pth'
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(
    type='AdamW',
    lr=6e-06,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
optimizer_config = dict()
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-07,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=20000)
checkpoint_config = dict(by_epoch=False, interval=1900)
evaluation = dict(
    interval=1900, metric=['mIoU', 'mFscore', 'mDice'], pre_eval=True)
checkpoint_file = 'C://Users//hyunseok.lee//Documents//workspace//mmsegmentation//work_dirs//_upernet_swin_base_patch4_window12_768_noresize//iter_160000.pth'
work_dir = './work_dirs\_upernet_swin_base_patch4_window12_768_noresize_3rd_videos_t_loss'
gpu_ids = [0]
auto_resume = False
