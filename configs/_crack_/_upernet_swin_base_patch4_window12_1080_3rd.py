_base_ = [
    '../_base_/models/upernet_swin.py', '../_base_/datasets/road_crack_dataset.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'

]

# checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_base_patch4_window12_384_22k_20220317-e5c09f74.pth'  # noqa
# checkpoint_file = '../checkpoints/pretrained/pretrained.pth'

model = dict(
    backbone=dict(
        # init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        pretrain_img_size=512,
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=12,
        use_abs_pos_embed=False,
        drop_path_rate=0.3,
        patch_norm=True),

    decode_head=dict(in_channels=[128, 256, 512, 1024], num_classes=2,
                    sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000),
                    loss_decode=[
                                # dict(type='CrossEntropyLoss', loss_name='loss_ce',class_weight=[1, 200], loss_weight=1.0),
                                dict(type='TverskyLoss', loss_name='loss_tversky', 
                                alpha=0.3, beta=0.7, smooth=1, class_weight=[0.01, 0.8], loss_weight=1.0,),
                                ]),

    auxiliary_head=dict(in_channels=512, num_classes=2,
                        sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000),
                    loss_decode=[
                        # dict(type='CrossEntropyLoss', loss_name='loss_ce', class_weight=[1, 200], loss_weight=1.0),
                        dict(type='TverskyLoss', loss_name='loss_tversky', 
                                alpha=0.2, beta=0.8, smooth=1, class_weight=[0.01, 0.8], loss_weight=0.3,)
                                ]))

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.000006, 
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-7,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

evaluation = dict(metric=['mIoU', 'mFscore', 'mDice'])

load_from = '../checkpoints/pretrained/pretrained.pth'
