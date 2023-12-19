auto_scale_lr = dict(base_batch_size=64, enable=False)
data_root = 'data/coco/'
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(interval=1, type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
file_client_args = dict(backend='disk')
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
model = dict(
    backbone=dict(
        depth=50,
        frozen_stages=1,
        init_cfg=dict(checkpoint='open-mmlab://detectron/resnet50_caffe', type='Pretrained'),
        norm_cfg=dict(requires_grad=False, type='BN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(3,),
        style='caffe',
        type='ResNet'),
    bbox_head=dict(
        anchor_generator=dict(
            ratios=[
                1.0,
            ], scales=[
                1,
                2,
                4,
                8,
                16,
            ], strides=[
                32,
            ], type='AnchorGenerator'),
        bbox_coder=dict(
            add_ctr_clamp=True,
            ctr_clamp=32,
            target_means=[
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            target_stds=[
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            type='DeltaXYWHBBoxCoder'),
        in_channels=512,
        loss_bbox=dict(loss_weight=1.0, type='GIoULoss'),
        loss_cls=dict(alpha=0.25, gamma=2.0, loss_weight=1.0, type='FocalLoss', use_sigmoid=True),
        num_classes=80,
        reg_decoded_bbox=True,
        type='YOLOFHead'),
    data_preprocessor=dict(
        bgr_to_rgb=False,
        mean=[
            103.53,
            116.28,
            123.675,
        ],
        pad_size_divisor=32,
        std=[
            1.0,
            1.0,
            1.0,
        ],
        type='DetDataPreprocessor'),
    neck=dict(
        block_dilations=[
            2,
            4,
            6,
            8,
        ],
        block_mid_channels=128,
        in_channels=2048,
        num_residual_blocks=4,
        out_channels=512,
        type='DilatedEncoder'),
    test_cfg=dict(
        max_per_img=100,
        min_bbox_size=0,
        nms=dict(iou_threshold=0.6, type='nms'),
        nms_pre=1000,
        score_thr=0.05),
    train_cfg=dict(
        allowed_border=-1,
        assigner=dict(neg_ignore_thr=0.7, pos_ignore_thr=0.15, type='UniformAssigner'),
        debug=False,
        pos_weight=-1),
    type='YOLOF')
optim_wrapper = dict(
    optimizer=dict(lr=0.12, momentum=0.9, type='SGD', weight_decay=0.0001),
    paramwise_cfg=dict(
        custom_keys=dict(backbone=dict(lr_mult=0.3333333333333333)), norm_decay_mult=0.0),
    type='OptimWrapper')
param_scheduler = [
    dict(begin=0, by_epoch=False, end=1500, start_factor=0.00066667, type='LinearLR'),
    dict(begin=0, by_epoch=True, end=12, gamma=0.1, milestones=[
        8,
        11,
    ], type='MultiStepLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        data_root='data/coco/',
        pipeline=[
            dict(file_client_args=dict(backend='disk'), type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1333,
                800,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='data/coco/annotations/instances_val2017.json',
    format_only=False,
    metric='bbox',
    type='CocoMetric')
test_pipeline = [
    dict(file_client_args=dict(backend='disk'), type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        1333,
        800,
    ), type='Resize'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]
train_cfg = dict(max_epochs=12, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=8,
    dataset=dict(
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/'),
        data_root='data/coco/',
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=[
            dict(file_client_args=dict(backend='disk'), type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(keep_ratio=True, scale=(
                1333,
                800,
            ), type='Resize'),
            dict(prob=0.5, type='RandomFlip'),
            dict(max_shift_px=32, prob=0.5, type='RandomShift'),
            dict(type='PackDetInputs'),
        ],
        type='CocoDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(file_client_args=dict(backend='disk'), type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(keep_ratio=True, scale=(
        1333,
        800,
    ), type='Resize'),
    dict(prob=0.5, type='RandomFlip'),
    dict(max_shift_px=32, prob=0.5, type='RandomShift'),
    dict(type='PackDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        data_root='data/coco/',
        pipeline=[
            dict(file_client_args=dict(backend='disk'), type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1333,
                800,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='data/coco/annotations/instances_val2017.json',
    format_only=False,
    metric='bbox',
    type='CocoMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer', type='DetLocalVisualizer', vis_backends=[
        dict(type='LocalVisBackend'),
    ])
