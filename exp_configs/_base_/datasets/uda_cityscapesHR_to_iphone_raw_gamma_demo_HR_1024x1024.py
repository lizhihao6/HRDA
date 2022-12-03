# ---------------------------------------------------------------
# Copyright (c) 2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# dataset settings
dataset_type = 'CityscapesDataset'
img_norm_cfg = dict(
    mean=[0, 0, 0], std=[255., 255., 255.], to_rgb=True)
crop_size = (1024, 1024)
cityscapes_train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations'),
    dict(type='CombineVegetationTerrainAnno'),
    dict(type='Resize', img_scale=(2048, 1024), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),  # is applied later in dacs.py
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
iphone_raw_gamma_demo_train_pipeline = [
    dict(type='LoadRAWFromFile'),
    dict(type='Demosaic'),
    dict(type='RAWNormalize'),
    dict(type='Gamma'),
    dict(type='Resize', img_scale=(2048, 1024)),  # original 1920x1080
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),  # is applied later in dacs.py
    dict(type='Normalize', mean=[0, 0, 0], std=[1, 1, 1], to_rgb=False),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img']),
]
test_pipeline = [
    dict(type='LoadRAWFromFile'),
    dict(type='Demosaic'),
    dict(type='RAWNormalize'),
    dict(type='Gamma'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(4032//2, 3024//2),  # original 1920x1080
        # MultiScaleFlipAug is disabled by not providing img_ratios and
        # setting flip=False
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', mean=[0, 0, 0], std=[1, 1, 1], to_rgb=False),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='UDADataset',
        source=dict(
            type='CityscapesDataset',
            data_root='/shared/cityscapes/',
            classes=['road', 'building', 'fence', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'car', 'truck', 'bus', 'bicycle'],
            img_dir='leftImg8bit/train',
            ann_dir='gtFine/train',
            pipeline=cityscapes_train_pipeline),
        target=dict(
            type='IphoneDataset',
            data_root='/lzh/datasets/multiRAW/iphone_xsmax/',
            img_dir='raw',
            img_suffix='.DNG',
            ann_dir='labels/segmentation',
            split='train.txt',
            pipeline=iphone_raw_gamma_demo_train_pipeline)),
    val=dict(
        type='IphoneDataset',
        data_root='/lzh/datasets/multiRAW/iphone_xsmax/',
        img_dir='raw',
        img_suffix='.DNG',
        ann_dir='labels/segmentation',
        split='test.txt',
        pipeline=test_pipeline),
    test=dict(
        type='IphoneDataset',
        data_root='/lzh/datasets/multiRAW/iphone_xsmax/',
        img_dir='raw',
        img_suffix='.DNG',
        ann_dir='labels/segmentation',
        split='test.txt',
        pipeline=test_pipeline))
