_base_ = [
    "../../_base_/default_runtime.py",
    "../../_base_/schedules/schedule_1x.py",
]

img_scale = (640, 640)

# model settings
model = dict(
    type='YOLOXAtt',
    detector=dict(
        type='YOLOX',
        input_size=img_scale,
        random_size_range=(15, 25),
        random_size_interval=10,
        backbone=dict(type='CSPDarknet', deepen_factor=0.67, widen_factor=0.75),
        neck=dict(
            type='YOLOXPAFPN',
            in_channels=[192, 384, 768],
            out_channels=192,
            num_csp_blocks=2),
        bbox_head=dict(
            type='YOLOXHead', num_classes=30, in_channels=192, feat_channels=192),
        train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
        # In order to align the source code, the threshold of the val phase is
        # 0.01, and the threshold of the test phase is 0.001.
        test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65))),
    memory=dict(
        type='MPN',
        in_channels=[192, 384, 768],
        strides=[8, 16, 32],
        before_fpn=True,
        start_level=0,
        pixel_sampling_train='random',
    ),
)

# dataset settings
dataset_type = "ImagenetVIDDataset"
data_root = "data/ILSVRC/"

train_pipeline = [
    dict(type='SeqMosaic', img_scale=img_scale, pad_val=114.0),
    dict(
        type='SeqRandomAffine',
        scaling_ratio_range=(0.1, 2),
        border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    dict(
        type='SeqMixUp',
        img_scale=img_scale,
        ratio_range=(0.8, 1.6),
        pad_val=114.0),
    dict(type='SeqYOLOXHSVRandomAug'),
    dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
    # According to the official implementation, multi-scale
    # training is not considered here but in the
    # 'mmdet/models/detectors/yolox.py'.
    dict(type='SeqResize', img_scale=img_scale, keep_ratio=True),
    dict(
        type='SeqPad',
        pad_to_square=True,
        # If the image is three-channel, the pad value needs
        # to be set separately for each channel.
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='SeqFilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(
        type="VideoCollect", keys=["img", "gt_bboxes", "gt_labels", "gt_instance_ids"]
    ),
    dict(type="ConcatVideoReferences"),
    dict(type="SeqDefaultFormatBundle", ref_prefix="ref"),
]

train_dataset = dict(
    type='MultiImageMixDataset',
    dataset=[
        dict(
            type=dataset_type,
            ann_file=data_root + "annotations/imagenet_vid_train.json",
            img_prefix=data_root + "Data/VID",
            ref_img_sampler=dict(
                num_ref_imgs=2,
                frame_range=9,
                filter_key_img=True,
                method="bilateral_uniform",
            ),
            pipeline=[
                dict(type="LoadMultiImagesFromFile"),
                dict(type="SeqLoadAnnotations", with_bbox=True, with_track=True),
            ],
        ),
        dict(
            type=dataset_type,
            load_as_video=False,
            ann_file=data_root + "annotations/imagenet_det_30plus1cls.json",
            img_prefix=data_root + "Data/DET",
            ref_img_sampler=dict(
                num_ref_imgs=2,
                frame_range=0,
                filter_key_img=False,
                method="bilateral_uniform",
            ),
            pipeline=[
                dict(type="LoadMultiImagesFromFile"),
                dict(type="SeqLoadAnnotations", with_bbox=True, with_track=True),
            ],
        ),
    ],
    pipeline=train_pipeline)

test_pipeline = [
    dict(type='LoadMultiImagesFromFile', to_float32=True),
    dict(
        type='SeqMultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Pad',
                pad_to_square=True,
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            # dict(type='DefaultFormatBundle'),
            # dict(type='Collect', keys=['img'])
        ]),
    dict(
        type="VideoCollect",
        keys=["img"],
        meta_keys=("num_left_ref_imgs", "frame_stride"),
    ),
    dict(type="ConcatVideoReferences"),
    dict(type="MultiImagesToTensor", ref_prefix="ref"),
    dict(type="ToList"),
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    persistent_workers=True,
    train=train_dataset,
    val=dict(
        type=dataset_type,
        ann_file=data_root + "annotations/imagenet_vid_val.json",
        img_prefix=data_root + "Data/VID",
        ref_img_sampler=dict(
            num_ref_imgs=14,
            frame_range=[-7, 7],
            method='test_with_adaptive_stride'
        ),
        pipeline=test_pipeline,
        test_mode=True,
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + "annotations/imagenet_vid_val.json",
        img_prefix=data_root + "Data/VID",
        ref_img_sampler=dict(
            num_ref_imgs=14,
            frame_range=[-7, 7],
            method='test_with_adaptive_stride'
        ),
        pipeline=test_pipeline,
        test_mode=True,
    )
)

# optimizer
# default 8 gpu
optimizer = dict(
    type='SGD',
    lr=0.001,
    momentum=0.9,
    weight_decay=5e-4,
    nesterov=True,
    paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.))
optimizer_config = dict(grad_clip=None)

max_epochs = 90
num_last_epochs = 5
resume_from = None
interval = 30

# learning policy
lr_config = dict(
    _delete_=True,
    policy='YOLOX',
    warmup='exp',
    by_epoch=False,
    warmup_by_epoch=True,
    warmup_ratio=1,
    warmup_iters=5,  # 5 epoch
    num_last_epochs=num_last_epochs,
    min_lr_ratio=0.05)

runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)

custom_hooks = [
    dict(
        type='YOLOXModeSwitchHook',
        num_last_epochs=num_last_epochs,
        priority=48),
    dict(
        type='SyncNormHook',
        num_last_epochs=num_last_epochs,
        interval=interval,
        priority=48),
    dict(
        type='ExpMomentumEMAHook',
        resume_from=resume_from,
        momentum=0.0001,
        priority=49)
]
checkpoint_config = dict(interval=interval)
evaluation = dict(metric=["bbox"], interval=max_epochs)
