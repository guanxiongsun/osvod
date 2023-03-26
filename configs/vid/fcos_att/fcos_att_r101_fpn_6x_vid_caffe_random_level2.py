_base_ = [
    './fcos_att_r50_fpn_3x_vid_caffe_random.py'
]

checkpoint = 'work_dirs/pretrains/fcos_r101_caffe_fpn_gn-head_mstrain_coco_backbone.pth'

model = dict(
    detector=dict(
        backbone=dict(
            depth=101,
            init_cfg=dict(
                type='Pretrained', checkpoint=checkpoint))),
    memory=dict(
        start_level=2,)
)

# learning policy
lr_config = dict(
    policy="step", warmup="linear", warmup_iters=500, warmup_ratio=1.0 / 3, step=[4]
)
# runtime settings
total_epochs = 6
checkpoint_config = dict(interval=3)
evaluation = dict(metric=["bbox"], interval=total_epochs)
runner = dict(type="EpochBasedRunner", max_epochs=total_epochs)
