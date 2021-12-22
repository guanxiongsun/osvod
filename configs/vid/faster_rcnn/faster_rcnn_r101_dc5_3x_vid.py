_base_ = [
    ".faster_rcnn_r50_dc5_3x_vid",
]

model = dict(
    detector=dict(
        backbone=dict(
            depth=101,
            init_cfg=dict(
                type='Pretrained', checkpoint='torchvision://resnet101'))))
