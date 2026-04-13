class Config:
    # model
    num_classes = 21
    backbone = "resnet50"
    in_channels = 2048
    inter_channels = 64
    rcca_r = 2

    # data
    input_size = (512, 512)
    output_stride = 8
