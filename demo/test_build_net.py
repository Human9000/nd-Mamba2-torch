import torch

from torchnssd import (
    BiMamba2_1D,
    BiMamba2,
    BiMamba2_2D,
    BiMamba2_3D,
    Mamba2,
    Backbone_VMAMBA2,
    VMAMBA2Block,
    BiMamba2Ac2d,
    ClsVSSD,
    SegVSSD,
    export_jit_script,
    export_onnx,
    statistics,
    test_run,
)


def test_bimamba2_1d():
    # 通用的多维度双向mamba2
    net = BiMamba2_1D(61, 128, 32).cuda()
    net.eval()
    x = torch.randn(1, 61, 63).cuda()
    export_jit_script(net, root='../export_weights')
    export_onnx(net, x, root='../export_weights')
    test_run(net, x)
    statistics(net, (61, 63))


def test_bimamba2():
    # 通用的多维度双向mamba2
    net = BiMamba2(61, 128, 32).cuda()
    net.eval()
    x1 = torch.randn(1, 61, 63).cuda()
    x2 = torch.randn(1, 61, 63, 17).cuda()
    x3 = torch.randn(1, 61, 63, 14, 28).cuda()
    export_jit_script(net, root='../export_weights')
    export_onnx(net, x1, root='../export_weights')
    statistics(net, (61, 63))
    test_run(net, x1)
    test_run(net, x2)
    test_run(net, x3)


def test_bimamba2_2d():
    net = BiMamba2_2D(61, 128, 64).cuda()
    net.eval()
    x = torch.randn(1, 61, 63, 17).cuda()
    export_jit_script(net, root='../export_weights')
    export_onnx(net, x, root='../export_weights')
    statistics(net, (61, 63, 17))
    test_run(net, x)


def test_bimamba2_3d():
    net = BiMamba2_3D(61, 128, 64).cuda()
    net.eval()
    x = torch.randn(1, 61, 63, 17, 18).cuda()
    export_jit_script(net, root='../export_weights')
    export_onnx(net, x, root='../export_weights')
    statistics(net, (61, 63, 17, 18))
    test_run(net, x)


def test_mamba2():
    net = Mamba2(d_model=64).cuda()
    net.eval()
    x = torch.randn(1, 256, 64).cuda()
    export_jit_script(net, root='../export_weights')
    export_onnx(net, x, root='../export_weights')
    statistics(net, (256, 64))
    test_run(net, x)


def test_backbone_vmamba2():
    net = Backbone_VMAMBA2(linear_attn_duality=False, ssd_chunk_size=32).cuda()
    net.eval()
    x = torch.randn(1, 3, 512, 512).cuda()
    export_jit_script(net, root='../export_weights')
    export_onnx(net, x, root='../export_weights')

    def this_test_run(net, x):
        ys = net(x)
        for i, y in enumerate(ys):
            print(i, y.shape)

    this_test_run(net, x)
    statistics(net, (3, 512, 512))


def test_vmamamba2block():
    net = VMAMBA2Block(
        dim=64,  # 通道数
        input_resolution=(15, 15),  # 特征图尺寸
        num_heads=8,
        linear_attn_duality=False,
        ssd_chunk_size=32).cuda()
    net.eval()
    x = torch.randn(1, 225, 64).cuda()  # batch, 特征图尺寸, 通道数
    export_jit_script(net, root='../export_weights')
    export_onnx(net, x, root='../export_weights')

    statistics(net, (225, 64))  # 特征图尺寸, 通道数
    test_run(net, x)


def test_bimamba2ac2d():
    net = BiMamba2Ac2d(61, 128, 32).cuda()
    net.eval()
    x = torch.randn(1, 61, 63, 63).cuda()
    export_jit_script(net, root='../export_weights')
    export_onnx(net, x, root='../export_weights')
    test_run(net, x)
    statistics(net, (61, 63, 63))


def test_segvssd():
    # 测试
    seg = SegVSSD(fact=(2, 2), in_chans=3, out_channel=2).cuda()
    seg.eval()
    x = torch.randn(1, 3, 256, 256).cuda()
    export_jit_script(seg, root='../export_weights')
    export_onnx(seg, x, root='../export_weights')
    test_run(seg, x)
    statistics(seg, tuple(x.shape[1:]))


def test_clsvssd():
    cls = ClsVSSD(fact=(2, 2), in_chans=3, out_channel=1024).cuda()
    cls.eval()
    x = torch.randn(1, 3, 256, 256).cuda()
    # 测试
    export_jit_script(cls, root='../export_weights')
    export_onnx(cls, x, root='../export_weights')
    test_run(cls, x)
    statistics(cls, tuple(x.shape[1:]))


if __name__ == '__main__':
    # test_bimamba2()
    # test_bimamba2_1d()
    # test_bimamba2_2d()
    # test_bimamba2_3d()
    # test_mamba2()
    # test_backbone_vmamba2()
    # test_vmamamba2block()
    # test_bimamba2ac2d()
    test_clsvssd()
    test_segvssd()
