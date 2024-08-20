from ex_bi_mamba2 import torch, nn, BiMamba2_1D


# 双向非对称mamba2 块
class BiMamba2Ac2d(nn.Module):
    def __init__(self, cin, cout, d_model, **mamba2_args):
        super().__init__()
        self.bi_mamba_h1 = BiMamba2_1D(cout, cout, d_model, **mamba2_args)
        self.bi_mamba_w1 = BiMamba2_1D(cin, cout, d_model, **mamba2_args)

        self.bi_mamba_h2 = BiMamba2_1D(cin, cout, d_model, **mamba2_args)
        self.bi_mamba_w2 = BiMamba2_1D(cout, cout, d_model, **mamba2_args)

        self.fc = nn.Conv2d(cout * 2, cout, 1)

    def forward(self, x):
        # 非对称卷积
        b, c, h, w = x.shape

        # 先 w 再 h
        x_w1 = x.transpose(1, 2).reshape(b * h, c, w)  # bh, c, w
        y_w1 = self.bi_mamba_w1(x_w1)
        y_w1 = y_w1.reshape(b, h, -1, w).transpose(1, 2)
        x_h1 = y_w1.transpose(1, 3).reshape(b * w, h, -1).transpose(1, 2)  # bw, c, h
        y1 = self.bi_mamba_h1(x_h1).transpose(1, 2).reshape(b, w, h, -1).transpose(1, 3)

        # 先 h 再 w
        x_h2 = x.transpose(1, 3).reshape(b * w, h, c).transpose(1, 2)  # bw, c, h
        y_h2 = self.bi_mamba_w1(x_h2).transpose(1, 2).reshape(b, w, h, -1).transpose(1, 3)
        x_w2 = y_h2.transpose(1, 2).reshape(b * h, -1, w)  # bh, c, w
        y2 = self.bi_mamba_h1(x_w2).reshape(b, h, -1, w).transpose(1, 2)

        # 合并结果
        y = torch.cat([y1, y2], dim=1)
        y = self.fc(y)
        return y


def test_export_jit_script(net, x):
    net_script = torch.jit.script(net)
    torch.jit.save(net_script, 'net.jit.script')
    net2 = torch.jit.load('net.jit.script')
    y = net2(x)
    print(y.shape)


def test_export_onnx(net, x):
    torch.onnx.export(net,
                      x,
                      "net.onnx",  # 输出的 ONNX 文件名
                      export_params=True,  # 存储训练参数
                      opset_version=14,  # 指定 ONNX 操作集版本
                      do_constant_folding=False,  # 是否执行常量折叠优化
                      input_names=['input'],  # 输入张量的名称
                      output_names=['output'],  # 输出张量的名称
                      dynamic_axes={'input': {0: 'batch_size'},  # 可变维度的字典
                                    'output': {0: 'batch_size'}})


if __name__ == '__main__':
    net = BiMamba2Ac2d(61, 128, 32).cuda()
    net.eval()

    x = torch.randn(1, 61, 63, 63).cuda()
    test_export_jit_script(net, x)
    test_export_onnx(net, x)
