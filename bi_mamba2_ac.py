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
        y_w1 = self.bi_mamba_w1(x_w1).reshape(b, h, c, w).transpose(1, 2)
        x_h1 = y_w1.transpose(1, 3).reshape(b * w, h, c).transpose(1, 2)  # bw, c, h
        y1 = self.bi_mamba_h1(x_h1).transpose(1, 2).reshape(b, w, h, c).transpose(1, 3)

        # 先 h 再 w
        x_h2 = x.transpose(1, 3).reshape(b * w, h, c).transpose(1, 2)  # bw, c, h
        y_h2 = self.bi_mamba_w1(x_h2).transpose(1, 2).reshape(b, w, h, c).transpose(1, 3)
        x_w2 = y_h2.transpose(1, 2).reshape(b * h, c, w)  # bh, c, w
        y2 = self.bi_mamba_h1(x_w2).reshape(b, h, c, w).transpose(1, 2)

        # 合并结果
        y = torch.cat([y1, y2], dim=1)
        y = self.fc(y)
        return y
