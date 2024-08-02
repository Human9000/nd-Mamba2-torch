# BiMamba2 for any dimension
只使用pytorch实现的双向Mamba2（BiMamba2）提供N维度支持，包括1d，2d，3d数据的支持，借助BiMamba2你可以很方便的缝合到任意模型中去提高精度。

## 特点
- ✅ 1d数据(batch,channel,length)
- ✅ 2d数据(batch,channel,height,width)
- ✅ 3d数据(batch,channel,deep,height,width)
- ✅ Nd数据(batch,channel,*size)
- ✅ 支持任意版本的CUDA不需要编译
  
## 提示
*如果你想要更快的速度，可以将本项目中的Mamba2替换为Mamba2官方的Cuda加速实现，并按照官方要求安装各种依赖包，这不会影响本项目对多维度数据的支持*
 
  
## 使用样例
### 代码
```python


if __name__ == '__main__':
    # 通用的多维度双向mamba2
    net_n = BiMamba2(64, 128, 64).cuda()

    # 定制的双向mamba2 1d, 2d, 3d
    net1 = BiMamba2_1d(64, 128, 64).cuda()
    net2 = BiMamba2_2d(64, 128, 64).cuda()
    net3 = BiMamba2_3d(64, 128, 64).cuda()

    # 多维度数据
    x1 = torch.randn(1, 64, 32).cuda() # 1d
    x2 = torch.randn(1, 64, 32, 77).cuda() # 2d
    x3 = torch.randn(1, 64, 32, 77, 25).cuda() # 3d
    x4 = torch.randn(1, 64, 32, 77, 25, 15).cuda() # 4d

    # 测试
    y1 = net_n(x1)
    print(y1.shape)
    y2 = net_n(x2)
    print(y2.shape)
    y3 = net_n(x3)
    print(y3.shape)
    y4 = net_n(x4)
    print(y4.shape)


    y1 = net1(x1)
    print(y1.shape)
    y2 = net2(x2)
    print(y2.shape)
    y3 = net3(x3)
    print(y3.shape)
```
### 效果
``` base
torch.Size([1, 128, 32])
torch.Size([1, 128, 32, 77])
torch.Size([1, 128, 32, 77, 25])
torch.Size([1, 128, 32, 77, 25, 15])
torch.Size([1, 128, 32])
torch.Size([1, 128, 32, 77])
torch.Size([1, 128, 32, 77, 25])
```
## 引用
 ```bibtex
[1] Mamba2
@inproceedings{mamba2,
  title={Transformers are {SSM}s: Generalized Models and Efficient Algorithms Through Structured State Space Duality},
  author={Dao, Tri and Gu, Albert},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2024}
}
 ```
