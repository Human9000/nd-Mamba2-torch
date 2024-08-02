# BiMamba2 for any dimension
只使用pytorch实现的双向Mamba2（BiMamba2）提供N维度支持，包括1d，2d，3d数据的支持，借助BiMamba2你可以很方便的缝合到任意模型中去提高精度。

## 支持的功能
- ✅ 1d数据(batch,channel,length)
- ✅ 2d数据(batch,channel,height,width)
- ✅ 3d数据(batch,channel,deep,height,width)
- ✅ Nd数据(batch,channel,*size)


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
