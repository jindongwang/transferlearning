### Python版的迁移学习代码

- - -

#### TCA(Transfer component analysis)

算法出处：Pan S J, Tsang I W, Kwok J T, et al. Domain adaptation via transfer component analysis[J]. IEEE Transactions on Neural Networks, 2011, 22(2): 199-210.

用法：

import da_tool.tca

```python
my_tca = da_tool.tca.TCA(dim=30,kerneltype='rbf', kernelparam=1, mu=1)
x_src_tca, x_tar_tca, x_tar_o_tca = my_tca.fit_transform(x_src, x_tar)
```

测试请见test.py文件。

- - -

### 未来增加的算法：

#### GFK (Geodesic flow kernel)
等等。