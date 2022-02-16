## 1. 必要的代码级样例展示

```python
import timm
class Encoder(nn.Module):
    B = 4
    def __init__(self, feedback_bits):
        super(Encoder, self).__init__()
        self.conv1 = conv3x3(2, 3)
        # self.efn_layer = timm.create_model('efficientnet_b1', pretrained=True, num_classes=0)
        self.efn_layer = timm.create_model('efficientnet_b1', num_classes=0)
        self.fc = nn.Linear(1280, int(feedback_bits // self.B))
        self.sig = nn.Sigmoid()
        self.quantize = QuantizationLayer(self.B)
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.efn_layer(out)
        out = self.fc(out)
        out = self.sig(out)
        out = self.quantize(out)
        return out
```



## 2. 算法思路

1. **编码阶段：** 由于输入为126 * 128 * 2的图像，可以使用一个卷积层使得通道数增加为3，然后使用efficientnet_b1模型（去除分类层）提取特征。然后通过一个全连接层降维，另一个全连接层输出编码向量
2. **解码阶段：** 解码阶段不做修改



## 3. 亮点解读

- 合并训练集与测试集，训练1000个epoch
- batch_size设置为16
- 学习率为3e-4
- 优化方法可使用AdamW不加衰减策略，或Adam加余弦退火衰减策略





## 4. 建模算力与环境
### a. 项目运行环境
#### i. 项目所需的工具包/框架
* numpy==1.19.5
* PyTorch=1.7.1
* scipy=1.7.3

#### ii. 项目运行的资源环境
* 单卡 16G-V100

### b. 项目运行办法
#### i. 项目的文件结构
#### ii. 项目的运行步骤
1. **模型训练：** Python Model_train.py
   - 在Modelsave文件夹中生成结果文件：encoder.pth.tar、decoder.pth.tar
2. **模型验证-编码阶段：** Python Model_evaluation_encoder.py
   - 在Modelsave文件夹中生成结果文件：encoder_output.npy
3. **模型验证-解码阶段：** Python Model_evaluation_decoder.py
   - 在终端中输出模型最终得分

#### 运行结果的位置
1. Modelsave文件夹：encoder.pth.tar、decoder.pth.tar、encoder_output.npy



## 5. 使用的预训练模型相关论文及模型下载链接

论文：[EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/pdf/1905.11946.pdf)

预训练模型（efficientnet_b1）：[下载链接](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b1-533bc792.pth)

