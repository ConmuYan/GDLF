import torch
import argparse
import torch
from torch import nn
from models.style_transformer import StyleTransformer

# 创建一个简单的参数类
class TestOptions:
    def __init__(self):
        # 基本配置
        self.output_size = 256  # 使用更小的输出尺寸以节省显存
        self.checkpoint_path = None
        self.stylegan_weights = None
        self.learn_in_w = False
        self.start_from_latent_avg = False  # 不使用预训练的平均潜在向量
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 添加 encoder 所需的参数
        self.input_nc = 3  # 输入图像的通道数
        self.n_styles = 18  # 样式向量的数量
        self.coarse_size = 256  # 使用更小的尺寸

def main():
    # 初始化参数
    opts = TestOptions()
    
    # 创建模型
    print("正在初始化模型...")
    model = StyleTransformer(opts).to(opts.device)
    
    # 使用随机初始化的权重
    def weights_init(m):
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
    
    # 应用随机初始化
    model.apply(weights_init)
    model.eval()
    
    # 打印模型结构
    print("模型初始化完成，模型结构：")
    print(model)
    
    # 创建一个测试输入张量
    batch_size = 1
    img_size = 256  # 使用256x256的输入尺寸
    test_input = torch.randn(batch_size, 3, img_size, img_size).to(opts.device)
    print(f"测试输入张量形状: {test_input.shape}")
    
    # 前向传播
    print("\n开始前向传播...")
    try:
        with torch.no_grad():
            output = model(test_input)
            print(f"输出张量形状: {output.shape if isinstance(output, torch.Tensor) else [o.shape for o in output]}")
            print("\n前向传播成功完成!")
    except Exception as e:
        print(f"\n前向传播过程中出现错误: {str(e)}")

if __name__ == "__main__":
    main()
