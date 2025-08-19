# Step1
import paddle
import numpy as np
import matplotlib.pyplot as plt

from Global_Parameter import *
from tools import logger


class EPS_round:
    def __init__(self, valiloader):
        """
        初始化函数
        Args:
            valiloader (paddle.io.DataLoader): PaddlePaddle 的数据加载器
        """
        self.valiloader = valiloader
        self.delta_S = 1
        self.mum_S = 0
        self.best_count = 0.0
        self.factor = 0.5
        self.lr_start = False

    def RoundlyAccount(self, old_global_model, eps_global, t, device, Epoch):
        """
        计算得该轮的隐私预算
        Args:
            old_global_model (paddle.nn.Layer): PaddlePaddle 模型
            eps_global (float): 全局隐私预算
            t (int): 当前轮次
            device (str): 运行设备, e.g., 'cpu', 'gpu:0'
            Epoch (int): 总训练轮数 (原代码传入但未使用)
        Returns:
            float: 当前轮次的隐私预算
        """
        # PaddlePaddle 的独特函数：paddle.no_grad()
        # 与 torch.no_grad() 功能相同，在该上下文管理器中，所有计算都不会被记录梯度
        with paddle.no_grad():
            old_global_model.eval()  # 切换到评估模式，用法与 PyTorch 相同
            total = 0
            correct = 0
            for data in self.valiloader:
                images, labels = data

                # 将数据移动到指定设备
                # 注意：在 PaddlePaddle 中，更推荐在程序开头使用 paddle.set_device(device)
                # 来设置全局设备，但 .to() 的写法也是支持的。
                images = images.to(device)
                labels = labels.to(device)

                # 模型前向传播
                outputs = old_global_model(images)

                # PaddlePaddle 的独特函数：paddle.max()
                # 使用 axis=1 指定在第一个维度上计算最大值，获取预测的类别索引
                predictions = paddle.argmax(outputs, axis=1)
                # _, predictions = paddle.max(outputs, axis=1)

                # PaddlePaddle 的独特函数：tensor.shape
                # 使用 tensor.shape[0] 获取批量大小
                total += labels.shape[0]

                # (predictions == labels) 创建一个布尔类型的 Tensor
                # .sum() 计算 True 的数量（即正确的预测数）
                # .item() 将结果从单元素 Tensor 转换为 Python 数字，用法与 PyTorch 相同
                correct += (predictions == labels).sum().item()

        daughter_S = correct / total

        # 对 delta_S 的处理逻辑 (这部分是纯 Python 和 Numpy，无需改动)
        if t > 2:
            if daughter_S - self.mum_S > 0.07:
                self.lr_start = True

            if not self.lr_start:
                if daughter_S <= self.best_count:
                    self.factor = self.factor * 0.4
                else:
                    self.factor = self.factor * 0.6

            if daughter_S > self.best_count:
                self.best_count = daughter_S

            if not self.lr_start:
                delta_S = self.factor
            else:
                delta_S = min(daughter_S - self.mum_S, self.factor)

            if 0.01 < delta_S < self.delta_S:
                self.delta_S = delta_S
            elif 0 <= delta_S < 0.01:
                self.delta_S = 0

        eps_round = np.exp(-1.0 * self.delta_S) * eps_global / (rounds - t + 1)

        self.mum_S = daughter_S

        return eps_round
