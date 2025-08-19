# Step3
# 在每个CLIENT进行本地训练时初始化，用来计算CLIENT本地训练过程的隐私消耗
import matplotlib.pyplot as plt
import numpy as np
import paddle

from Global_Parameter import *
from dlgAttack import DLA
from tools import logger


# k_instance = EPS_instance(data_ind, model, Epoch, BSize, eps)
class EPS_instance:
    def __init__(self, data_ind, model, Epoch, BSize, eps, device):
        """
        初始化函数
        Args:
            data_ind (list or np.array): 数据索引
            model (paddle.nn.Layer): PaddlePaddle 模型
            Epoch (int): 训练轮数
            BSize (int): 批量大小
            eps (float): 隐私预算
            device (str): 运行设备, e.g., 'cpu', 'gpu:0'
        """
        self.ori_data = data_ind
        np.random.shuffle(self.ori_data)
        splitNum = np.ceil(len(self.ori_data) / BSize)
        self.data_ind = np.array_split(np.asarray(self.ori_data), splitNum, 0)

        self.model = model
        # PaddlePaddle 的独特函数：paddle.optimizer.SGD
        # model.parameters() 的用法与 PyTorch 相同
        self.optimizer = paddle.optimizer.SGD(learning_rate=learning_rate, parameters=self.model.parameters())

        self.Epoch = Epoch
        self.BSize = BSize
        logger.info("Epoch : {} BSize : {}".format(self.Epoch, self.BSize))

        self.rho = (eps**2) / (4 * np.log(1 / delta))
        self.device = device

    # Decay 和 DecayBudget 方法使用纯 Python 和 Numpy，无需改动
    def Decay(self, local_iter):
        if DecayClip == "LD":
            C0 = fix_Clip
            kc = 0.5
            return C0 * (1 - kc * local_iter)
        elif DecayClip == "ED":
            C0 = fix_Clip
            kc = 0.01
            return C0 * np.exp((-1) * kc * local_iter)
        else:
            C0 = fix_Clip
            kc = 0.5
            return C0 / (1 + kc * local_iter)

    def DecayBudget(self, e, K_segma):
        rho_0 = self.rho / (self.Epoch)
        C0 = 1 / np.sqrt(2 * rho_0)
        if DecayMODE == "LD":
            kc = K_segma
            return C0 * (1 - kc * e)
        elif DecayMODE == "ED":
            kc = K_segma
            return C0 * np.exp((-1) * kc * e)
        else:
            kc = K_segma
            return C0 / (1 + kc * e)

    def runOurs(self, data_set_asarray, label_set_asarray, loss_fn, DecayClip, Attack, device, xvali, yvali):
        num_weights = 0
        self.model.train()  # 切换到训练模式，用法与 PyTorch 相同
        for epoch in range(self.Epoch):
            logger.info("mode:%s,epoch:%d" % ("ours but gradient clip", epoch))
            if epoch == 0:
                rho = self.rho
                logger.info(f"rho : {rho}")
            else:
                rho = rho - 1 / (2 * (segma**2))
                logger.info(f"rho : {rho}")
                if rho <= 0:
                    return self.model, num_weights

            segma = self.DecayBudget(epoch, K_segma)
            logger.info("segma~ : {}".format(segma))
            running_loss = 0.0

            for local_iter in range(len(self.data_ind)):
                batch_ind = self.data_ind[local_iter]
                batch_instance = [batch_ind[i] for i in range(len(batch_ind))]

                # 注意：这里需要确保输入的数据已经是 paddle.Tensor 类型
                # 如果 data_set_asarray 是 numpy array, 需要先转换
                x = data_set_asarray[[int(j) for j in batch_instance]].to(device)
                y = label_set_asarray[[int(j) for j in batch_instance]].to(device)

                # PaddlePaddle 的独特函数：optimizer.clear_grad()
                self.optimizer.clear_grad()

                logits = self.model(x)

                # 注意: 为了能计算逐样本梯度，loss_fn 的 reduction 参数应为 'none'
                # e.g., paddle.nn.CrossEntropyLoss(reduction='none')
                loss_value = loss_fn(logits, y)

                myGrad = []
                Ct = self.Decay(local_iter)

                for idx in range(len(logits)):
                    with paddle.no_grad():
                        running_loss += loss_value[idx]

                    # PaddlePaddle 的独特函数：paddle.grad()
                    # 计算单个样本损失对模型参数的梯度
                    dy_dx = paddle.grad(loss_value[idx], self.model.parameters(), retain_graph=True)

                    # .detach().clone() 的用法与 PyTorch 相同
                    cur_gradients = list((_.detach().clone() for _ in dy_dx))
                    num_weights = len(cur_gradients)

                    l2_norm = paddle.to_tensor(0.0)  # 初始化为 paddle tensor
                    for i in range(num_weights):
                        # PaddlePaddle 的独特函数：paddle.norm()
                        l2_norm = l2_norm + (paddle.norm(cur_gradients[i], p=2) ** 2)

                    # PaddlePaddle 的独特函数：paddle.sqrt()
                    l2_norm_ = paddle.sqrt(l2_norm)

                    factor = l2_norm_ / Ct
                    if factor < 1.0:
                        factor = 1.0

                    clip_gradients = [cur_gradients[i] / factor for i in range(num_weights)]

                    if idx == 0:
                        # PaddlePaddle 的独特函数：paddle.unsqueeze()，使用 axis 参数
                        per_gradient = [paddle.unsqueeze(clip_gradients[i], axis=-1) for i in range(num_weights)]
                    else:
                        # PaddlePaddle 的独特函数：paddle.concat()，使用 axis 参数
                        per_gradient = [
                            paddle.concat((per_gradient[i], paddle.unsqueeze(clip_gradients[i], axis=-1)), axis=-1)
                            for i in range(num_weights)
                        ]

                # PaddlePaddle 的独特函数：paddle.mean()，使用 axis 参数
                myGrad = [paddle.mean(per_gradient[i], axis=-1) for i in range(num_weights)]

                GaussianNoises = [
                    1.0 / x.shape[0] * np.random.normal(loc=0.0, scale=float(segma * Ct), size=myGrad[i].shape)
                    for i in range(num_weights)
                ]

                # .cpu().numpy() 的用法与 PyTorch 相同
                noiseGrad = [myGrad[i].cpu().numpy() + GaussianNoises[i] for i in range(num_weights)]

                # 手动为模型参数赋予梯度
                for p, newGrad in zip(self.model.parameters(), noiseGrad):
                    # PaddlePaddle 的独特函数：paddle.to_tensor()
                    p.grad = paddle.to_tensor(newGrad, dtype='float32').to(device)

                # optimizer.step() 的用法与 PyTorch 相同
                self.optimizer.step()

                if local_iter % 100 == 0 and local_iter != 0:
                    with paddle.no_grad():
                        # logger.info("local iter:%d loss:%.6f"%(local_iter,running_loss/100))
                        running_loss = 0.0

            logger.info("loss:%.6f" % (running_loss / len(self.data_ind)))
            np.random.shuffle(self.ori_data)
            splitNum = np.ceil(len(self.ori_data) / self.BSize)
            self.data_ind = np.array_split(np.asarray(self.ori_data), splitNum, 0)

        return num_weights
