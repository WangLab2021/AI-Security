import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np

from Global_Parameter import *
from dlgAttack import DLA
from tools import logger


class EPS_instance:
    def __init__(self, data_ind, model, Epoch, BSize, eps, device):
        self.ori_data = data_ind.copy()
        np.random.shuffle(self.ori_data)
        splitNum = np.ceil(len(self.ori_data) / BSize)
        self.data_ind = np.array_split(np.asarray(self.ori_data), splitNum, 0)

        self.model = model
        self.optimizer = nn.SGD(self.model.trainable_params(), learning_rate=learning_rate)
        self.Epoch = Epoch
        self.BSize = BSize
        logger.info("Epoch : {} BSize : {}".format(self.Epoch, self.BSize))
        self.rho = (eps**2) / (4 * np.log(1 / delta))  # privacy budget
        self.device = device

    def Decay(self, local_iter):
        if DecayClip == "LD":
            C0 = fix_Clip
            kc = 0.5
            return C0 * (1 - kc * local_iter)
        elif DecayClip == "ED":
            C0 = fix_Clip
            kc = 0.01
            return C0 * np.exp(-kc * local_iter)
        else:
            C0 = fix_Clip
            kc = 0.5
            return C0 / (1 + kc * local_iter)

    def DecayBudget(self, e, K_segma):
        rho_0 = self.rho / self.Epoch
        C0 = 1 / np.sqrt(2 * rho_0)
        if DecayMODE == "LD":
            kc = K_segma
            return C0 * (1 - kc * e)
        elif DecayMODE == "ED":
            kc = K_segma
            return C0 * np.exp(-kc * e)
        else:
            kc = K_segma
            return C0 / (1 + kc * e)

    def runOurs(self, data_set_asarray, label_set_asarray, loss_fn, DecayClip, Attack, device, xvali, yvali):
        """
        使用 MindSpore 实现的带有逐样本梯度裁剪和噪声的训练函数。

        注意: 传入的 loss_fn 必须设置为 a reduction='none'，以便返回每个样本的 loss。
        例如: loss_fn = nn.CrossEntropyLoss(reduction='none')
        """
        num_weights = 0

        self.model.set_train(True)

        def single_sample_loss_fn(data, label):
            logits = self.model(data)
            loss = loss_fn(logits, label)
            return loss[0]

        # 3. 创建一个梯度函数，用于计算单个样本的梯度
        grad_fn = ms.value_and_grad(single_sample_loss_fn, grad_position=None, weights=self.model.trainable_params())

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
            logger.info(f"segma~: {segma}")

            running_loss = 0.0
            for local_iter in range(len(self.data_ind)):
                batch_ind = self.data_ind[local_iter]

                # 4. 准备批次数据，并转换为 MindSpore Tensor
                x_np = data_set_asarray.asnumpy()[batch_ind]
                y_np = label_set_asarray.asnumpy()[batch_ind]
                x = ms.Tensor(x_np, ms.float32)
                y = ms.Tensor(y_np, ms.int32)

                # self.optimizer.zero_grad()

                myGrad = []
                Ct = self.Decay(local_iter)
                grad_record = {}
                for idx in range(x.shape[0]):
                    # 获取当前样本的梯度
                    # grad_fn 需要一个批次维度，因此使用切片 x[idx:idx+1]
                    (cur_loss), dy_dx = grad_fn(x[idx : idx + 1], y[idx : idx + 1])
                    running_loss += cur_loss.asnumpy().item()
                    cur_gradients = [g for g in dy_dx]
                    num_weights = len(cur_gradients)

                    # 使用ops.norm计算每个张量的范数，然后求和平方
                    # for g in cur_gradients:
                    #     logger.info(f"grad shape: {g.shape}, dtype: {g.dtype}")

                    l2_norm = 0.0
                    for grad in cur_gradients:
                        _norm_val = ops.norm(grad.flatten(), 2)
                        l2_norm += _norm_val ** 2
                        # l2_norm += ops.norm(grad, 2) ** 2
                    #     l2_norm += tf.reduce_sum(ops.square(grad))
                    # l2_norm_ = tf.sqrt(l2_norm)
                    # l2_norm_sq = ops.sum([ops.norm(g, 2) ** 2 for g in cur_gradients])
                    l2_norm_ = ops.sqrt(l2_norm)

                    factor = l2_norm_ / Ct
                    if factor < 1.0:
                        factor = 1.0

                    clip_gradients = [g / factor for g in cur_gradients]

                    for i in range(len(clip_gradients)):
                        grad_record.setdefault(i, []).append(clip_gradients[i])

                myGrad = [ops.mean(ops.stack(grad, axis=-1), axis=-1) for grad in grad_record.values()]

                # 7. 添加高斯噪声
                GaussianNoises = [
                    ms.from_numpy(
                        1.0
                        / x.shape[0]
                        * np.random.normal(loc=0.0, scale=float(segma * Ct), size=g.shape).astype(np.float32)
                    )
                    for g in myGrad
                ]

                final_grads = [myGrad[i] + GaussianNoises[i] for i in range(num_weights)]

                # 8. 使用优化器更新权重
                self.optimizer(tuple(final_grads))

                if local_iter % 100 == 0 and local_iter != 0:
                    # 在MindSpore中，可以使用 ms.metrics.Accuracy 来计算精度
                    # preds = self.model(xvali)
                    # acc_metric = ms.metrics.Accuracy()
                    # acc_metric.update(preds, yvali)
                    # acc = acc_metric.eval()
                    # logger.info(f"local iter: {local_iter} loss: {running_loss / 100:.6f}")
                    # logger.info(f"acc: {acc:.6f}")
                    running_loss = 0.0

                # if local_iter == 999:
                #     logger.info('\n')

            logger.info(f"loss: {running_loss / len(self.data_ind):.6f}")
            np.random.shuffle(self.ori_data)
            splitNum = np.ceil(len(self.ori_data) / self.BSize)
            self.data_ind = np.array_split(np.asarray(self.ori_data), splitNum, 0)

        return num_weights
