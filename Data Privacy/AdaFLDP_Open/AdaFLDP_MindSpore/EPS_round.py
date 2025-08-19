import mindspore as ms
import mindspore.nn as nn
import numpy as np
from Global_Parameter import rounds
from tools import logger


class EPS_round:

    def __init__(self, valiloader):
        self.valiloader = valiloader
        self.delta_S = 1.0
        self.mum_S = 0.0
        self.best_count = 0.0
        self.factor = 0.5
        self.lr_start = False

    def RoundlyAccount(self, old_global_model, eps_global, t, device, Epoch) -> float:
        total = 0
        correct = 0

        old_global_model.set_train(False)  # 设置为评估模式，不更新 BN 和 Dropout

        for data in self.valiloader.create_tuple_iterator():
            images, labels = data
            predictions_logits = ms.ops.stop_gradient(old_global_model(images))
            predictions = predictions_logits.argmax(axis=1).astype(labels.dtype)
            batch_correct = (predictions == labels).sum().asnumpy()
            correct += batch_correct
            total += labels.shape[0]

        # 计算验证集准确率
        daughter_S = correct / total if total > 0 else 0.0

        # Step 1.2: 根据准确率动态调整隐私预算分配策略
        if t > 2:
            if daughter_S - self.mum_S > 0.07:
                self.lr_start = True

            if not self.lr_start:
                if daughter_S <= self.best_count:
                    self.factor *= 0.4
                else:
                    self.factor *= 0.6

            if daughter_S > self.best_count:
                self.best_count = daughter_S

            if not self.lr_start:
                delta_S = self.factor
            else:
                delta_S = min(daughter_S - self.mum_S, self.factor)

            # 更新 delta_S
            if 0.01 < delta_S < self.delta_S:
                self.delta_S = delta_S
            elif 0 <= delta_S < 0.01:
                self.delta_S = 0.0

        # Step 1.3: 计算当前轮次的隐私预算
        eps_round = np.exp(-1.0 * self.delta_S) * eps_global / (rounds - t + 1)

        # 更新上一轮的准确率记录
        self.mum_S = daughter_S

        return float(eps_round)
