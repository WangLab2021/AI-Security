# Step3
# 在每个CLIENT进行本地训练时初始化，用来计算CLIENT本地训练过程的隐私消耗
from Global_Parameter import *
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from dlgAttack import DLA
from tools import logger


# k_instance = EPS_instance(data_ind,model,Epoch,BSize,eps)
class EPS_instance:
    def __init__(self, data_ind, model, Epoch, BSize, eps, device):
        self.ori_data = data_ind
        np.random.shuffle(self.ori_data)
        splitNum = np.ceil(len(self.ori_data) / BSize)
        self.data_ind = np.array_split(np.asarray(self.ori_data), splitNum, 0)
        self.model = model
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        self.Epoch = Epoch
        self.BSize = BSize
        logger.info("Epoch : {} BSize : {}".format(self.Epoch, self.BSize))
        # self.rho = 4*np.log(1/delta)*(eps**2)
        self.rho = (eps**2) / (4 * np.log(1 / delta))
        self.device = device

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
                # logger.info(batch_instance,len(batch_instance))
                # batch_instance = np.array(batch_instance, dtype=np.int64)
                indices = tf.constant([int(j) for j in batch_instance], dtype=tf.int32)
                x = tf.gather(data_set_asarray, indices)
                y = tf.gather(label_set_asarray, indices)
                # y = label_set_asarray[[int(j) for j in batch_instance]]

                # Convert to TensorFlow tensors
                x = tf.convert_to_tensor(x, dtype=tf.float32)
                # logger.info(y)
                y = tf.cast(y, tf.int64)

                Ct = self.Decay(local_iter)
                per_gradient = []

                # Process each sample in the batch individually for per-sample gradients
                for idx in range(len(x)):
                    x_sample = tf.expand_dims(x[idx], 0)
                    y_sample = tf.expand_dims(y[idx], 0)

                    with tf.GradientTape() as tape:
                        logits = self.model(x_sample, training=True)
                        loss_value = loss_fn(y_sample, logits)
                        running_loss += loss_value

                    # Compute gradients for this sample
                    cur_gradients = tape.gradient(loss_value, self.model.trainable_variables)
                    num_weights = len(cur_gradients)

                    # Compute L2 norm of gradients
                    l2_norm = 0.0

                    for grad in cur_gradients:
                        l2_norm += tf.square(tf.norm(grad, ord=2))
                    l2_norm_ = tf.sqrt(l2_norm)

                    # Gradient clipping
                    factor = l2_norm_ / Ct
                    factor = tf.maximum(factor, 1.0)
                    clip_gradients = [grad / factor for grad in cur_gradients]

                    if idx == 0:
                        per_gradient = [tf.expand_dims(grad, -1) for grad in clip_gradients]
                    else:
                        per_gradient = [
                            tf.concat([per_gradient[i], tf.expand_dims(clip_gradients[i], -1)], -1)
                            for i in range(num_weights)
                        ]

                # Average gradients across the batch
                myGrad = [tf.reduce_mean(per_gradient[i], -1) for i in range(num_weights)]

                # Add Gaussian noise
                GaussianNoises = [
                    tf.random.normal(shape=myGrad[i].shape, mean=0.0, stddev=float(segma * Ct) / x.shape[0])
                    for i in range(num_weights)
                ]
                noiseGrad = [myGrad[i] + GaussianNoises[i] for i in range(num_weights)]

                # Apply gradients
                self.optimizer.apply_gradients(zip(noiseGrad, self.model.trainable_variables))

                if local_iter % 100 == 0 and local_iter != 0:
                    # logger.info("local iter:%d loss:%.6f"%(local_iter,running_loss/100))
                    running_loss = 0.0
                # if local_iter == 999:
                #     logger.info('\n')

            logger.info("loss:%.6f" % (running_loss / len(self.data_ind)))
            np.random.shuffle(self.ori_data)
            splitNum = np.ceil(len(self.ori_data) / self.BSize)
            self.data_ind = np.array_split(np.asarray(self.ori_data), splitNum, 0)

        return num_weights
