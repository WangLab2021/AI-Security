#Step1
import tensorflow as tf
from Global_Parameter import *
import numpy as np
import matplotlib.pyplot as plt
from tools import logger


class EPS_round:
    def __init__(self, valiloader):
        self.valiloader = valiloader
        self.delta_S = 1
        self.mum_S = 0
        self.best_count = 0.0
        self.factor = 0.5
        self.lr_start = False
    
    def RoundlyAccount(self, old_global_model, eps_global, t, device, Epoch):  # 计算得该轮的隐私预算
        old_global_model.trainable = False  # Set to evaluation mode
        
        total = 0
        correct = 0
        
        for images, labels in self.valiloader:
            # Move data to specified device if needed
            if device == '/GPU:0' or 'gpu' in device.lower():
                with tf.device(device):
                    images = tf.cast(images, tf.float32)
                    labels = tf.cast(labels, tf.int64)
            else:
                images = tf.cast(images, tf.float32)
                labels = tf.cast(labels, tf.int64)
            
            # Forward pass
            logits = old_global_model(images, training=False)
            predictions = tf.argmax(logits, axis=1)
            
            total += tf.shape(labels)[0]
            correct += tf.reduce_sum(tf.cast(tf.equal(predictions, labels), tf.int32))
        
        # Convert tensors to numpy for calculations
        total = total.numpy() if hasattr(total, 'numpy') else total
        correct = correct.numpy() if hasattr(correct, 'numpy') else correct
        
        daughter_S = correct / total
        
        # 对delta_S的处理 #
        
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
            if delta_S > 0.01 and delta_S < self.delta_S:
                self.delta_S = delta_S
            elif delta_S < 0.01 and delta_S >= 0:
                self.delta_S = 0
        
        eps_round = np.exp((-1.0) * (self.delta_S)) * eps_global / (rounds - t + 1)
        
        self.mum_S = daughter_S
        
        return eps_round