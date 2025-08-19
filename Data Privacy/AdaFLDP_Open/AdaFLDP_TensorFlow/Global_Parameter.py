# Global_Parameter
###终端的数量
from pickle import TRUE
import numpy as np

# delta
delta = 1 / 100000

total_client_num = 50
###全局的round数量
rounds = 15
###采样率

###每次选取的用户数量
parti_client_num = 10
l2_norm_clip = 10
# noise_multiplier = 0.01

###学习率设置
learning_rate = 0.25
# 是否使用PRIVACY MECHANISM
PRIVACY_MODE = 'ori'  #'fix','ori','adp'
# 数据集初始化设置
DATAMODE = 'mnist'
# 是否使用Adaptive_privacy
ADP = TRUE
# Epoch global
E = 1
# DecayClip
# LD:Linear Decay
# ED:Exponential Decay
# STC:Search-Then-Converge Decay
DecayClip = "ED"
# DecayBudget hpParameter:
K_segma = 0.1
K_clip = 0.5
beta = 1.11
DecayMODE = "ED"
# PLD中，用来控制dummy_gradient的倍率Gama
gama = 1.0
pldT = 60
# step 2
Lambda = 0.2
# Fix privacy parameters
fix_segma = 0.2
fix_Clip = 4

channel_first = False  # TensorFlow 默认使用 channels_last 格式，但可以设置为 channels_first