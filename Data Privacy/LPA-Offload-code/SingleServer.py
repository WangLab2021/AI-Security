import math
import time
import numpy as np
import sympy
import pandas as pd

from system_initial import task_allocate,distance_calculation1,offloading_cost
# from Parameters import WEIGHT3,EPSILON

from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline

EPSILON = 0.9
ALPHA = 0.1
LAMBDA = 0.9

# Default parameter settings
''' 
    channel bandwidth:     B=5(MHz)
    background noise:    N_0=-174(dBM/Hz)
    Energy consumed by CPU/per revolution:  K_cpu = 10^(-27)(J/cycle)
    user CPU frequence: freq_cpu=1(GHz) 
    MEC_CPU frequence:	freq_MEC=10(GHz)
    Calculate density: gama=1000(cycles/bit)
    Transmission power:  p=500(mW)

    #from user to edge
    Path loss constant : g_0 = -40(dB)
    Path loss index: thita = 4
    gauged distance: l_0 =1(m)

'''

STANDARD_S_N = 0.25 * pow(10, 13)  # rate
TRANSMIT_POWER = 0.1  # w
BANDWIDTH = 5 * pow(10, 6)  # Hz
ENERGY_CONSUMPTION_USER = pow(10, -6)  # J/bit
TIME_BIT_USER = pow(10, -6)  # s/bit
TIME_BIT_SERVER = pow(10, -7)  # s/bit

WEIGHT1 = 1
WEIGHT2 = 1
# WEIGHT3 = 0.002
WEIGHT3 = 0.6
epsilon = 0.1

'''
generate_probability——produce the Pr(l'|l)
distance: where the l' located----matrix [0,2,4,6...18,20], the half lenth of (l2-l1)
probability: probability of the l in l'
('distance' and 'probability' have a one-to-one correspondence)
****parameter****
    epsilon: epsilon Laplace related parameter
    l: the original location of user
    l1: the lower Scope after confusion
    l2 : the upper Scope after confusion
'''
def generate_probability(epsilon, l, l1, l2):
    # array 'distance' and 'probability' have a one-to-one correspondence
    distance = np.zeros(int((l2 - l1) / 2) + 1)
    probability = np.zeros(int((l2 - l1) / 2) + 1)
    j = 0
    for i in np.arange(l1, l2 + 1, 2):
        distance[j] = i
        j = j + 1

    deltaL = (l2 - l1)
    x = sympy.symbols('x')
    y = (epsilon / (2 * deltaL)) * sympy.exp(-epsilon * abs(x - l) / deltaL) *1/(1-0.5*
                sympy.exp((epsilon * l1 - epsilon * l) / deltaL) -0.5 * sympy.exp((-epsilon * (l2 - l)) / deltaL))
    # y = (epsilon / 2* deltaL)*(-epsilon)^(-epsilon*(x-1)) + epsilon^(epsilon*l1-epsilon*l) + epsilon(-epsilon*(l2-l))/(2*(l2-l1)
    j = 0
    for i in np.arange(l1, l2 + 1, 2):
        probability[j] = sympy.integrate(y, (x, i, i + 2))
        # print("probilities=",probability[j])
        j = j + 1
    # print(probability)
    privacyLeakage = Privacy_leakage_comp(epsilon, l1, l2, l)

    return distance, probability, privacyLeakage


def Privacy_leakage_comp(epsilon, l1, l2, l):
    deltaL = (l2 - l1)

    privacyLeakage = 0

    ll = sympy.symbols('ll')
    P_pdf = (epsilon / (2 * deltaL)) * sympy.exp(-epsilon * abs(ll - l) / deltaL) *1/(1-0.5*
                sympy.exp((epsilon * l1 - epsilon * l) / deltaL) -0.5 * sympy.exp((-epsilon * (l2 - l)) / deltaL))
    P_pro = sympy.integrate(sympy.log(1/P_pdf), (ll, l, l + 2))
    privacyLeakage = -P_pro
    return privacyLeakage


def obfuscate(distance, probability):
    x = np.random.uniform(0, 1)
    item =0.0
    itemProb =0.0
    cumulativeProb = 0.0
    for item, itemProb in zip(distance, probability):
        cumulativeProb += itemProb
        if x < cumulativeProb:
            break
    return item, itemProb


'''
Main calculation function
******parameters******
allocate_rate:  allocated proportion of bandwidth(eta)
TASK: the situation of task. TASK = [1,0,0,1,...,1]
TASKNUM: the number of task in user
l: user's location 
lmax: range of server
******returns********
ZATA: final cost
opt_l1: final low range 
opt_l2: final high range
K: number of server
'''
def update_range(TASK,vi,TASKNUM,l,lmax,K):
    ZATA=0
    opt_l1=0
    opt_l2=0
    global epsilon
    for l1 in range(0, l, 1):
        for l2 in range(l, lmax, 1):
            # lk1 = [0] * K  # distance between user and server
            # lk2 = [0] * K  # fakedistance between fakelocation and server
            time3 = time.time()
            distance, probability, privacyLeakage = generate_probability(epsilon, l, int(l1), int(l2))
            time4 = time.time()
            print("Time consuming to generate random points：",time4-time3)
            lk1 = distance_calculation1(l)      # initial distance
            utility =0
            for rands in range(1000):
                # print(i)
                obfuscatedDistance, prob = obfuscate(distance, probability)         # produce different obfuscatedDistance each time
                lk2 = obfuscatedDistance
                count = 0
                opt_TA = []
                max_R = 0
                opt_cost = 0
                for j in range(2 ** TASKNUM):  # Task Allocation (TA) traverse
                    #initial
                    s = j
                    e = [0] * TASKNUM
                    for i in range(TASKNUM):      # each task(bit)
                        e[i] = int(s % 2)
                        s = s // 2
                    e.reverse()
                    lk2_temp = []
                    lk2_temp.append(lk2)
                    fakeCost = offloading_cost(e, vi, lk2_temp, K, weight3)  # obfuscate location lk2
                    # fakeCost = offloading_cost(e,vi, lk2,K,weight3)     # obfuscate location lk2
                    R = -fakeCost
                    if count == 0:  # run only once for ynitializing max_R value
                        max_R = R
                        count += 1
                    elif R >= max_R:
                        max_R = R
                        opt_TA = e
                        # cost = offloading_cost(e,vi, lk1, K,weight3)  # initial location lk1
                        lk1_temp = []
                        lk1_temp.append(lk1)
                        cost = offloading_cost(e, vi, lk1_temp, K, weight3)  # obfuscate location lk1
                        opt_cost = cost
                print("range:[%.2f,%.2f]----Rounds %d, best offload matrix:%s"%(l1,l2,rands,opt_TA))
                utility += - (opt_cost + WEIGHT3 * privacyLeakage)
            utility = float(utility / 1000)
            print("[%.2f,%.2f],utility=%f;ZATA=%f"%(l1,l2,utility,ZATA))
            if utility > ZATA:
                ZATA = utility
                opt_l1=l1
                opt_l2=l2

    return ZATA,opt_l1,opt_l2

if __name__ == '__main__':
    time1 = time.time()
    #pre-set
    # global WEIGHT3,EPSILON
    weight3 = WEIGHT3
    epsilon = EPSILON
    user1 = 20      #user1's location will random selected in [1,50]
    # lmax = 50
    lmax = 1000
    # x_k = [1,0,1,0]
    # vi = [2,3,1,1]
    x_k = [1,0,1,0,1,1,0,1,0,1,0]
    vi = [2,3,1,1,2,3,10,5,7,19,3]
    vi = [1024*1024*i for i in vi]
    print("vi:",vi)
    T = len(x_k)
    K = max(x_k)       # single tags ===> k = 1
    allocate_rate = 1/T
    # main calculation function
    max_tradeoff,opt_l1,opt_l2 = update_range(x_k,vi,T,user1,lmax,K)
    time2 = time.time()
    print("Total Cost：",time2-time1)


