import math


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

gamma=1000
kappa = math.pow(10,-27)
freq_cpu_local=math.pow(10,9)
freq_cpu_server=math.pow(10,10)
# B_k=5*math.pow(10,6)
B_k = 0.1*math.pow(10,6)
# N_0=math.pow(10,-13)
N_0=3.9810717055349565*math.pow(10,-21)
#eta = 0.5       #this parameter may be a list. It's a variable, and should be pre-set.    allocate_rate = 1 / T
rho = 0.1
#h_k = 1         #this parameter may be a list
# g_0 = math.pow(10,-4)
#l_0 = 1
theta = 4


WEIGHT3 = 0.6

EPSILON = 0.1