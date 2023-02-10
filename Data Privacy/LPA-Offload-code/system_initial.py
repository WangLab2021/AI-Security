import numpy as np
import random
import math
import pandas as pd


from Parameters import  gamma,kappa,\
                        freq_cpu_local,freq_cpu_server,\
                        B_k,N_0,rho,theta

'''
Calculating user offloads matrix
****parameter****
K: the number of server
T: Temperature parameters in SA algorithm
'''


def task_allocate(K, T):        #random allocation
    # task assign matrix—— x axis is i,represents the id of task; y axis is k, represents the id of server
    matrix_A =  [[0 for j in range(T)] for i in range(K)]
    # if the user offloads the task_i to the server k
    matrix_A = np.array(matrix_A)
    seed = "01"
    for k in range(K):     #k server
        for t in range(T):
            # repeats delate
            if (matrix_A[:,t]==1).any():
                continue
            else:
                matrix_A[k][t] = random.choice(seed)
    # print(matrix_A)
    return matrix_A

'''
Calculating user-side consumption
****parameter****
list x_k is a One-dimensional array, indicates where the task is computed 
SingleServer: x_k = [0,1,0,1,0....,0,1,...,1]. if x_k[i] = 1, task i is assigned to server; otherwise task i is at the local
MultiServer: x_k = [1,9,2,0,0,3,3,...,1,2,4,....,9].  if x_k[i] = 3, task i is assigned to server 3; if x_k[i] = 0, task i is at the local 
x_k and vi is the same size
'''


def local_consumption(x_k,vi):
    LocalDelay = 0.0
    LocalEnergy = 0.0
    for i in range(len(x_k)):
        if x_k[i] == 0:   #at local
            LocalDelay = LocalDelay + vi[i] * gamma/freq_cpu_local
            LocalEnergy = LocalEnergy + vi[i] * gamma * freq_cpu_local * freq_cpu_local * kappa
    return LocalDelay,LocalEnergy

'''
Calculating server-side consumption
****parameter****
x_k: the list that user offloads matrix
vi: the sizes of task
K: the number of server
lk: contains the distance between user and all servers——lk[i] = 50(m) means the distance between user and server i is 50;
SingleServer: lk = [6], with only one value
MultiServer: lk = [4,30,9,18,...,12], with more than one value
****return****
ServerDelay: Server delay time
ServerEnergy: server energy cost
'''


def server_consumption(x_k,vi,lk,server_num):
    ServerDelay = np.zeros(int(server_num))
    ServerEnergy = np.zeros(int(server_num))
    for i,value in enumerate(x_k):
        num = int(x_k[i])
        if x_k[i] != 0 and i<len(x_k)-1:
            H_k = math.pow((1 / lk[num - 1]), theta)
            R_k = B_k * math.log(1 + H_k * rho / (B_k * N_0), 2)

            ServerDelay[num-1] = ServerDelay[num-1] + vi[i]/R_k + vi[i]*gamma/freq_cpu_server
            ServerEnergy[num-1] = ServerEnergy[num-1] + vi[i]*rho/R_k
    return ServerDelay,ServerEnergy

'''
Calculate the cost of offloading data from the user to the server 
****parameter****
x_k: the list that user offloads matrix----[1,2,3,3,4]
vi: the size of tasks---[0.2,0.8,1.0,0.5]
lk: the distance between user and server(s)
****return****
Total_cost: the cost of user in offloading decision of x_k
'''


def offloading_cost(x_k, vi, lk, server_num,weight3):

    WEIGHT1 = (1 - weight3) / 2
    WEIGHT2 = (1 - weight3) / 2
    # print("WEIGHT1=%f,WEIGHT2=%f"%(WEIGHT1,WEIGHT2))
    # K = max(x_k)
    # print("offloading x_k:",x_k)
    LocalDelay, LocalEnergy = local_consumption(x_k, vi)
    ServerDelay, ServerEnergy = server_consumption(x_k,vi,lk,server_num)
    # print("LocalDelay,LocalEnergy:", LocalDelay, LocalEnergy)
    # print("ServerDelay, ServerEnergy:", ServerDelay, ServerEnergy)
    if ServerDelay.size != 0:
        Total_Delay = max(LocalDelay, max(ServerDelay))
    else:
        Total_Delay = LocalDelay
    if ServerEnergy.size != 0:
        Total_Energy = LocalEnergy + sum(ServerEnergy)
    else:
        Total_Energy = LocalEnergy
    Total_cost = WEIGHT1 * Total_Delay + WEIGHT2 * Total_Energy

    return Total_cost

'''
Function extensions of offloading_cost：Add return of Total_Delay and Total_Energy
'''


def offloading_cost_ext(x_k, vi, lk, server_num,weight3):

    WEIGHT1 = (1 - weight3) / 2
    WEIGHT2 = (1 - weight3) / 2
    LocalDelay, LocalEnergy = local_consumption(x_k, vi)
    ServerDelay, ServerEnergy = server_consumption(x_k,vi,lk,server_num)
    if ServerDelay.size != 0:
        Total_Delay = max(LocalDelay, max(ServerDelay))
    else:
        Total_Delay = LocalDelay
    if ServerEnergy.size != 0:
        Total_Energy = LocalEnergy + sum(ServerEnergy)
    else:
        Total_Energy = LocalEnergy
    Total_cost = WEIGHT1 * Total_Delay + WEIGHT2 * Total_Energy

    return Total_cost,Total_Delay,Total_Energy

'''
calculate the distance between user and servers in single server scenario
'''


def distance_calculation1(user1):       
    distance = math.pow(user1,1)
    return distance

'''
calculate the distance between user and servers in multi server scenario
'''


def distance_calculation2(user2,server2):       
    distances = []
    for k in range(len(server2)):
        euclidean_distance = math.pow((user2['X_INDEX']-server2[k]['LATITUDE'])**2 + (user2['Y_INDEX']-server2[k]['LONGITUDE'])**2 , 0.5)
        distances.append(euclidean_distance)
    return distances
