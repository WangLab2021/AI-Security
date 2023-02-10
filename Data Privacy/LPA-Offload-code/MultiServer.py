# -*- coding: UTF-8 -*-
import math
import time
import numpy as np
import sympy
import pandas as pd
import random
import sys
import os
from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline

from system_initial import task_allocate,distance_calculation2,offloading_cost,offloading_cost_ext
from GA import choiceFuction,crossoverFuntion,matrix_translate
from Parameters import WEIGHT3,EPSILON



'''
generate_probability——produce the Pr(l'|l)
****parameter****
    epsilon: DP Laplace related parameter
    r_max: the selected obfuscated range
    angle1&angle2: the angle range [angle1, angle2]
*****return******
privacyLeakage: the privacy leakage of user at the r_max
'''


def privacy_leakage_comp(epsilon,r_max,angle1,angle2):
    privacyLeakage = 0
    r , angles= sympy.symbols('r, ANGLE')
    deltaR = (r_max - 0)
    C = (1 - sympy.exp(-epsilon))/2
    P_pdf = 1/(abs(angle1-angle2)) * ( (epsilon/(2*deltaR)) * (sympy.exp(-(epsilon*r)/deltaR)) ) * (1/C)
    P_pro = sympy.integrate(sympy.log(1/P_pdf), (angles,angle1,angle2),(r,0,r_max))
    privacyLeakage = -sympy.log(P_pro)
    return privacyLeakage


'''
random select a value x in [0,1]; cdf(x,y)-->y is the obfuscated range
'''


def obfuscated_range(epsilon,r_max):
    x = np.random.uniform(0, 1)
    cdf = - (r_max*math.log(1-(1-math.exp(-epsilon))*x))/epsilon
    r_new = cdf
    return r_new

'''
calculate the angle1&angle2 based on the distribution of user2&server2
'''


def calculate_angle(user2,server2,r_max):

    angle1 = 45/180 * math.pi
    angle2 = 315/180 * math.pi
    return angle1,angle2

'''
random select the perturb location of user3
'''


def obfuscate_location(user3, r_new, angle1, angle2):
    rand_r1 = random.random()*(2) - 1
    rand_r2 = random.choice([-1,1]) * math.sqrt(1-rand_r1*rand_r1)
    # print("rand_1,rand_2",rand_r1,rand_r2)
    user3_new = {'X_INDEX': user3['X_INDEX']+ rand_r1 * r_new, 'Y_INDEX': user3['Y_INDEX']+ rand_r2 * r_new}
    return user3_new


'''
pop_temp: Intermediate representation of populations,
NUM：server_num
b: the individual with the greatest contemporary adaptation
'''


def choice2(pop_temp,NUM,b,lk,server_num):
    # The best value of the previous generation replaces the worst value in the current generation
    for s in range(NUM):
        cost = offloading_cost(pop_temp[s][:-1], vi, lk,server_num,weight3)
        pop_temp[s][-1] = cost
    c,d =choiceFuction(pop_temp)
    # The optimal value of the previous generation replaces the worst value in the current generation
    pop_temp[c] = b
    return pop_temp

def choice3(pop_temp,NUM,b,lk,server_num):
    #select NUM better pop from pop_temp(NUM+new num)
    for s in range(len(pop_temp)):
        cost = offloading_cost(pop_temp[s][:-1], vi, lk,server_num,weight3)
        pop_temp[s][-1] = cost
    pop_temp = pop_temp[np.argsort(pop_temp[:, -1]), :]
    pop_temp2 = pop_temp[0:NUM,:]
    return pop_temp2



'''
Generate the best individual with the greatest fitness
******parameters*******
Z: Iterations
p_Mating: Crossover rate
p_Mutation: Mutation rate
the size of tasks: vi
Distance matrix: lk
x_k: offloading decision
*******return**********
C_min: minimum cost
matrix_X: offloading strategy
'''


def genetic_algorithm(Z,p_Mating,p_Mutation,vi,lk,server_num):

    # parameter preset
    # Z = 180     # Iteration times
    pop = []  # Stores the order of visits and the degree of adaptation of each individual
    S_NUM = 200  # Number of initialized groups

    #initial groups
    AM_ga = [[0 for col in range(server_num)] for row in range(S_NUM)]          # shape(AM) = [S,M+1]----S groups; M tasks
    #print("AM_AM:",AM_ga)
    for s in range(S_NUM):
        # x_k = []  # the list that user offloads matrix----[1,2,3,3,4]
        matrix_A = task_allocate(K, T)
        x_k1 = matrix_translate(matrix_A)        #list_A = [[A1,A2,...,Ax]]
        AM_ga[s] = x_k1    #A[s] is the offload decision
        # add a new culumn, as the fitness value
        # print("GA_LK",lk)
        cost = offloading_cost(x_k1, vi, lk,server_num,weight3)
        AM_ga[s].append(cost)
    AM_ga2 = np.array(AM_ga)       # list to array
    best = []           # the best individual with the greatest fitness
    for i in range(Z):
        # a is the index of the individual with the smallest contemporary adaptation
        # b is the individual with the largest contemporary adaptation
        a, b = choiceFuction(AM_ga2) 
        AM_temp = crossoverFuntion(AM_ga2, p_Mating, server_num, p_Mutation, S_NUM)  # mating & variation
        # AM = choice(AM_temp, S_NUM, b)
        # AM_ga2 = choice2(AM_temp, S_NUM, b,lk,server_num)
        AM_ga2 = choice3(AM_temp, S_NUM, b, lk, server_num)
    a, b = choiceFuction(AM_ga2)
    best = b[:-1]
    return best


'''
Simulated annealing algorithm
******parameters*******
Iterations: L, 
Temperature :T, 
Minimum temperature: T_min, 
Cycles to fit pdf: Cycles, 
Decay rate: alpha, 
disturbance: delta
user2: user's location
server2: servers's location
x_k: initialized [offloading decision, C_min]
distance: the distance matrix between user2 and server2
*******teturn*********
Cost: the min cost at the obfuscated location: x_2
ZATA: the best utility at the x_2
R_max: the best range after SA algorithm
AM:  the best offloading decision matrix (same as x_k) at x_2----using GA
'''


def simulated_annealing(L,T,T_min,Cycles,alpha,delta,user2,server2,x_k,distance,server_num,vi,max_range,weight3,epsilon):
    # records setting
    recordIter = []                 # Initialization, number of outer cycles
    recordU_old = []                # Initialization, the objective function value of the current solution
    recordU_new = []               # Initialization, objective function value of the best solution
    recordPBad = []                 # Initialization, acceptance probability of inferior solutions
    kIter = 0                       # number of temperature states
    # epsilon = 0.1               #preset DP
    cost_opt = 0
    pl_opt = 0
    AM = []  # Storage access order and per individual adaptation
    # AM_original = []
    # AM_max_range = []

    ZATA = float("inf")        #preset the best utility
    r_opt = 0      #preset the best utility
    r_init = 0.001   #the range increases from zero
    r_init_old = delta

    #GA parameter preset
    Z = 10  # Iteration times
    p_Mating = 0.7  # Mating probability
    p_Mutation = 0.2  # Variation probability
    # calculate the utility of original location
    cost_old = offloading_cost(x_k, vi, distance, server_num,weight3)
    print("vi:", vi)
    angle1, angle2 = calculate_angle(user2, server2, r_init_old)
    privacyLeakage = privacy_leakage_comp(epsilon, max(distance), angle1, angle2)
    print("cost,privacy", cost_old, privacyLeakage)
    privacyLeakage = privacy_leakage_comp(epsilon, r_init_old, angle1, angle2)
    print("cost,init_privacy", cost_old, privacyLeakage)
    U_old = (cost_old + weight3 * privacyLeakage)  # later use

    '''
    compare experiments setting:
        max_range offload& original location offloading
    '''
    # cost_new_original = 0
    cost_new_maxrange = 0
    AM_original = genetic_algorithm(Z, p_Mating, p_Mutation, vi, distance, server_num)
    cost_new_original = offloading_cost(AM_original, vi, distance, server_num,weight3)
    maxMarkov = 1
    for k in range(maxMarkov):

        r_new_maxrange = obfuscated_range(epsilon, max_range)  # cdf -- random select a new range  with cdf
        angle1, angle2 = calculate_angle(user2, server2, max_range)
        user_new_maxrange = obfuscate_location(user2, r_new_maxrange, angle1, angle2)
        distance_new_maxrange = distance_calculation2(user_new_maxrange, server2)
        AM_max_range = genetic_algorithm(Z, p_Mating, p_Mutation, vi, distance_new_maxrange, server_num)
        # print(AM_max_range)
        cost_new_maxrange += offloading_cost(AM_max_range, vi, distance, server_num,weight3)
    cost_new_avg_maxrange = cost_new_maxrange / maxMarkov
    privacyLeakage_new_maxrange = privacy_leakage_comp(epsilon, max_range, angle1, angle2)

    while T > T_min:                # outer circulation
        kBetter = 0                 # times of access to superior solutions
        kBadAccept = 0              # times to accept inferior solutions
        kBadRefuse = 0              # times to reject inferior solutions
        pBadAccept = 0              # probability to accept inferior solutions
        meanMarkov = 500        # preset Inner Circulation

        # U_old = 0       # preset the utility of original location
        # U_new = 0       # preset the utility of obfuscated location

        r_max_txt = []  # save r_max
        cost_new_avg_txt = [] 
        privacyLeakage_new_txt = [] 
        U_new_txt = []  

        for cycle in range(Cycles):
            time1_single = time.time()
            r_max = r_init + delta
            # preset
            cost_new = 0
            print("old:",U_old)
            print("r_max:",r_max)
            for k in range(meanMarkov):
                r_new = obfuscated_range(epsilon, r_max)        #cdf -- random select a new range  with cdf
                angle1, angle2 = calculate_angle(user2, server2, r_max)
                user_new = obfuscate_location(user2, r_new, angle1, angle2)
                distance_new = distance_calculation2(user_new, server2)
                # print("distance_new",distance_new)
                # print("distance_old",distance)
                AM = genetic_algorithm(Z, p_Mating, p_Mutation, vi, distance_new, server_num)

                # print("AM:",AM)
                cost_new += offloading_cost(AM, vi, distance, server_num,weight3)

            cost_new_avg = cost_new/meanMarkov
            print("cost_new_avg",cost_new_avg)
            r_max_txt.append(r_max)
            cost_new_avg_txt.append(cost_new_avg)
            time2_single = time.time()
            print("single_time_cost:",time2_single-time1_single)


            privacyLeakage_new = privacy_leakage_comp(epsilon, r_max, angle1, angle2)
            privacyLeakage_new_txt.append(privacyLeakage_new)
            U_new = (cost_new_avg + weight3 * privacyLeakage_new)
            U_new_txt.append(U_new)
            deltaU = U_new - U_old
            print("(U_new,U_old)--deltaU--PL:", U_new, U_old, deltaU,weight3 *privacyLeakage_new)
            U_old =U_new

            if deltaU < 0:
                accept = True
                kBetter += 1
                r_init = r_max
            else:
                # Accept the worse value with a certain probability-p
                p = math.exp(-(U_new - U_old) / T)
                r = np.random.uniform(low=0, high=1)
                if r < p:
                    accept = True
                    kBadAccept += 1
                    r_init = r_max
                else:
                    accept = False
                    kBadRefuse += 1

            if U_new < ZATA:
                ZATA = U_new
                r_opt = r_max
                cost_opt = cost_new_avg
                pl_opt = privacyLeakage_new

        if kIter % 1 == 0:  # record every 1 times
            print('i:{},t(i):{:.2f}, badAccept:{:.6f}, f(x)_best:{:.6f}'. \
                  format(kIter, T, pBadAccept, ZATA))
        # cooling
        T = alpha * T
        kIter = kIter + 1
    return cost_opt,ZATA,r_opt,AM,pl_opt,cost_new_original,cost_new_avg_maxrange,privacyLeakage_new_maxrange



'''
Module extensions of simulated_annealing：
Add return of Total_Delay and Total_Energy(opt,init,maxrange)
'''


def simulated_annealing_ext(L,T,T_min,Cycles,alpha,delta,user2,server2,x_k,distance,server_num,vi,max_range,weight3,epsilon):
    # records setting
    recordIter = []                
    recordU_old = []               
    recordU_new = []             
    recordPBad = []               
    kIter = 0                    
    cost_opt = 0
    pl_opt = 0
    AM = []  
    # AM_original = []
    # AM_max_range = []

    ZATA = float("inf")        #preset the best utility
    r_opt = 0      #preset the best utility
    r_init = 0.001   #the range increases from zero
    r_init_old = delta

    #GA parameter preset
    Z = 10  # Iteration times
    p_Mating = 0.7  # Mating probability
    p_Mutation = 0.2  # Variation probability
    # calculate the utility of original location
    cost_old = offloading_cost(x_k, vi, distance, server_num,weight3)
    print("vi:", vi)
    angle1, angle2 = calculate_angle(user2, server2, r_init_old)
    privacyLeakage = privacy_leakage_comp(epsilon, max(distance), angle1, angle2)
    print("cost,privacy", cost_old, privacyLeakage)
    privacyLeakage = privacy_leakage_comp(epsilon, r_init_old, angle1, angle2)
    print("cost,init_privacy", cost_old, privacyLeakage)
    U_old = (cost_old + weight3 * privacyLeakage)  # later use

    '''
    TMC comments response:
        deley_cost_opt
        energy_cost_opt
    '''
    deley_cost_opt = 0
    energy_cost_opt = 0
    '''
    compare experiments:
        max_range offload& original location offloading
    '''
    cost_new_maxrange = 0
    delay_cost_maxrange = 0
    energy_cost_maxrange = 0
    AM_original = genetic_algorithm(Z, p_Mating, p_Mutation, vi, distance, server_num)
    cost_new_original,deley_cost_original,energy_cost_original = offloading_cost_ext(AM_original, vi, distance, server_num,weight3)

    maxMarkov = 10
    for k in range(maxMarkov):
        r_new_maxrange = obfuscated_range(epsilon, max_range)  # cdf -- random select a new range  with cdf
        angle1, angle2 = calculate_angle(user2, server2, max_range)
        user_new_maxrange = obfuscate_location(user2, r_new_maxrange, angle1, angle2)
        distance_new_maxrange = distance_calculation2(user_new_maxrange, server2)
        AM_max_range = genetic_algorithm(Z, p_Mating, p_Mutation, vi, distance_new_maxrange, server_num)
        # print(AM_max_range)
        cost_new_maxrange_temp,delay_cost_maxrange_temp,energy_cost_maxrange_temp = offloading_cost_ext(AM_max_range, vi, distance, server_num,weight3)
        cost_new_maxrange += cost_new_maxrange_temp
        delay_cost_maxrange += delay_cost_maxrange_temp
        energy_cost_maxrange += energy_cost_maxrange_temp

    cost_new_avg_maxrange = cost_new_maxrange / maxMarkov
    delay_new_avg_maxrange = delay_cost_maxrange / maxMarkov
    energy_new_avg_maxrange = energy_cost_maxrange / maxMarkov


    privacyLeakage_new_maxrange = privacy_leakage_comp(epsilon, max_range, angle1, angle2)
    step = 1.5
    while T > T_min:        #       outer circulation
        kBetter = 0                
        kBadAccept = 0            
        kBadRefuse = 0            
        pBadAccept = 0           
        meanMarkov = 500        # preset Inner Circulation

        # U_old = 0       #preset the utility of original location
        # U_new = 0       #preset the utility of obfuscated location

        r_max_txt = []
        cost_new_avg_txt = []
        privacyLeakage_new_txt = []
        U_new_txt = []


        for cycle in range(Cycles):
            time1_single = time.time()
            r_max = r_init + delta
            # preset
            cost_new = 0
            delay_new = 0
            energy_new = 0
            print("old:",U_old)
            print("r_max:",r_max)
            # Records of historical offloading cost
            offloadRecord = {}
            for i in np.arange(0, r_max, step):
                offloadRecord[round(i, 1)] = 0 

            for k in range(meanMarkov):
                r_new = obfuscated_range(epsilon, r_max)        #cdf -- random select a new range  with cdf
                angle1, angle2 = calculate_angle(user2, server2, r_max)
                user_new = obfuscate_location(user2, r_new, angle1, angle2)
                distance_new = distance_calculation2(user_new, server2)

                key = round(int(r_new / step) * step,1)
                if offloadRecord[key] != 0: 
                    cost_new += offloadRecord[key] 
                    continue

                # print("distance_new",distance_new)
                # print("distance_old",distance)
                AM = genetic_algorithm(Z, p_Mating, p_Mutation, vi, distance_new, server_num)

                # print("AM:",AM)
                cost_new_temp,delay_cost_temp,energy_cost_temp = offloading_cost_ext(AM, vi, distance, server_num,weight3)
                offloadRecord[key] =  cost_new_temp
                cost_new += cost_new_temp
                delay_new += delay_cost_temp
                energy_new += energy_cost_temp

            cost_new_avg = cost_new/meanMarkov
            delay_new_avg = delay_new / meanMarkov
            energy_new_avg = energy_new / meanMarkov

            print("cost_new_avg",cost_new_avg)
            r_max_txt.append(r_max)
            cost_new_avg_txt.append(cost_new_avg)
            time2_single = time.time()
            print("single_time_cost:",time2_single-time1_single)


            privacyLeakage_new = privacy_leakage_comp(epsilon, r_max, angle1, angle2)
            privacyLeakage_new_txt.append(privacyLeakage_new)
            U_new = (cost_new_avg + weight3 * privacyLeakage_new)
            U_new_txt.append(U_new)
            deltaU = U_new - U_old
            print("(U_new,U_old)--deltaU--PL:", U_new, U_old, deltaU,weight3 *privacyLeakage_new)
            U_old = U_new

            if deltaU < 0:
                accept = True
                kBetter += 1
                r_init = r_max
            else:
                # Accept the worse value with a certain probability-p
                p = math.exp(-(U_new - U_old) / T)
                r = np.random.uniform(low=0, high=1)
                if r < p:
                    accept = True
                    kBadAccept += 1
                    r_init = r_max
                else:
                    accept = False
                    kBadRefuse += 1

            if U_new < ZATA:
                ZATA = U_new
                r_opt = r_max
                cost_opt = cost_new_avg
                deley_cost_opt = delay_new_avg
                energy_cost_opt = energy_new_avg
                pl_opt = privacyLeakage_new

        if kIter % 1 == 0:  # record every 1 times
            print('i:{},t(i):{:.2f}, badAccept:{:.6f}, f(x)_best:{:.6f}'. \
                  format(kIter, T, pBadAccept, ZATA))
        # cooling
        T = alpha * T
        kIter = kIter + 1

    # r_new_final = obfuscated_range(epsilon, r_opt)  # cdf -- random select a new range  with cdf
    # angle1, angle2 = calculate_angle(user2, server2, r_opt)
    # user_new = obfuscate_location(user2, r_new_final, angle1, angle2)
    # distance_new = distance_calculation2(user_new, server2)
    # AM = genetic_algorithm(Z, p_Mating, p_Mutation, vi, distance_new, server_num)
    # cost_new_final = offloading_cost(AM, vi, distance, server_num)
    return cost_opt,ZATA,r_opt,AM,pl_opt,cost_new_original,deley_cost_original,energy_cost_original,cost_new_avg_maxrange,delay_new_avg_maxrange,energy_new_avg_maxrange,privacyLeakage_new_maxrange,deley_cost_opt,energy_cost_opt


'''
main calculation process
******parameters******
x_k: the situation of task. TASK = [1,2,4,2,...,8]
T: the number of task in user
vi: the size of tasks
user2: user's location
server2: servers's location 
K: number of server
******returns********
ZATA: final cost
'''


def update_range2(x_k,vi,user2,server2,K,max_range,times,weight3,epsilon):
    #initial
    ZATA=0
    L = 100     # iteration times
    T = 100
    T_min = 1
    Cycles = 10
    alpha = 0.2
    delta = max_range /(Cycles*3)
    #distance = []
    distance = distance_calculation2(user2, server2)
    C_min = offloading_cost(x_k, vi, distance,K,weight3)
    x_k.append(C_min)       #Alignment with GA algorithm
    # cost_opt,ZATA,r_max,pop,pl_opt,cost_new_original,cost_new_avg_maxrange,privacyLeakage_new_maxrange = simulated_annealing(L,T,T_min,Cycles,alpha,delta,user2,server2,x_k,distance,K,vi,max_range,weight3,epsilon)
    cost_opt,ZATA,r_opt,AM,pl_opt,cost_new_original,deley_cost_original,energy_cost_original,\
    cost_new_avg_maxrange,delay_new_avg_maxrange,energy_new_avg_maxrange,privacyLeakage_new_maxrange,deley_cost_opt,\
    energy_cost_opt = simulated_annealing_ext(L,T,T_min,Cycles,alpha,delta,user2,server2,x_k,distance,K,vi,max_range,weight3,epsilon)

    # sout to text
    flag = str(K)
    time = str(times)
    # file_server = input_args[0]
    dir1= "./experiment/server_compare"
    file_name = dir1 + "/server_" + flag + "_time_"+time+ "_sameTask.txt"
    if not os.path.exists(dir1):
        os.makedirs(dir1, mode=0o777)
    print(file_name)
    f = open(file_name, 'w+')
    f.write("[cost_opt , pl_opt , ZATA  ,  deley_cost_opt  , energy_cost_opt]\n")
    f.write("[" + str(cost_opt) + "," + str(pl_opt) + "," + str(ZATA) + "," + str(deley_cost_opt) + "," + str(energy_cost_opt) + "]\n")
    f.close()

    dir2= "./experiment/server_compare"
    file_name = dir2 + "/originalserver_" + flag + "_time_"+time+ "_sameTask.txt"
    if not os.path.exists(dir2):
        os.makedirs(dir2, mode=0o777)
    print(file_name)
    f = open(file_name, 'w+')
    f.write("[cost_original , deley_cost_original , energy_cost_original]\n")
    f.write("[" + str(cost_new_original) + "," + str(deley_cost_original) + "," + str(energy_cost_original) +"]\n")
    f.close()

    dir3= "./experiment/server_compare"
    file_name = dir3 + "/maxrangeserver_" + flag + "_time_"+time+ "_sameTask.txt"
    if not os.path.exists(dir3):
        os.makedirs(dir3, mode=0o777)
    print(file_name)
    f = open(file_name, 'w+')
    f.write("[ cost_new_avg_maxrange , delay_new_avg_maxrange , energy_new_avg_maxrange , privacyLeakage_new_maxrange ]\n")
    f.write("[" + str(cost_new_avg_maxrange) + "," + str(delay_new_avg_maxrange) + "," + str(energy_new_avg_maxrange) + "," + str(privacyLeakage_new_maxrange) + "]\n")
    f.close()

    return cost_opt,ZATA

if __name__ == '__main__':
    time1 = time.time()
    #pre-set
    global WEIGHT3,EPSILON
    weight3 = WEIGHT3
    epsilon = EPSILON
    x_k = [0,1,2,2,2,1,2,1,0,0,1,0,1,1]
    # vi =  [1,3,1,3,1,3,2,1,1,2,4,10,4,1]
    vi  = [2,1,2,2,1,3,4,3,4,3,8,2,4,1]
    for index,v in enumerate(vi):
        vi[index] = v * 1024*1024
    T = len(x_k)
    K = int(sys.argv[1])      # largest number of servers; be sure the x_k has the largest number of server
    times = int(sys.argv[2])
    #Initial- Data Assignment
    print("服务器数量：",K)
    max_range = 1000
    # user2  = {'X_INDEX':random.random()*max_range,'Y_INDEX':random.random()*max_range}    #user2 is random selected at [0-10,0-10]
    user2 = {'X_INDEX': 0, 'Y_INDEX': 0}
    server2 = {}  # server2 has multiple server, with random selected location&range

    total_angle = math.pi * 2
    for k in range(K):      #K server
        # server = {'LATITUDE':1000*math.cos(each_angle*k),'LONGITUDE':1000*math.sin(each_angle*k),'RANGE':random.random()*max_range}
        server = {'LATITUDE': 2000 * random.random()-1000, 'LONGITUDE': 2000 * random.random()-1000,
                  'RANGE': random.random() * max_range}
        server2.update({k:server})

    # main calculation process
    cost, ZATA = update_range2(x_k,vi,user2,server2,K,max_range,times,weight3,epsilon)
    time2 = time.time()
    print("total cost:", time2 - time1)

    #sout to text

