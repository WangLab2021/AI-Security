from itertools import permutations
import numpy as np
import copy
import matplotlib
import matplotlib.pyplot as plt
from itertools import combinations, permutations



'''
parameters: matrix_A: 2-demission matrix, only 01; x_ik = 1 means the task i is offloaded to server k
return: list_A : 2-demission list=[[A1,A2,...,Ax]], in which A1[3] = 4 means task 3 is offloaded to server 3
(server 0 means local)
'''


def matrix_translate(matrix_A):
    list_A = [0] * matrix_A.shape[1]
    for i in range(matrix_A.shape[1]):
        x = np.where(matrix_A[:, i] == 1)
        # print(x)
        if len(x[0]) == 0:
            list_A[i] = 0
        else:
            list_A[i] = int(x[0]) + 1
    return list_A

'''
parameters: pop metric
return: Index of the individual with the lowest adaptation, the individual with the highest adaptation
'''


def choiceFuction(pop):
    return np.argmax(pop[:, -1]),pop[np.argmin(pop[:, -1])]


'''
parameters: pop metric, mating probability, number of genes, mutation probability, number of initialized groups
return: Intermediate population status
'''


def crossoverFuntion(pop, p_Mating, server_num, p_Mutation, NUM):
    mating_matrix = np.array(1 - (np.random.rand(NUM) > p_Mating), dtype=bool)  # Mating matrix, if true then mating takes place
    a = list(pop[mating_matrix][:, :-1])  # Mating Individuals 
    a2 = copy.deepcopy(a)
    a2 = [list(i) for i in a2]
    b = list(pop[np.array(1 - mating_matrix, dtype=bool)][:, :-1])  # Individuals that have not been mated 
    b = [list(j) for j in b]
    #     print(a)
    if len(a) % 2 != 0:     #odd number
        b.append(a.pop())
    for i in range(int(len(a) / 2)):        #switch a[m]&a[n]
        p1 = np.random.randint(1, int(pop.shape[1]) - 1 - 1)        # index  = num - cost(1) - (1) ---- 5-1-1
        p2 = np.random.randint(1, int(pop.shape[1]) - 1 - 1)
        if p1>p2:       #switch
            temp = p2
            p2 = p1
            p1 = temp
        x1 = list(a.pop())
        x2 = list(a.pop())
        x1,x2 = mutation(x1, x2, p1, p2)
        # mutation
        x1 = mutationFunction(x1, p_Mutation, server_num)
        x2 = mutationFunction(x2, p_Mutation, server_num)
        # new individuals will be added to the next generation
        b.append(x1)
        b.append(x2)
    new_pop = a2 + b
    ZEROS = np.zeros((len(new_pop), 1))
    temp = np.column_stack((new_pop, ZEROS))       #two row, add a column of fitness; the fitness haven't been calculated, 0 represented;
    return temp



def mutation(x1, x2, p1, p2):
    temp = x1[:p1]
    x1[:p1] = x2[:p1]
    x2[:p1] = temp
    temp = x1[p2:]
    x1[p2:] = x2[p2:]
    x2[p2:] = temp
    return x1,x2

'''
parameters: pop metric, mating probability, number of genes, mutation probability, number of initialized groups
return: Intermediate population status
'''


def mutationFunction(list_a, p_Mutation, server_num):
    #Swap after determining a range
    if np.random.rand() < p_Mutation:
        mutation_num = np.random.randint(1, int(len(list_a))/2)      #must change at least 1 point
        while mutation_num>0:
            point = np.random.randint(0, int(len(list_a)) - 1)      # change point
            # list_a[point] = np.random.randint(0, int(server_num) - 1)           #change value
            list_a[point] = np.random.randint(0, int(server_num))
            mutation_num -=1

    return list_a


