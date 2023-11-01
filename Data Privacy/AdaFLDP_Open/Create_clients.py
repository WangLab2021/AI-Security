import pickle
import numpy as np
import os

def create_clients(num, dir):

    '''
    This function creates clients that hold non-iid MNIST data accroding to the experiments in https://research.google.com/pubs/pub44822.html. (it actually just creates indices that point to data.
    but the way these indices are grouped, they create a non-iid client.)
    :param num: Number of clients
    :param dir: where to store
    :return: _
    '''

if __name__ == '__main__':
    List_of_clients = [10,20,50,100,200]
    
    total = np.arange(40000)
    print(total)
    np.random.shuffle(total)
    for j in List_of_clients:
        path = './DATA/clients_cifar10/'+str(j)+'clients'
        arr = np.split(np.asarray(total), j, 0)
        np.save(path,arr)
