from scipy import spatial
from scipy.spatial.distance import pdist
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
import numpy as np


dmax = 0.00
cdis = 2.0



def flatten(items, result=[]):
    for item in items:
        if isinstance(item, list):
            flatten(item, result)
        else:
            result.append(item)


def do_flatten(gradients_c):
    gradients = []
    for items in gradients_c:
        g_item = []
        for i in items:
            #print("item:",i)
            now = i.tolist()
            result = []
            flatten(now, result)
            for g in result:
                g_item.append(g)
        #print("g_item:",g_item)
        gradients.append(g_item)
    '''for i in range(len(gradients)):
        result = []
        flatten(gradients[i], result)
        gradients[i] = result'''
    return gradients


g = []
def find(x,father):
    if father[x] != x:
        father[x]=find(father[x],father)
    return father[x]


def get_entropy(args,models, dict_common):
    entropy = []
    for model in models:
        # Compute the loss and entropy for each device on public dataset
        common_acc, common_loss_sync, common_entropy_sample = test_entropy(args, model,
                                                                        DatasetSplit(train_dataset,
                                                                        dict_common))
        entropy.append(common_entropy_sample)
    return entropy


def reclustering_KMeans(gradients_c):
    f = open("./test0403.txt", "a")
    tag = []
    cluster = []
    #print("gradients:",gradients_c)
    gradients = do_flatten(gradients_c)
    #f.write(str(gradients))
    #print(gradients)
    '''for i in gradients:
        f.write(str(len(i))+"\n")'''
    clustering = KMeans(n_clusters=20, random_state=9).fit_predict(gradients)
    #clus = clustering.labels_.tolist()
    clus = clustering
    print("clustering:",clus)

    num_cluster = max(clus) + 1
    for i in range(num_cluster):
        tag.append(0)
        cluster.append([])
    for idx in range(len(clus)):
        cluster[clus[idx]].append(idx)

    return tag, cluster, num_cluster


def reclustering_DBSCAN(gradients_c):
    tag = []
    cluster = []
    #print("gradients:",gradients_c)
    gradients = do_flatten(gradients_c)
    clustering = DBSCAN(eps=cdis, min_samples=1).fit(gradients)
    clus = clustering.labels_.tolist()
    print("clustering:",clus)

    num_cluster = max(clus) + 1
    for i in range(num_cluster):
        tag.append(0)
        cluster.append([])
    for idx in range(len(clus)):
        cluster[clus[idx]].append(idx)

    return tag, cluster, num_cluster


def reclustering_kruscal(args,gradients_c,tag, model_c, dict_common):
    tag = []
    cluster = []
    #print("gradients:",gradients_c)
    gradients = do_flatten(gradients_c)
    father = [_ for _ in range(args.num_users)]
    g = []
    ddd = []
    for i in range(args.num_users-1):
        j = i+1
        while j<args.num_users:
            g.append(tuple((i,j,spatial.distance.cosine(gradients[i],gradients[j]))))
            ddd.append(spatial.distance.cosine(gradients[i],gradients[j]))
            j+=1
    ddd.sort()
    ff = open("./dis.txt", "a")
    entropy = get_entropy(args,model_c, dict_common)
    idx = []
    for i in range(len(entropy)):
        idx.append(i)
    ff.write('Entropy:' + '\n' + str(entropy) + '\n' + str(idx) + '\n')
    ff.write('Cosine:'+'\n'+str(ddd)+'\n')

    num_cluster = args.num_users
    for u,v,w in g:
        if w>cdis :
            continue
        fu=find(u,father)
        fv=find(v,father)
        if fu!=fv:
            father[fv]=fu
            num_cluster-=1

    clustering = []
    for i in range(args.num_users):
        clustering.append(find(i,father))
    print("clustering:",clustering)

    for i in range(num_cluster):
        tag.append(0)
    for idx in range(max(clustering) + 1):
        cluster.append([])
    for idx in range(args.num_users):
        cluster[clustering[idx]].append(idx)
    dellist = []
    for idx in range(max(clustering) + 1):
        if len(cluster[idx]) == 0:
            dellist.append(idx)
    print("dellist:",dellist)
    t=0
    for idx in dellist:
        del cluster[idx-t]
        t+=1
    return tag, cluster, num_cluster


def similarity(gradients_c):
    gradients = do_flatten(gradients_c)
    g_mean = np.mean(gradients, axis=0, keepdims=True)
    dis = 0.000000
    for item in gradients :
        dis = dis + spatial.distance.cosine(item, g_mean)
    return dis


def similarity_test(gradients_c):
    gradients = do_flatten(gradients_c)
    g_mean = np.mean(gradients, axis=0, keepdims=True)
    dis = 0.000000
    for item in gradients :
        dis = dis + spatial.distance.cosine(item, g_mean)
    if dis > dmax * len(gradients):
        return 1
    return 0



def reclustering(args,gradients_c):
    if args.clustering == 'K-Means':
        tag_normal, cluster, num_cluster = reclustering_KMeans(gradients_c)
    else:
        if args.clustering == 'DBSCAN':
            tag_normal, cluster, num_cluster = reclustering_DBSCAN(gradients_c)
        else :
            if args.clustering == 'graph':
                tag_normal, cluster, num_cluster = reclustering_kruscal(args, gradients_c, tag, model_c, dict_common)

    f = open("./test0411.txt", "a")
    print("cluster:", cluster)
    f.write("Total "+ str(num_cluster)+" cluster:"+str(cluster) + '\n')
    dmax = 0
    for clu in cluster:
        g_clu = []
        for client in clu:
            g_clu.append(gradients_c[client])
        dmax = max(similarity(g_clu) / len(clu), dmax)
    f.write("dmax:" + str(dmax) + '\n')

    f.close()

    return tag_normal, cluster, num_cluster
