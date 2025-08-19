import copy
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses, metrics
import logging
from random import sample
from datetime import datetime
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from scipy import spatial
import random

# 设置TensorFlow日志级别
tf.get_logger().setLevel('ERROR')
logger = logging.getLogger(__name__)
date = datetime.now().strftime('%Y-%m-%d %H:%M')

class Argument():
    def __init__(self):
        self.user_num = 100       # number of total clients P
        self.K = 20               # number of participant clients K
        self.lr = 0.0005          # learning rate of global model
        self.batch_size = 8       # batch size of each client for local training
        self.itr_test = 100       # number of iterations for the two tests on test datasets
        self.test_batch_size = 128 # batch size for test datasets
        self.total_iterations = 100000  # total number of iterations
        self.seed = 1             # parameter for the server to initialize the model
        self.classNum = 1         # number of data classes on each client
        self.cuda_use = True
        self.frac = 0.1
        self.clustering = 'K-Means'  # DBSCAN, K-Means, Graph
        self.standard = 'Cosine'  # Cosine, entropy
        self.l_epoch = 1          # local epochs
        self.momentum = False
        self.alpha = 0.1
        self.function = 0         # 0: 1/x, 1: (e/2)^(-t)
        self.inter = 1            # 0: average, 1: weight
        self.intergroup = 1       # 0: 按组数据总量分, 1: 按客户端数据量分

args = Argument()

# 设置随机种子 - 更严格的控制
tf.random.set_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# 设置数值精度
tf.keras.backend.set_floatx('float32')

# 检查GPU可用性
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    # 设置确定性操作
    tf.config.experimental.enable_op_determinism()
    print("GPU is available")
else:
    print("GPU not available, using CPU")

class Net(tf.keras.Model):
    """CNN模型定义，完全对应原PyTorch的Net类"""
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = layers.Conv2D(6, 5)
        self.conv2 = layers.Conv2D(16, 5)  
        self.fc1 = layers.Dense(120)
        self.fc2 = layers.Dense(84)
        self.fc3 = layers.Dense(10)
        
    def call(self, x):
        # 对应 F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = tf.nn.max_pool2d(tf.nn.relu(self.conv1(x)), (2, 2), strides=2, padding='VALID')
        # 对应 F.max_pool2d(F.relu(self.conv2(x)),2)
        x = tf.nn.max_pool2d(tf.nn.relu(self.conv2(x)), 2, strides=2, padding='VALID')
        # 对应 x.view(x.size()[0],-1)
        x = tf.reshape(x, [tf.shape(x)[0], -1])
        # 对应 F.relu(self.fc1(x))
        x = tf.nn.relu(self.fc1(x))
        # 对应 F.relu(self.fc2(x))
        x = tf.nn.relu(self.fc2(x))
        # 对应 self.fc3(x)
        x = self.fc3(x)
        return x

def GetModelLayers(model):
    """获取模型层数和各层的形状，对应原函数"""
    Layers_shape = []
    Layers_nodes = []
    Layers_num = 0
    
    for var in model.trainable_variables:
        Layers_shape.append(var.shape)
        Layers_nodes.append(tf.size(var).numpy())
        Layers_num += 1
    
    return Layers_num, Layers_shape, Layers_nodes

def ZerosGradients(Layers_shape):
    """设置各层的梯度为0，对应原函数"""
    ZeroGradient = []
    for i in range(len(Layers_shape)):
        ZeroGradient.append(tf.zeros(Layers_shape[i]))
    return ZeroGradient

def L_norm(Tensor):
    """计算范数，对应原函数"""
    norm_Tensor = tf.constant(0.)
    for i in range(len(Tensor)):
        norm_Tensor += tf.norm(tf.cast(Tensor[i], tf.float32))**2
    return tf.sqrt(norm_Tensor)

def TensorClip(Tensor, ClipBound):
    """定义剪裁，对应原函数"""
    norm_Tensor = L_norm(Tensor)
    if ClipBound < norm_Tensor:
        Layers_num = len(Tensor)
        for i in range(Layers_num):
            Tensor[i] = Tensor[i] * ClipBound / norm_Tensor
    return Tensor

def test(model, test_data, test_labels):
    """测试函数，对应原函数"""
    # 确保数据类型一致
    test_data = tf.cast(test_data, tf.float32)
    test_labels = tf.cast(test_labels, tf.int64)
    
    # 创建测试数据集
    test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_labels))
    test_dataset = test_dataset.batch(args.test_batch_size)
    
    correct = 0
    total = 0
    test_loss = 0.0
    
    for batch_data, batch_labels in test_dataset:
        outputs = model(batch_data, training=False)
        predicted = tf.argmax(outputs, axis=1)
        # 确保predicted和batch_labels类型一致
        predicted = tf.cast(predicted, tf.int64)
        batch_labels = tf.cast(batch_labels, tf.int64)
        
        # 转换为Python数值进行累加
        total += int(tf.shape(batch_labels)[0])
        correct += int(tf.reduce_sum(tf.cast(predicted == batch_labels, tf.int32)))
        
        # 计算损失
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=batch_labels, logits=outputs)
        test_loss += float(tf.reduce_mean(loss))
    
    test_acc = 100. * correct / total
    print('10000张测试集: testacc is {:.4f}%, testloss is {}.'.format(test_acc, test_loss))
    return test_loss, test_acc

def copy_model(original_model):
    """创建模型副本的函数，替代copy.deepcopy"""
    new_model = Net()
    # 使用相同的输入形状初始化新模型
    dummy_input = tf.random.normal((1, 32, 32, 3))
    _ = new_model(dummy_input)
    
    # 复制权重
    for new_var, orig_var in zip(new_model.trainable_variables, original_model.trainable_variables):
        new_var.assign(orig_var)
    
    return new_model

def train(learning_rate, model, train_data, train_targets, optimizer):
    """训练过程，返回梯度，对应原函数"""
    # 确保数据类型一致
    train_data = tf.cast(train_data, tf.float32)
    train_targets = tf.cast(train_targets, tf.int64)
    
    with tf.GradientTape() as tape:
        outputs = model(train_data, training=True)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=train_targets, logits=outputs)
        loss = tf.reduce_mean(loss)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    Gradients_Tensor = []
    for grad in gradients:
        if grad is not None:
            Gradients_Tensor.append(grad)
        else:
            # 如果梯度为None，添加零梯度
            Gradients_Tensor.append(tf.zeros_like(model.trainable_variables[len(Gradients_Tensor)]))
    
    return Gradients_Tensor, loss

def JaDis(datasNum, userNum):
    """计算Non-IID程度，对应原函数"""
    sim = []
    for i in range(userNum):
        data1 = datasNum[i]
        for j in range(i+1, userNum):
            data2 = datasNum[j]
            sameNum = [min(x, y) for x, y in zip(data1, data2)]
            sim.append(sum(sameNum) / (sum(data1) + sum(data2) - sum(sameNum)))
    distance = 2*sum(sim)/(userNum*(userNum-1))
    return distance

# TensorFlow版本的数据加载器
class TensorFlowDataLoader:
    def __init__(self):
        # 加载CIFAR10数据，对应原代码的数据加载
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        
        # 数据预处理，对应原代码的transform
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        # 归一化，对应 transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
        x_train = (x_train - 0.5) / 0.5
        x_test = (x_test - 0.5) / 0.5

        # 确保标签是正确的数据类型并展平
        y_train = y_train.astype('int64').flatten()
        y_test = y_test.astype('int64').flatten()
        
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def dataset_federate_noniid(self, user_num, classNum):
        """创建联邦学习数据分布，对应原cifar10_dataloader.dataset_federate_noniid函数"""
        federated_data = {}
        dataNum = []
        user_tag = []
        
        # 按类别组织数据，对应原代码的dataset字典
        dataset = {}
        for i in range(10):
            indices = np.where(self.y_train == i)[0]
            dataset[str(i)] = self.x_train[indices]  # 只存储数据，不存储标签
        
        for i in range(user_num):
            user_data = []
            user_label = []
            
            # 对应原代码：labelClass = torch.randperm(10)[0:classNum]
            labelClass = np.random.permutation(10)[:classNum]
            # 对应原代码：dataRate = torch.rand([classNum])
            dataRate = np.random.rand(classNum)
            dataRate = dataRate / np.sum(dataRate)
            # 对应原代码：dataNum = torch.randperm(40)[0] + 500
            dataNum_base = np.random.permutation(40)[0] + 500
            # 对应原代码：dataNum = torch.round(dataNum * dataRate)
            dataNum_per_class = np.round(dataNum_base * dataRate).astype(int)
            
            if classNum > 1:
                # 对应原代码：datasnum = torch.zeros([10])
                datasnum = np.zeros(10)
                datasnum[labelClass] = dataNum_per_class
                dataNum.append(datasnum)
                
                for j in range(classNum):
                    datanum = int(dataNum_per_class[j])
                    if datanum > 0:
                        # 对应原代码：index = torch.randperm(5000)[0:datanum]
                        available_data = len(dataset[str(labelClass[j])])
                        if available_data > 0:
                            indices = np.random.permutation(available_data)[:min(datanum, available_data)]
                            user_data.append(dataset[str(labelClass[j])][indices])
                            user_label.append(np.full(len(indices), labelClass[j]))
                
                if user_data:
                    # 对应原代码：user_data = torch.cat(user_data, 0)
                    user_data = np.vstack(user_data)
                    # 对应原代码：user_label = torch.cat(user_label, 0)
                    user_label = np.hstack(user_label)
                else:
                    # 如果没有数据，给一个默认的小数据集
                    user_data = self.x_train[:10]
                    user_label = self.y_train[:10]
            else:
                j = 0
                # 对应原代码：datasnum = torch.zeros([10])
                datasnum = np.zeros(10)
                datasnum[labelClass[j]] = dataNum_per_class[j]
                dataNum.append(datasnum)
                
                datanum = int(dataNum_per_class[j])
                if datanum > 0:
                    # 对应原代码：index = torch.randperm(5000)[0:datanum]
                    available_data = len(dataset[str(labelClass[j])])
                    if available_data > 0:
                        indices = np.random.permutation(available_data)[:min(datanum, available_data)]
                        user_data = dataset[str(labelClass[j])][indices]
                        user_label = np.full(len(indices), labelClass[j])
                    else:
                        user_data = self.x_train[:10]
                        user_label = self.y_train[:10]
                else:
                    user_data = self.x_train[:10]
                    user_label = self.y_train[:10]
                
                # 对应原代码：user_tag.append(labelClass[j].tolist())
                user_tag.append(int(labelClass[j]))
            
            federated_data[f'user{i}'] = {
                'x': user_data,
                'y': user_label
            }
        
        return federated_data, dataNum, user_tag

def set_client_ability():
    """设置客户端能力，对应原函数"""
    user_t = [0.49336529344218094, 0.7419722977437847, 0.7435860892179768, 0.7645615919317725, 0.5104520761056377, 0.5572634855316788, 0.321596027726486,
              0.618696925358594, 0.5322114862629864, 0.3424596074044717, 0.534755099277278, 0.07246607100179137, 0.9676770500453077, 0.7754494970280262,
              0.8951835498681103, 0.00678893311117601, 0.9090264434088875, 0.29131768030324234, 0.3758819537950038, 0.11533938812900713, 0.8106254736717647,
              0.38625399738401056, 0.4389741942049826, 0.4849635415283956, 0.5190461004230923, 0.032433833547466095, 0.3258454199732823, 0.06794295143348383,
              0.9874823933391431, 0.9982671343826565, 0.615447935114033, 0.8012078340348346, 0.04047292434658534, 0.8095490891191124, 0.4759057446151107,
              0.8684312873670791, 0.7060307221250842, 0.8081360122199542, 0.24966861913507776, 0.30094707232629103, 0.5501437556711757, 0.4415320855619229,
              0.3766698955786826, 0.2763616001152166, 0.38989170980582344, 0.14634427462949273, 0.38768776175636044, 0.31904706771056957, 0.31151483841753635,
              0.02718520051036588, 0.8016236385848411, 0.46928267160194226, 0.32726859522270346, 0.12560395509618383, 0.6534519377767415, 0.9167650926340386,
              0.7526938139111854, 0.3827226044851466, 0.170978114262332, 0.6806789141197864, 0.7158381830953974, 0.10801202171278179, 0.5204183871338607,
              0.14064795005539343, 0.4481328621939724, 0.2124923697421094, 0.8510022066015445, 0.4783136403058307, 0.7278364244828298, 0.15439781317830326,
              0.037363217268408855, 0.8888596017984666, 0.42302759152123215, 0.5890794629092465, 0.4030387035158527, 0.4764411303512921, 0.7701671383552889,
              0.3149626070625875, 0.5063648331280326, 0.6983948789498797, 0.7433698453366614, 0.538812302684744, 0.20399133271254954, 0.15647473971930803,
              0.6559853605629543, 0.9050497216909228, 0.9064762069788875, 0.4882350497787674, 0.5352643655737026, 0.8422296596234986, 0.8818893232366467,
              0.931997783966021, 0.9873582802867131, 0.06645011467307105, 0.4159688304342325, 0.9367985016243104, 0.03190583535981506, 0.3609720639234858,
              0.39171518697290486, 0.7170063129900919]
    return user_t

def flatten(items, result=None):
    """展平函数，对应原clustering.py，修正result参数问题"""
    if result is None:
        result = []
    for item in items:
        if isinstance(item, list):
            flatten(item, result)
        else:
            result.append(item)

def do_flatten(gradients_c):
    """梯度展平函数，完全对应原clustering.py的do_flatten函数"""
    gradients = []
    for items in gradients_c:
        g_item = []
        for i in items:
            # 对应原代码：now = i.tolist()
            now = i.numpy().tolist()
            result = []
            flatten(now, result)
            for g in result:
                g_item.append(g)
        gradients.append(g_item)
    return gradients

def reclustering_KMeans(gradients_c):
    """K-Means聚类，完全对应原clustering.py的reclustering_KMeans函数"""
    f = open("./test0403.txt", "a")
    tag = []
    cluster = []
    # 对应原代码：gradients = do_flatten(gradients_c)
    gradients = do_flatten(gradients_c)
    # 对应原代码：clustering = KMeans(n_clusters=20, random_state=9).fit_predict(gradients)
    clustering = KMeans(n_clusters=20, random_state=9).fit_predict(gradients)
    # 对应原代码：clus = clustering
    clus = clustering
    print("clustering:",clus)

    # 对应原代码：num_cluster = max(clus) + 1
    num_cluster = max(clus) + 1
    for i in range(num_cluster):
        tag.append(0)
        cluster.append([])
    for idx in range(len(clus)):
        cluster[clus[idx]].append(idx)

    f.close()  # 添加关闭文件
    return tag, cluster, num_cluster

def reclustering(args, gradients_c):
    """重新聚类，完全对应原clustering.py的reclustering函数"""
    if args.clustering == 'K-Means':
        tag_normal, cluster, num_cluster = reclustering_KMeans(gradients_c)
    else:
        tag_normal, cluster, num_cluster = reclustering_KMeans(gradients_c)
    
    f = open("./test0411.txt", "a")
    print("cluster:", cluster)
    f.write("Total "+ str(num_cluster)+" cluster:"+str(cluster) + '\n')
    f.close()  # 添加关闭文件
    
    return tag_normal, cluster, num_cluster

def main():
    """主函数，完全对应原main.py的逻辑"""
    
    # 创建模型，对应原代码
    model = Net()
    
    # 创建客户端模型和优化器，对应原代码
    models = {}
    optims = {}
    taus = {}
    d_workers = {}
    full = []
    
    for i in range(args.user_num):
        # 完全对应原代码：models["user{}"] = model.copy()
        models[f"user{i}"] = copy_model(model)  # 从服务器模型复制，而不是重新初始化
        optims[f"user{i}"] = optimizers.SGD(learning_rate=args.lr)
        taus[f"user{i}"] = 1
        d_workers[f'user{i}'] = i
        full.append(f'user{i}')
    
    # 设置客户端能力
    user_t = set_client_ability()
    user_now = user_t.copy()
    
    # 服务器优化器，对应原代码：optim_sever = optim.SGD(params=model.parameters(), lr=args.lr)
    optim_server = optimizers.SGD(learning_rate=args.lr)
    
    # 数据载入，完全对应原代码
    data_loader = TensorFlowDataLoader()
    # 在数据分布生成前重新设置种子，确保与PyTorch版本一致
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    federated_data, dataNum, user_tag = data_loader.dataset_federate_noniid(args.user_num, args.classNum)
    Jaccard = JaDis(dataNum, args.user_num)
    print('Jaccard distance is {}'.format(Jaccard))
    
    # 测试数据
    x_test, y_test = data_loader.x_test, data_loader.y_test
    
    # 定义记录字典，完全对应原代码
    logs = {'train_loss': [], 'test_loss': [], 'test_acc': []}
    test_loss, test_acc = test(model, x_test, y_test)
    # 完全对应原代码：logs['test_acc'].append(test_acc.item())
    # 注意：PyTorch版本在初始测试时使用了.item()，但在后续测试中没有使用
    logs['test_acc'].append(float(test_acc))
    logs['test_loss'].append(test_loss)
    # 对应原代码的文件名：f = open("./jiannoweight+noni+data1+K20.txt", "a")
    f = open("./tensorflow_results.txt", "a")
    
    # 获取模型层数和各层形状
    Layers_num, Layers_shape, Layers_nodes = GetModelLayers(model)
    # 对应原代码：e = torch.exp(torch.tensor(1.))
    e = np.e
    
    v = ZerosGradients(Layers_shape)
    
    # 初始化聚类变量
    cluster = None
    num_cluster = 0
    
    # 联邦学习主循环，完全对应原代码逻辑
    for itr in range(1, args.total_iterations + 1):
        print('total itr:',itr)
        
        # 初始化训练损失，完全对应原代码
        Loss_train = tf.constant(0.)
        
        # 对应原代码的 if itr == 1: 逻辑，只在第一轮进行聚类
        if itr == 1:
            gradient_c = []
            for i in range(args.user_num):
                gradient_c.append(ZerosGradients(Layers_shape))
            
            # 模拟原代码的federated_train_loader逻辑，对所有客户端进行训练收集梯度
            for i in range(args.user_num):
                user_id = f'user{i}'
                model_round = models[user_id]
                optimizer = optims[user_id]
                client_idx = d_workers[user_id]
                
                # 获取客户端数据
                client_data = federated_data[user_id]
                train_data = tf.convert_to_tensor(client_data['x'][:args.batch_size])
                train_targets = tf.convert_to_tensor(client_data['y'][:args.batch_size])
                
                # 训练并获取梯度
                Gradients_Sample, loss = train(args.lr, model_round, train_data, train_targets, optimizer)
                for j in range(Layers_num):
                    gradient_c[client_idx][j] += Gradients_Sample[j]
            
            # 聚类
            tag_normal, cluster, num_cluster = reclustering(args, gradient_c)
            
            # 输出聚类标签信息
            tag_cluster = []
            for idx_cluster in range(num_cluster):
                tag_now = []
                for client in cluster[idx_cluster]:
                    tag_now.append(user_tag[client])
                tag_cluster.append(tag_now)
            print("tag:", tag_cluster)
            f.write("tag: " + str(tag_cluster) + ' \n')
        
        # 组内异步聚合，完全对应原代码逻辑
        All_Gradients = ZerosGradients(Layers_shape)
        G_tau = []
        # 对应原代码：all = 0
        all_participants = 0
        for idx_cluster in range(num_cluster):
            all_participants += max(int(args.frac * len(cluster[idx_cluster])), 1)
        
        worker_m = []
        # 用于跟踪最后一个聚类组的训练批次数，对应PyTorch版本的idx_outer
        last_idx_outer = 0
        
        for idx_cluster in range(num_cluster):
            cluster_model = copy_model(model)
            m = max(int(args.frac * len(cluster[idx_cluster])), 1)
            cluster_user_now = []
            for client in cluster[idx_cluster]:
                cluster_user_now.append(user_now[client])
            now = np.sort(cluster_user_now)
            
            workers_list_idx = []
            for client in cluster[idx_cluster]:
                if user_now[client] <= now[m - 1]:
                    workers_list_idx.append(client)
            
            for client in cluster[idx_cluster]:
                user_now[client] -= now[m - 1]
            for idx in workers_list_idx:
                user_now[idx] = user_t[idx]
            
            workers_list = []
            for idx in workers_list_idx:
                workers_list.append('user' + str(idx))
                worker_m.append('user' + str(idx))
                gradient_c[idx] = ZerosGradients(Layers_shape)
            
            f.write(str(idx_cluster)+": "+str(workers_list)+str(cluster[idx_cluster])+'\n')
            
            # 完全对应原代码：Loss_train = torch.tensor(0.) - 在每个聚类组内重新初始化
            cluster_loss_train = tf.constant(0.)
            K_tau = []
            Collect_Gradients = ZerosGradients(Layers_shape)
            
            # 对每个选中的客户端进行训练，完全对应原代码的 federated_train_loader 循环
            for idx_outer, worker_id in enumerate(workers_list):
                model_round = models[worker_id]
                optimizer = optims[worker_id]
                client_idx = d_workers[worker_id]
                user_tau = taus[worker_id]
                K_tau.append(user_tau)
                
                # 获取客户端数据，完全模拟FederatedDataLoader的采样行为
                client_data = federated_data[worker_id]
                # 随机选择batch_size个样本，对应原代码的shuffle=True
                if len(client_data['x']) >= args.batch_size:
                    data_indices = np.random.choice(len(client_data['x']), args.batch_size, replace=False)
                else:
                    # 如果数据不足batch_size，则全部使用
                    data_indices = np.arange(len(client_data['x']))
                train_data = tf.convert_to_tensor(client_data['x'][data_indices])
                train_targets = tf.convert_to_tensor(client_data['y'][data_indices])
                
                # 训练并获取梯度，完全对应原代码
                Gradients_Sample, loss = train(args.lr, model_round, train_data, train_targets, optimizer)
                cluster_loss_train += loss
                
                if args.function == 0:
                    weight = 1 / user_tau
                else:
                    if args.function == 1:
                        weight = (e / 2) ** (-user_tau)
                
                for j in range(Layers_num):
                    Collect_Gradients[j] += Gradients_Sample[j] * args.lr * weight / m
                for j in range(Layers_num):
                    gradient_c[client_idx][j] += Gradients_Sample[j]
            
            # 记录最后一个聚类组的idx_outer，对应PyTorch版本的逻辑
            last_idx_outer = idx_outer if workers_list else 0
            
            G_tau.append(min(K_tau) if K_tau else 1)
            if args.function == 0:
                weight = 1 / min(K_tau) if K_tau else 1
            else:
                if args.function == 1:
                    weight = (e / 2) ** (-min(K_tau)) if K_tau else 1
            
            if args.inter == 0:
                weight = 1
            if args.intergroup == 1:
                weight_data = len(workers_list_idx) / all_participants
            else:
                weight_data = len(cluster[idx_cluster]) / args.user_num
            
            if args.momentum == True:
                for j in range(Layers_num):
                    All_Gradients[j] += Collect_Gradients[j] * weight * weight_data
                if itr == 1:
                    v = [grad.numpy().copy() for grad in All_Gradients]
                else:
                    for j in range(Layers_num):
                        v[j] = v[j] * args.alpha + All_Gradients[j].numpy() * (1 - args.alpha)
                # 转换回tensorflow tensor，完全对应原代码的copy.deepcopy(v)
                All_Gradients = [tf.convert_to_tensor(grad) for grad in v]
            else:
                for j in range(Layers_num):
                    All_Gradients[j] += Collect_Gradients[j] * len(cluster[idx_cluster]) * (weight / args.user_num)
            
            # 更新延时信息，对应原代码
            for i in cluster[idx_cluster]:
                taus['user'+str(i)] += 1
            for worker in workers_list:
                taus[worker] = 1
            
            # 累积各个聚类组的损失
            Loss_train += cluster_loss_train
            
            # 更新集群模型，完全对应原代码的cluster_model更新
            # for grad_idx, params_sever in enumerate(cluster_model.parameters()):
            #     params_sever.data.add_(-1., Collect_Gradients[grad_idx])
            for grad_idx, params_server in enumerate(cluster_model.trainable_variables):
                new_value = params_server.numpy() + (-1.0) * Collect_Gradients[grad_idx].numpy()
                params_server.assign(new_value)
        
        # 组间同步聚合 - 更新全局模型，完全对应原代码的params_sever.data.add_(-1., All_Gradients[grad_idx])
        for grad_idx, params_server in enumerate(model.trainable_variables):
            # 对应PyTorch的 params_sever.data.add_(-1., All_Gradients[grad_idx])
            new_value = params_server.numpy() + (-1.0) * All_Gradients[grad_idx].numpy()
            params_server.assign(new_value)
        
        # 同步客户端模型，完全对应原代码的模型同步逻辑
        for worker_idx in range(len(worker_m)):
            worker_model = models[worker_m[worker_idx]]
            for idx, (params_server, params_client) in enumerate(zip(model.trainable_variables, worker_model.trainable_variables)):
                # 对应PyTorch的 params_client.data = params_server.data
                params_client.assign(params_server)
        
        # 定期测试
        if itr == 1 or itr % args.itr_test == 0:
            print('itr: {}'.format(itr))
            test_loss, test_acc = test(model, x_test, y_test)
            # 完全对应原代码：logs['test_acc'].append(test_acc)
            logs['test_acc'].append(test_acc)  # 不使用float()转换，保持原始类型
            logs['test_loss'].append(test_loss)
            f.write(str(test_acc)+'\n\n\n')
        
        if itr == 1 or itr % args.itr_test == 0:
            # 完全对应原代码的训练损失计算：Loss_train /= (idx_outer + 1)
            Loss_train /= (last_idx_outer + 1)
            logs['train_loss'].append(float(Loss_train))
    
    f.close()
    
    # 保存结果，完全对应原代码的格式
    if not os.path.exists('./results'):
        os.makedirs('./results')
    
    with open('./results/cifar10_TensorFlow_testacc.txt', 'a+') as fl:
        fl.write('\n' + date + ' TensorFlow Results (UN is {}, K is {}, classnum is {}, BZ is {}, LR is {}, itr_test is {}, total itr is {})\n'.
                 format(args.user_num, args.K, args.classNum, args.batch_size, args.lr, args.itr_test, args.total_iterations))
        fl.write('TensorFlow: ' + str(logs['test_acc']))

    with open('./results/cifar10_TensorFlow_trainloss.txt', 'a+') as fl:
        fl.write('\n' + date + ' TensorFlow Results (UN is {}, K is {}, BZ is {}, LR is {}, itr_test is {}, total itr is {})\n'.
                 format(args.user_num, args.K, args.batch_size, args.lr, args.itr_test, args.total_iterations))
        fl.write('train_loss: ' + str(logs['train_loss']))

    with open('./results/cifar10_TensorFlow_testloss.txt', 'a+') as fl:
        fl.write('\n' + date + ' TensorFlow Results (UN is {}, K is {}, BZ is {}, LR is {}, itr_test is {}, total itr is {})\n'.
                 format(args.user_num, args.K, args.batch_size, args.lr, args.itr_test, args.total_iterations))
        fl.write('test_loss: ' + str(logs['test_loss']))

if __name__ == "__main__":
    main() 