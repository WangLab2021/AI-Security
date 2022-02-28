from collections import defaultdict
import torch
import torch.utils.data
from torch.utils.data import DataLoader
from helper import Helper
import copy
import random
import logging
import numpy as np
from read_prop import myDataset
from models.LFW_model import LFWNet
logger = logging.getLogger("logger")
POISONED_PARTICIPANT_POS = 0



class ImageHelper(Helper):
    def poison(self):
        return
    def create_model(self):
        #
        local_model = LFWNet(name='Local',
                    created_time=self.params['current_time'])
        local_model.cuda()
        target_model = LFWNet(name='Target',
                        created_time=self.params['current_time'])
        target_model.cuda()
        shadow_model1 = LFWNet(name='Shadow1',
                        created_time=self.params['current_time'])
        shadow_model1.cuda()

        shadow_model2 = LFWNet(name='Shadow2',
                        created_time=self.params['current_time'])
        shadow_model2.cuda()

        if self.params['resumed_model']:
            loaded_params = torch.load(f"saved_models/{self.params['resumed_model']}")
            target_model.load_state_dict(loaded_params['state_dict'])
            self.start_epoch = loaded_params['epoch']
            self.params['lr'] = loaded_params.get('lr', self.params['lr'])
            logger.info(f"Loaded parameters from saved model: LR is"
                        f" {self.params['lr']} and current epoch is {self.start_epoch}")
        else:
            self.start_epoch = 1

        self.local_model = local_model
        self.target_model = target_model
        self.shadow_model = shadow_model1
        self.shadow_model2 = shadow_model2

    def sample_dirichlet_train_data(self, no_participants, alpha=0.9):
        """
            Input: Number of participants and alpha (param for distribution)
            Output: A list of indices denoting data in CIFAR training set.
            Requires: cifar_classes, a preprocessed class-indice dictionary.
            Sample Method: take a uniformly sampled 10-dimension vector as parameters for
            dirichlet distribution to sample number of images in each class.
        """

        cifar_classes = {}
        for ind, x in enumerate(self.train_dataset):
            _, label = x
            if ind in self.poison_images or ind in self.poison_images_test or \
                    ind in self.poison_images_client:
                continue
            if label in cifar_classes:
                cifar_classes[label].append(ind)
            else:
                cifar_classes[label] = [ind]
        # class_size = len(cifar_classes[0])
        class_size = int(len(cifar_classes[0]) * 0.8)
        per_participant_list = defaultdict(list)
        # no_classes = len(cifar_classes.keys())
        no_classes = [0, 1]
        rest_class = {}
        for n in no_classes:
            rest_class[n] = cifar_classes[n][class_size:]
            cifar_classes[n] = cifar_classes[n][:class_size]

        for n in no_classes:
            random.shuffle(cifar_classes[n])
            sampled_probabilities = class_size * np.random.dirichlet(
                np.array(no_participants * [alpha]))
            for user in range(no_participants):
                no_imgs = int(round(sampled_probabilities[user]))
                sampled_list = cifar_classes[n][:min(len(cifar_classes[n]), no_imgs)]
                per_participant_list[user].extend(sampled_list)
                cifar_classes[n] = cifar_classes[n][min(len(cifar_classes[n]), no_imgs):]
                # logger.info('user: {} has {} length of dataset'.format(user, len(per_participant_list[user])))

        return per_participant_list, rest_class


    def poison_dataset(self):
        # return [(self.train_dataset[self.params['poison_image_id']][0],
        # torch.IntTensor(self.params['poison_label_swap']))]
        indices = copy.deepcopy(self.train_indices)
        #create candidates:
        for image in self.poison_images + self.poison_images_test + self.poison_images_client:
            if image in indices:
                indices.remove(image)
        indices = random.sample(indices, self.params['total_len'])
        return torch.utils.data.DataLoader(self.train_dataset,
                           batch_size=self.params['batch_size'],
                           sampler=torch.utils.data.sampler.SubsetRandomSampler(indices), drop_last=True)

    def poison_test_dataset(self):
        #
        # return [(self.train_dataset[self.params['poison_image_id']][0],
        # torch.IntTensor(self.params['poison_label_swap']))]
        poison_test_indices = random.sample(self.train_indices, 1000)
        return torch.utils.data.DataLoader(self.train_dataset,
                           batch_size=self.params['batch_size'],
                           sampler=torch.utils.data.sampler.SubsetRandomSampler(
                             poison_test_indices
                           ))


    def load_data(self, task='gender', attr='race'):
        logger.info('Loading data')
        X_train = torch.load(f'data\\{attr}\\X_train.pkl')
        y_train = torch.load(f'data\\{attr}\\y_train.pkl')
        X_test = torch.load(f'data\\{attr}\\X_test.pkl')
        y_test = torch.load(f'data\\{attr}\\y_test.pkl')
        X_train = torch.cuda.FloatTensor(X_train)
        y_train = torch.cuda.LongTensor(y_train[:, 0])
        X_test = torch.cuda.FloatTensor(X_test)
        y_test = torch.cuda.LongTensor(y_test[:, 0])
        self.train_dataset = myDataset(X_train, y_train)
        self.test_dataset = myDataset(X_test, y_test)

        cifar_classes = {}
        for ind, x in enumerate(self.train_dataset):
            _, label = x
            if ind in self.poison_images or ind in self.poison_images_test or ind in self.poison_images_client:
                continue
            if label in cifar_classes:
                cifar_classes[label].append(ind)
            else:
                cifar_classes[label] = [ind]

        indices = list()
        for n in cifar_classes.keys():
            indices.extend(cifar_classes[n])
        self.train_indices = indices

        cifar_classes = {}
        for ind, x in enumerate(self.test_dataset):
            _, label = x
            if label in cifar_classes:
                cifar_classes[label].append(ind)
            else:
                cifar_classes[label] = [ind]
        need_class = []
        for label_i in cifar_classes.keys():
            need_class.extend(cifar_classes[label_i])
        self.test_indices = need_class
        print('len test:', len(self.test_indices))
        if self.params['sampling_dirichlet']:
            ## sample indicecs for participants using Dirichlet distribution
            indices_per_participant, self.restclass = self.sample_dirichlet_train_data(
                self.params['number_of_total_participants'],
                alpha=self.params['dirichlet_alpha'])
            train_loaders = [(pos, self.get_train(indices)) for pos, indices in
                             indices_per_participant.items()]

        else:
            ## sample indices for participants that are equally
            self.client_indices = self.train_indices[:500]
            # self.client_indices = self.train_indices
            self.shadow_indices = self.train_indices[1000:1500]
            random.shuffle(self.client_indices)
            train_loaders = [(pos, self.get_train_old(self.client_indices, pos))
                             for pos in range(self.params['number_of_total_participants'])]
        self.train_data = train_loaders
        self.test_data = self.get_test()
        self.poisoned_data_for_train = self.poison_dataset()
        self.test_data_poison = self.poison_test_dataset()

    def get_train(self, indices):
        """
        This method is used along with Dirichlet distribution
        :param params:
        :param indices:
        :return:
        """
        train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                           batch_size=self.params['batch_size'],
                                           sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                               indices))
        return train_loader

    def get_train_old(self, all_range, model_no):
        """
        This method equally splits the dataset.
        :param params:
        :param all_range:
        :param model_no:
        :return:
        """
        data_len = int(len(self.train_indices)*0.8 / self.params['number_of_total_participants'])
        # print('user has {} of data'.format(data_len))
        sub_indices = all_range[model_no * data_len: (model_no + 1) * data_len]
        # logger.info('{} dataset length of user'.format(len(sub_indices)))
        train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                           batch_size=self.params['batch_size'],drop_last=True,
                                           sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                               sub_indices))
        return train_loader

    def get_test(self):
        print(len(self.test_indices))

        test_loader = torch.utils.data.DataLoader(self.test_dataset,
                                                  batch_size=self.params['test_batch_size'],
                                                  sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                      self.test_indices))
        return test_loader
    def get_batch(self, train_data, bptt, evaluation=False):
        data, target = bptt
        data = data.cuda()
        target = target.cuda()
        if evaluation:
            data.requires_grad_(False)
            target.requires_grad_(False)
        return data, target

