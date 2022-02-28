import os

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset, TensorDataset
import copy

import logging
logger = logging.getLogger("logger")

import random

def shadow_data(helper):
    len_sdata = helper.params['total_len']
    shadow_indices = copy.deepcopy(helper.shadow_indices)
    nonp_shadow_indices = random.sample(shadow_indices, len_sdata)
    random.shuffle(nonp_shadow_indices)

    train_loader = DataLoader(dataset=helper.train_dataset, batch_size=helper.params['batch_size'],
                              sampler=torch.utils.data.sampler.SubsetRandomSampler(nonp_shadow_indices), drop_last=True)

    shadow_dataset = train_loader
    shadow_dataset_nonp = copy.deepcopy(shadow_dataset)

    set1 = random.sample(helper.poison_images_test, helper.params['shadow_p_len'])
    set2 = random.sample(shadow_indices, len_sdata-helper.params['shadow_p_len'])
    p_shadow_indices=set1
    p_shadow_indices.extend(set2)
    random.shuffle(p_shadow_indices)
    train_loader_p= DataLoader(dataset=helper.train_dataset, batch_size=helper.params['batch_size'],
                                sampler=torch.utils.data.sampler.SubsetRandomSampler(p_shadow_indices), drop_last=True)
    shadow_dataset_p = train_loader_p

    return shadow_dataset_p, shadow_dataset_nonp

def train_shadow(helper, shadow_model, shadow_model_id, shadow_data, epoch, poison_epoch):
    shadow_model.train()

    optimizer = torch.optim.SGD(shadow_model.parameters(), lr=helper.params['lr'],
                                momentum=helper.params['momentum'],
                                weight_decay=helper.params['decay'])
    poison_perbatch = random.sample(helper.poison_images, helper.params['shadow_poi'])
    for s_epoch in range(helper.params['shadow_epoch']):
        for batch_id, batch in enumerate(shadow_data):
            optimizer.zero_grad()
            if epoch in poison_epoch and helper.params['is_poison']:
                print('***poison shadow model***')
                for pos, image in enumerate(poison_perbatch):
                    poison_pos = min(len(batch[0])-1, pos)
                    # random.randint(0, len(batch))
                    batch[0][poison_pos] = helper.train_dataset[image][0]
                    batch[0][poison_pos].add_(
                        torch.cuda.FloatTensor(batch[0][poison_pos].shape).normal_(0, helper.params['noise_level']))
                    # batch[1][poison_pos] = helper.params['poison_label_swap']
                    true_label = batch[1][poison_pos]
                    batch[1][poison_pos] = torch.abs((true_label - 1))

            data, target = helper.get_batch(shadow_data, batch, evaluation=False)
            out = shadow_model(data)
            loss = nn.functional.cross_entropy(out, target)
            loss.backward()
            optimizer.step()

    model_dict = copy.deepcopy(shadow_model.state_dict())
    # torch.save(model_dict, 'saved_params/shadow_model_{}_epoch{}.pkl'.format(shadow_model_id, epoch))
    return model_dict


def save_shadow_model(model_dict, model_id, with_p):
    if with_p:
        torch.save(model_dict, 'saved_params/shadow_p_{}.pkl'.format(model_id))
    else:
        torch.save(model_dict, 'saved_params/shadow_nonp_{}.pkl'.format(model_id))

def shadow_update(helper, epoch, target_dict, trained_dict,  exp_time, model_id, with_p):
    update = {}
    for name, params in target_dict.items():
        if helper.params.get('tied', False) and name == 'decoder.weight' or '__' in name or 'running' in name or 'num_batches_tracked' in name:
            continue
        update[name] = trained_dict[name] - params
    root = helper.params['root_path']
    folder_path = f'{root}/exp{exp_time}/train'
    try:
        os.mkdir(folder_path)
    except FileExistsError:
        print('Folder already exists')

    each_update = []
    for key, tensor_val in update.items():
        array_layer = tensor_val.cpu().numpy()
        array_layer = np.mean(array_layer, axis=0).flatten()
        each_update.extend(array_layer)
    each_update = torch.cuda.FloatTensor(np.array(each_update).reshape(1, -1))
    if with_p:
        torch.save(each_update, '{}/S{}_p_{}_epoch.pkl'.format(folder_path, model_id, epoch))
    else:
        torch.save(each_update, '{}/S{}_nonp_{}_epoch.pkl'.format(folder_path,model_id, epoch))
    return update




