import argparse
import json
import datetime
import os
import logging
import random

import torch
import torch.nn as nn
import copy
from image_helper import ImageHelper
from torch.utils.data import DataLoader
from shadow_training import train_shadow, shadow_update, shadow_data
from tensorboardX import SummaryWriter
logger = logging.getLogger("logger")
# logger.setLevel("ERROR")
import yaml
import time

import numpy as np
from read_prop import read_prop_ind


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

criterion = torch.nn.CrossEntropyLoss()


def train(helper, epoch, train_data_sets, local_model, target_model, is_poison, poison_epoch, prop_epoch, writer, last_weight_accumulator=None):

    ### Accumulate weights for all participants.
    weight_accumulator = dict()
    for name, data in target_model.state_dict().items():
        #### don't scale tied weights:
        if helper.params.get('tied', False) and name == 'decoder.weight' or '__'in name:
            continue
        weight_accumulator[name] = torch.zeros_like(data)

    current_number_of_adversaries = 0

    ###train_data_sets key:user_id value:train_data
    for model_id, _ in train_data_sets:
        if model_id == -1 or model_id in helper.params['adversary_list']:
            current_number_of_adversaries += 1
    logger.info(f'There are {current_number_of_adversaries} adversaries in the training.')


    ### local training
    for model_id in range(helper.params['no_models']):
        model = local_model
        ## Synchronize LR and models
        model.copy_params(target_model.state_dict())
        optimizer = torch.optim.SGD(model.parameters(), lr=helper.params['lr'],
                                    momentum=helper.params['momentum'],
                                    weight_decay=helper.params['decay'])
        model.train()
        start_time = time.time()
        _, (current_data_model, train_data) = train_data_sets[model_id]
        batch_size = helper.params['batch_size']
        ### For a 'poison_epoch' we perform single shot poisoning
        if current_data_model == -1:
            ### The participant got compromised and is out of the training.
            #  It will contribute to poisoning,
            continue
        if is_poison and current_data_model in helper.params['adversary_list'] and \
                (epoch in poison_epoch):

            logger.info('poison_now')
            poisoned_data = helper.poisoned_data_for_train
            _, acc_initial = test(helper=helper, epoch=epoch, data_source=helper.test_data,
                             model=model, is_poison=False, visualize=False)
            poison_lr = helper.params['poison_lr']

            retrain_no_times = helper.params['retrain_poison']
            step_lr = helper.params['poison_step_lr']
            poison_optimizer = torch.optim.SGD(model.parameters(), lr=poison_lr,
                                               momentum=helper.params['momentum'],
                                               weight_decay=helper.params['decay'])
            scheduler = torch.optim.lr_scheduler.MultiStepLR(poison_optimizer,
                                                             milestones=[0.2 * retrain_no_times,
                                                                         0.8 * retrain_no_times],
                                                             gamma=0.1)
            try:
                ###retrain_no_times = retrain poison
                for internal_epoch in range(1, retrain_no_times + 1):
                    if step_lr:
                        scheduler.step()
                        logger.info(f'Current lr: {scheduler.get_lr()}')
                    data_iterator = copy.deepcopy(poisoned_data)
                    logger.info(f"PARAMS: {helper.params['retrain_poison']} epoch: {internal_epoch},"
                                f" lr: {scheduler.get_lr()}")
                    poison_perbatch = random.sample(helper.poison_images, helper.params['poisoning_per_batch'])
                    for batch_id, batch in enumerate(data_iterator):
                        #### poisoned data
                        if helper.params['type'] == 'image':
                            for pos, image in enumerate(poison_perbatch):
                                poison_pos = min(len(batch[0])-1, pos)
                                #random.randint(0, len(batch))
                                batch[0][poison_pos] = helper.train_dataset[image][0]
                                batch[0][poison_pos].add_(torch.cuda.FloatTensor(batch[0][poison_pos].shape).normal_(0, helper.params['noise_level']))
                                # batch[1][poison_pos] = helper.params['poison_label_swap']
                                true_label = batch[1][poison_pos]
                                batch[1][poison_pos] = torch.abs((true_label - 1))
                        data, targets = helper.get_batch(poisoned_data, batch, False)
                        poison_optimizer.zero_grad()
                        output = model(data)
                        class_loss = nn.functional.cross_entropy(output, targets)
                        loss = class_loss
                        loss.backward()
                        poison_optimizer.step()

            except ValueError:
                logger.info('')


            ### Adversary wants to scale his weights. Baseline model doesn't do this
            if not helper.params['baseline']:
                ### We scale data according to formula: L  = 100*X-99*G = G + (100*X- 100*G).
                clip_rate = (helper.params['scale_weights'] / current_number_of_adversaries)

                logger.info(f"Scaling by  {clip_rate}")
                for key, value in model.state_dict().items():
                    #### don't scale tied weights:
                    if helper.params.get('tied', False) and key == 'decoder.weight' or '__'in key:
                        continue
                    target_value = target_model.state_dict()[key]
                    new_value = target_value + (value - target_value) * clip_rate

                    model.state_dict()[key].copy_(new_value)

            for key, value in model.state_dict().items():
                #### don't scale tied weights:
                if helper.params.get('tied', False) and key == 'decoder.weight' or '__'in key:
                    continue
                target_value = target_model.state_dict()[key]
                new_value = target_value + (value - target_value) * current_number_of_adversaries
                model.state_dict()[key].copy_(new_value)



        else:
            ### we will load helper.params later
            if helper.params['fake_participants_load']:
                continue
            for internal_epoch in range(1, helper.params['retrain_no_times'] + 1):
                total_loss = 0.

                data_iterator = copy.deepcopy(train_data)
                # logger.info('user {} has {} length of dataset'.format(model_id, len(train_data)))
                if model_id == range(helper.params['no_models'])[-1] and epoch in prop_epoch:
                    p_shadow_indices = random.sample(helper.poison_images_client, helper.params['prop_len'])
                    p_shadow_indices.extend(random.sample(helper.client_indices, helper.params['total_len']-helper.params['prop_len']))
                    random.shuffle(p_shadow_indices)
                    dataset_p = DataLoader(dataset=helper.train_dataset, batch_size=helper.params['batch_size'],
                                                  sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                      p_shadow_indices))
                    data_iterator = dataset_p


                for batch_id, batch in enumerate(data_iterator):
                    optimizer.zero_grad()

                    data, targets = helper.get_batch(data_iterator, batch, evaluation=False)
                    output = model(data)
                    loss = nn.functional.cross_entropy(output, targets)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.data

        for name, data in model.state_dict().items():
            #### don't scale tied weights:
            if helper.params.get('tied', False) and name == 'decoder.weight' or '__'in name:
                continue
            weight_accumulator[name].add_(data - target_model.state_dict()[name])

    return weight_accumulator




def test(helper, epoch, data_source,
         model, is_poison=False):
    model.eval()
    total_loss = 0
    correct = 0
    dataset_size = len(helper.test_indices)
    data_iterator = data_source
    with torch.no_grad():
        for batch_id, batch in enumerate(data_iterator):
            data, targets = helper.get_batch(data_source, batch, evaluation=True)
            try:

                output = model(data)
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    print("WARNING: out of memory")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    raise exception
            total_loss += nn.functional.cross_entropy(output, targets,
                                                      reduction='sum').item()   # sum up batch loss
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()


    acc = 100.0 * (float(correct) / float(dataset_size))
    total_l = total_loss / dataset_size
    logger.info('___Test {} with_poisoned: {}, epoch: {}: Average loss: {:.4f}, '
                    'Accuracy: {}/{} ({:.4f}%)'.format(model.name, is_poison, epoch,
                                                       total_l, correct, dataset_size,
                                                       acc))
    model.train()
    return (total_l, acc)

    acc = 100.0 * (correct / dataset_size)
    total_l = total_loss / dataset_size
    logger.info('___Test {} with_poisoned: {}, epoch: {}: Average loss: {:.4f}, '
                'Accuracy: {}/{} ({:.0f}%)'.format(model.name, is_poison, epoch,
                                                   total_l, correct, dataset_size,
                                                   acc))
    model.train()
    return total_l, acc


if __name__ == '__main__':
    exp_times = 5
    start =1
    for exp_i in range(start, exp_times+start):
        logger.info('Start training')
        time_start_load_everything = time.time()
        parser = argparse.ArgumentParser(description='train')
        parser.add_argument('--params', dest='params')
        args = parser.parse_args()
        shadow_model_update = []
        with open(f'./{args.params}', 'r') as f:
            params_loaded = yaml.load(f)
        current_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')
        helper = ImageHelper(current_time=current_time, params=params_loaded,
                                 name=params_loaded.get('name', 'image'))

        root = helper.params['root_path']
        exp_folder = f'{root}/exp{exp_i}'
        try:
            os.mkdir(exp_folder)
        except FileExistsError:
            print('Folder already exists')
        read_prop_ind(helper)
        helper.load_data()
        helper.create_model()
        if helper.params['p_epoch_manual']:
            poison_epoch = helper.params['poison_epochs']
        else:
            poison_epoch = list(range(1, helper.params['epochs']+1))
        ### Create models
        if helper.params['is_poison']:
            helper.params['adversary_list'] = [0]+ \
                                    random.sample(range(helper.params['number_of_total_participants']),
                                                          helper.params['number_of_adversaries']-1)      #todo:从所有参与者中挑选出规定数目的攻击者，因为[0]一定是攻击者所以-1
            logger.info(f"Poisoned following participants: {len(helper.params['adversary_list'])}")
        else:
            helper.params['adversary_list'] = list()

        best_loss = float('inf')
        participant_ids = range(len(helper.train_data))
        mean_acc = list()
        weight_accumulator = None
        with open(f'{helper.folder_path}/params.yaml', 'w') as f:
            yaml.dump(helper.params, f)
        dist_list = list()
        model_update = []
        model_update_label = []
        gmodel_path = helper.params['gmodel_path']
        target_model0 = copy.deepcopy(helper.target_model.state_dict())
        torch.save(target_model0, f'{gmodel_path}/Gmodel_{0}_epoch.pkl')
        if helper.params['epochs']>1999:
            prop_epoch = random.sample(range(1,3000), 1500)
        else:
            prop_epoch = helper.params['participant_prop_epoch']

        helper.target_model.load_state_dict(torch.load('saved_params/model.pkl'))
        for epoch in range(helper.start_epoch, helper.params['epochs'] + 1):
            start_time = time.time()
            if epoch in poison_epoch:
                ### For poison epoch we put one adversary and other adversaries just stay quiet
                subset_data_chunks = [participant_ids[0]] + [-1] * (
                helper.params['number_of_adversaries'] - 1) + \
                                            random.sample(participant_ids[1:],
                                                        helper.params['no_models'] - helper.params[
                                                            'number_of_adversaries'])
                logger.info(f'Selected models: {subset_data_chunks}')
            else:
                subset_data_chunks = random.sample(participant_ids[1:], helper.params['no_models'])
                logger.info(f'Selected models: {subset_data_chunks}')

            t = time.time()
            weight_accumulator = train(helper=helper, epoch=epoch,
                                       train_data_sets=[(pos, helper.train_data[pos]) for pos in
                                                        subset_data_chunks],
                                       local_model=helper.local_model, target_model=helper.target_model,
                                       is_poison=helper.params['is_poison'], poison_epoch=poison_epoch,
                                       prop_epoch=prop_epoch,
                                       last_weight_accumulator=weight_accumulator)

            logger.info(f'time spent on training: {time.time() - t}')
            # Average the models
            helper.average_shrink_models(target_model=helper.target_model,
                                         weight_accumulator=weight_accumulator, epoch=epoch, exp_time=exp_i)

            if epoch == 2000 or epoch ==3000:

                torch.save(helper.target_model.state_dict(), 'saved_params/Gmodel_{}_epoch.pkl'.format(epoch))

            epoch_loss, epoch_acc = test(helper=helper, epoch=epoch, data_source=helper.test_data,
                                         model=helper.target_model, is_poison=False, visualize=False)

            logger.info(f'Done in {time.time()-start_time} sec.')

        for shadow_model_id in range(helper.params['no_shadow_models']):
            shadow_model = helper.shadow_model
            # shadow_model.load_state_dict(torch.load(f'{gmodel_path}/Gmodel_{0}_epoch.pkl'))
            shadow_model.load_state_dict(torch.load(f'saved_params/Gmodel_{2000}_epoch.pkl'))
            shadow_dataset_p, shadow_dataset_nonp = shadow_data(helper)
            for epoch_s in range(helper.start_epoch, helper.params['epochs'] + 1):
                if epoch_s in helper.params['shadow_prop_epoch']:
                    logger.info('train shadow model{} with p-data.....'.format(shadow_model_id))
                    shadow_dict1 = copy.deepcopy(shadow_model.state_dict())
                    modelparams_p = train_shadow(helper, shadow_model=shadow_model, shadow_model_id=shadow_model_id,
                                                 shadow_data=shadow_dataset_p,
                                                 epoch=epoch_s, poison_epoch=poison_epoch)
                    update_p = shadow_update(helper, epoch_s, shadow_dict1, modelparams_p, exp_i, model_id=shadow_model_id,
                                             with_p=True)
                    p_loss, p_acc = test(helper=helper, epoch=epoch_s, data_source=helper.test_data,
                                                 model=shadow_model, is_poison=False, visualize=False)
                else:
                    # shadow_model.load_state_dict(torch.load('saved_params/shadow_model_{}.pkl'.format(shadow_model_id)))
                    logger.info('train shadow model{} with nonp-data.....'.format(shadow_model_id))
                    shadow_dict2 = copy.deepcopy(shadow_model.state_dict())
                    modelparams_nonp = train_shadow(helper, shadow_model=shadow_model, shadow_model_id=shadow_model_id,
                                                    shadow_data=shadow_dataset_nonp,
                                                    epoch=epoch_s, poison_epoch=poison_epoch)
                    update_nonp = shadow_update(helper, epoch_s, shadow_dict2, modelparams_nonp, exp_i, model_id=shadow_model_id,
                                                with_p=False)
                    nonp_loss, nonp_acc = test(helper=helper, epoch=epoch_s, data_source=helper.test_data,
                                                 model=shadow_model, is_poison=False, visualize=False)
