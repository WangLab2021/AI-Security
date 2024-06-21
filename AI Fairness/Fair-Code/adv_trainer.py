from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from torch.autograd import Function
from torch.optim.lr_scheduler import ReduceLROnPlateau
from fairness_metric import val
from utils.logging import save_checkpoint
import os
from utils.schedules import get_lr_policy


class MLP(nn.Module):
    def __init__(self, feature_size, hidden_dim, num_classes=None, num_layer=3, adv=False, adv_lambda=1.):
        super(MLP, self).__init__()
        try: #list
            in_features = self.compute_input_size(feature_size)
        except : #int
            in_features = feature_size

        num_outputs = num_classes
        self.adv = adv
        if self.adv:
            self.adv_lambda = adv_lambda
        self._make_layer(in_features, hidden_dim, num_classes, num_layer)

    def forward(self, feature, get_inter=False):
        feature = torch.flatten(feature, 1)
        if self.adv:
            feature = ReverseLayerF.apply(feature, self.adv_lambda)

        h = self.features(feature)
        out = self.head(h)
        out = out.squeeze()

        if get_inter:
            return h, out
        else:
            return out

    def compute_input_size(self, feature_size):
        in_features = 1
        for size in feature_size:
            in_features = in_features * size

        return in_features

    def _make_layer(self, in_dim, h_dim, num_classes, num_layer):

        if num_layer == 1:
            self.features = nn.Identity()
            h_dim = in_dim
        else:
            features = []
            for i in range(num_layer-1):
                features.append(nn.Linear(in_dim, h_dim) if i == 0 else nn.Linear(h_dim, h_dim))
                features.append(nn.ReLU())
            self.features = nn.Sequential(*features)

        self.head = nn.Linear(h_dim, num_classes)


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class Trainer:
    def __init__(self, model, optimizer, device):
        self.adv_lambda = 1
        self.adv_lr = 0.001
        self.target_criterion = 'eo'
        self.model = model
        self.optimizer = optimizer
        self.criterion = torch.nn.CrossEntropyLoss()
        self.scheduler = None
        self.device = device

    def train(self, train_loader, test_loader, epochs, logger, args, result_sub_dir, lr_policy):
        model = self.model
        num_groups = 2
        num_classes = 2
        self._init_adversary(num_groups, num_classes, train_loader)
        lr_policy_adv = get_lr_policy(args.lr_schedule)(self.adv_optimizer, args)  # lr_scheduler
        # self.scheduler = ReduceLROnPlateau(self.optimizer, patience=5)
        best_prec1 = 0
        best_DEO = 100
        best_DI = 100

        for epoch in range(epochs):
            lr_policy(epoch)
            lr_policy_adv(epoch)
            self._train_epoch(epoch, train_loader, model)

            prec1, DI, DEO = val(model, self.device, test_loader, self.criterion, None)

            prec1 = prec1.item()
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            is_best_deo = DEO < best_DEO
            best_DEO = min(DEO, best_DEO)
            is_best_DI = DI > best_DI
            best_DI = max(DI, best_DI)

            logger.info(
                f"Epoch {epoch}, validation accuracy {prec1:.2f}, best_prec {best_prec1:.2f}, DEO {DEO:.4f}, best_DEOM {best_DEO:.4f}, DI {DI:.4f}, best_DI {best_DI:.4f}"
            )

            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "arch": args.arch,
                    "state_dict": model.state_dict(),
                    "best_prec1": best_prec1,
                    "optimizer": self.optimizer.state_dict(),
                },
                args,
                result_dir=os.path.join(result_sub_dir, "checkpoint"),
                epoch=epoch,
                save_dense=args.save_dense,
            )

            # if self.scheduler != None:
            #     self.scheduler.step(eval_loss)
            #     self.adv_scheduler.step(eval_adv_loss)

        print('Training Finished!')
        return model

    def _train_epoch(self, epoch, train_loader, model):
        num_classes = 2
        num_groups = 2

        model.train()

        running_acc = 0.0
        running_loss = 0.0
        running_adv_loss = 0.0
        batch_start_time = time.time()

        for i, data in enumerate(train_loader):
            # Get the inputs
            inputs, _, groups, targets, _ = data
            labels = targets
            # groups = groups.long()

            if self.cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()
                groups = groups.cuda()

            labels = labels.long()
            groups = groups.long()

            outputs = model(inputs)

            inputs_for_adv = outputs
            logits = outputs

            adv_inputs = None
            if self.target_criterion =='eo':
                repeat_times = num_classes
                input_loc = F.one_hot(labels.long(), num_classes).repeat_interleave(repeat_times, dim=1)
                adv_inputs = inputs_for_adv.repeat(1, repeat_times) * input_loc
                adv_inputs = torch.cat((inputs_for_adv, adv_inputs), dim=1)

            elif self.target_criterion == 'dp':
                adv_inputs = inputs_for_adv

            adv_preds = self.sa_clf(adv_inputs)
#             adv_loss = self.adv_criterion(self.sa_clf, adv_preds, groups)
            adv_loss = self.adv_criterion(adv_preds, groups)

            self.optimizer.zero_grad()
            self.adv_optimizer.zero_grad()

            #adv_loss.backward()#retain_graph=True)
            #adv_loss.backward(retain_graph=True)
            #for n, p in model.named_parameters():
            #    unit_adv_grad = p.grad / (p.grad.norm() + torch.finfo(torch.float32).tiny)
            #    p.grad += torch.sum(p.grad * unit_adv_grad) * unit_adv_grad # gradients are already reversed

            loss = self.criterion(logits, labels)

            (loss+adv_loss).backward()

            self.optimizer.step()
            self.adv_optimizer.step()

#             running_loss += loss.item()
#             running_adv_loss += adv_loss.item()
#             # binary = True if num_classes ==2 else False
#             running_acc += get_accuracy(outputs, labels)
#
# #             self.optimizer.step()
# #             self.adv_optimizer.step()
#
#             if i % self.term == self.term - 1:  # print every self.term mini-batches
#                 avg_batch_time = time.time() - batch_start_time
#                 print_statement = '[{}/{}, {:5d}] Method: {} Train Loss: {:.3f} Adv Loss: {:.3f} Train Acc: {:.2f} [{:.2f} s/batch]'\
#                     .format(epoch + 1, self.epochs, i + 1, self.method, running_loss / self.term,
#                             running_adv_loss / self.term,running_acc / self.term, avg_batch_time / self.term)
#                 print(print_statement)
#
#                 running_loss = 0.0
#                 running_acc = 0.0
#                 running_adv_loss = 0.0
#                 batch_start_time = time.time()

    # def evaluate(self, model, adversary, loader, criterion, adv_criterion):
    #     model.eval()
    #     num_groups = 2
    #     num_classes = 2
    #     eval_acc = 0
    #     eval_adv_acc = 0
    #     eval_loss = 0
    #     eval_adv_loss = 0
    #     eval_eopp_list = torch.zeros(num_groups, num_classes).cuda()
    #     eval_data_count = torch.zeros(num_groups, num_classes).cuda()
    #
    #     if 'Custom' in type(loader).__name__:
    #         loader = loader.generate()
    #     with torch.no_grad():
    #         for j, eval_data in enumerate(loader):
    #             # Get the inputs
    #             inputs, _, groups, classes, _ = eval_data
    #             #
    #             labels = classes
                # groups = groups.long()
                # if self.cuda:
                #     inputs = inputs.cuda()
                #     labels = labels.cuda()
                #     groups = groups.cuda()
                #
                # labels = labels.long()
                #
                # get_inter = False
                # outputs = model(inputs, get_inter=get_inter)
                #
                # inputs_for_adv = outputs[-2] if get_inter else outputs
                # logits = outputs[-1] if get_inter else outputs
                #
                # adv_inputs = None
                # if self.target_criterion == 'eo':
                #     repeat_times = num_classes
                #     input_loc = F.one_hot(labels.long(), num_classes).repeat_interleave(repeat_times, dim=1)
                #     adv_inputs = inputs_for_adv.repeat(1, repeat_times) * input_loc
                #     adv_inputs = torch.cat((inputs_for_adv, adv_inputs), dim=1)
                #
                # elif self.target_criterion == 'dp':
                #     adv_inputs = inputs_for_adv
                #
                # loss = criterion(logits, labels)
                # eval_loss += loss.item() * len(labels)
                # binary = True if num_classes == 2 else False
                # acc = get_accuracy(outputs, labels, reduction='none')
                # eval_acc += acc.sum()
                #
                # for g in range(num_groups):
        #             for l in range(num_classes):
        #                 eval_eopp_list[g, l] += acc[(groups == g) * (labels == l)].sum()
        #                 eval_data_count[g, l] += torch.sum((groups == g) * (labels == l))
        #
        #         adv_preds = adversary(adv_inputs)
        #         # groups = groups.float() if num_groups == 2 else groups.long()
        #         groups = groups.long()
        #         adv_loss = adv_criterion(adv_preds, groups)
        #         eval_adv_loss += adv_loss.item() * len(labels)
        #         # binary = True if num_groups == 2 else False
        #         eval_adv_acc += get_accuracy(adv_preds, groups)
        #
        #     eval_loss = eval_loss / eval_data_count.sum()
        #     eval_acc = eval_acc / eval_data_count.sum()
        #     eval_adv_loss = eval_adv_loss / eval_data_count.sum()
        #     eval_adv_acc = eval_adv_acc / eval_data_count.sum()
        #     eval_eopp_list = eval_eopp_list / eval_data_count
        #     eval_max_eopp = torch.max(eval_eopp_list, dim=0)[0] - torch.min(eval_eopp_list, dim=0)[0]
        #     eval_max_eopp = torch.max(eval_max_eopp).item()
        # model.train()
        # return eval_loss, eval_acc, eval_adv_loss, eval_adv_acc, eval_max_eopp

    def _init_adversary(self, num_groups, num_classes, dataloader):
        self.model.eval()
        if self.target_criterion == 'eo':
            feature_size = num_classes * (num_classes + 1)
        elif self.target_criterion == 'dp':
            feature_size = num_classes


        sa_clf = MLP(feature_size=feature_size, hidden_dim=32, num_classes=num_groups,
                     num_layer=2, adv=True, adv_lambda=self.adv_lambda)
        if self.cuda:
            sa_clf.cuda()
        sa_clf.train()
        self.sa_clf = sa_clf
        self.adv_optimizer = optim.Adam(sa_clf.parameters(), lr=self.adv_lr)
        self.adv_scheduler = ReduceLROnPlateau(self.adv_optimizer, patience=5)
        self.adv_criterion = self.criterion

    # def criterion(self, model, outputs, labels):
    #     return nn.CrossEntropyLoss()(outputs, labels)