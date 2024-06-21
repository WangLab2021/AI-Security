# fairness-guided trainer
import time

import torch
import torchvision
from utils.logging import AverageMeter, ProgressMeter
from utils.eval import accuracy


def get_time():
    return str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))


def train(model, discriminator, device, train_loader, criterion, optimizer_c, optimizer_d, epoch, args, logger):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    top1 = AverageMeter("Acc_1", ":6.2f")
    top5 = AverageMeter("Acc_5", ":6.2f")
    eval_eopp_list = torch.zeros(args.num_groups, args.num_classes).to(args.device)
    eval_data_count = torch.zeros(args.num_groups, args.num_classes).to(args.device)

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch),
    )

    model.train()
    end = time.time()

    dataloader = train_loader
    bce_loss = torch.nn.BCELoss()
    loss_item_classifier = None
    loss_item_discriminator = None

    for i, data in enumerate(dataloader):
        inputs, _, groups, targets, (idx, _) = data
        images = inputs.to(device)
        target = targets.to(device)
        groups = groups.to(device)
        # images, target = data[0].to(device), data[1].to(device)

        # forward classifier
        optimizer_c.zero_grad()
        output = model(images)

        # train fairness discriminator
        optimizer_d.zero_grad()
        output_d = discriminator(output.detach())
        d_loss = bce_loss(output_d, groups)               # sl_train -> sensitive features
        d_loss.backward()
        optimizer_d.step()

        # update classifier
        loss_c = criterion(output, target)                  # todo loss_update

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss_c.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        loss_c.backward()
        optimizer_c.step()

        batch_time.update(time.time() - end)
        end = time.time()

        loss_item_classifier = loss_c.item()
        loss_item_discriminator = loss_c.item()



    logger.info(
        f"{get_time()} Epoch {epoch}, loss_classifier {loss_item_classifier:.2f}, loss_discriminator {loss_item_discriminator:.2f}"
    )

