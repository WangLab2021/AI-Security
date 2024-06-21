import os
from pathlib import Path
from args import parse_args
from utils.logging import parse_configs_file, create_subdirs, save_checkpoint
from utils.model import get_layers, prepare_model, show_gradients
from utils.schedules import get_optimizer, get_lr_policy
import logging
import torch
import numpy as np
import random
from torchvision import transforms
from celeba_data import CelebAData
from torch.utils.data import DataLoader
import torch.nn as nn
import models
from fairness_metric import val
from adv_trainer import Trainer

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def main():
    args = parse_args()
    parse_configs_file(args)

    result_main_dir = os.path.join(Path(args.result_dir), args.exp_name, args.exp_mode)

    if os.path.exists(result_main_dir):
        n = len(next(os.walk(result_main_dir))[-2])  # prev experiments with same name
        result_sub_dir = os.path.join(
            result_main_dir,
            "{}--k-{:.2f}_lr-{}_epochs-{}_warmuplr-{}_warmupepochs-{}".format(
                n + 1,
                args.k,
                args.lr,
                args.epochs,
                args.warmup_lr,  # 0.1
                args.warmup_epochs,
            ),
        )
    else:
        os.makedirs(result_main_dir, exist_ok=True)
        result_sub_dir = os.path.join(
            result_main_dir,
            "1--k-{:.2f}_lr-{}_epochs-{}_warmuplr-{}_warmupepochs-{}".format(
                args.k,
                args.lr,
                args.epochs,
                args.warmup_lr,
                args.warmup_epochs,
            ),
        )
    create_subdirs(result_sub_dir)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger()
    logger.addHandler(
        logging.FileHandler(os.path.join(result_sub_dir, "setup.log"), "a")
    )
    logger.info(args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    gpu_list = [int(i) for i in args.gpu.strip().split(",")]
    device = torch.device(f"cuda:{gpu_list[0]}" if use_cuda else "cpu")
    args.device = device

    if args.dataset == 'adult':
        args.img_size = 97
    elif args.dataset == 'compas':
        args.img_size = 400
    else:
        args.img_size = 224

    setup_seed(1)

    celeba_data_transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
    ])
    training_dir = 'data/celeba/'
    training_dataset = CelebAData(root_dir=training_dir, target=args.target, transform=celeba_data_transform)

    valid_dir = 'data/celeba/'
    valid_dataset = CelebAData(root_dir=valid_dir, target=args.target, transform=celeba_data_transform, iftrain=False,
                               ifvalid=True)

    test_dir = 'data/celeba/'
    testing_dataset = CelebAData(root_dir=test_dir, target=args.target, transform=celeba_data_transform, iftrain=False)

    setup_seed(1)
    batchsize_train = 64
    train_loader = DataLoader(training_dataset, shuffle=True, batch_size=batchsize_train, persistent_workers=True,
                              pin_memory=True, num_workers=2)

    batchsize_val = 256
    valid_loader = DataLoader(valid_dataset, shuffle=False, batch_size=batchsize_val, pin_memory=True, num_workers=2)

    test_loader = DataLoader(testing_dataset, shuffle=False, batch_size=batchsize_val, pin_memory=True, num_workers=2)

    setup_seed(1)

    args.num_classes = 2
    args.num_groups = 2
    logger.info(args.num_classes)
    logger.info(args.num_groups)
    logger.info(len(train_loader.dataset))
    logger.info(len(test_loader.dataset))

    cl, ll = get_layers(args.layer_type)
    if len(gpu_list) > 1:
        print("Using multiple GPUs")
        model = nn.DataParallel(
            models.__dict__[args.arch](
                cl, ll, args.init_type, num_classes=args.num_classes
            ),
            gpu_list,
        ).to(device)
    else:
        model = models.__dict__[args.arch](
            cl, ll, args.init_type, num_classes=args.num_classes
        ).to(device)
    logger.info(model)

    prepare_model(model, args)

    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, args)  # lr
    lr_policy = get_lr_policy(args.lr_schedule)(optimizer, args)  # lr_scheduler
    logger.info([criterion, optimizer, lr_policy])

    trainer = Trainer(model, optimizer, device)     # todo

    if args.source_net:
        if os.path.isfile(args.source_net):
            logger.info("=> loading source model from '{}'".format(args.source_net))
            checkpoint = torch.load(args.source_net, map_location=device)
            model.load_state_dict(checkpoint["state_dict"])
            logger.info("=> loaded checkpoint '{}'".format(args.source_net))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    show_gradients(model)

    trainer.train(train_loader, test_loader, args.epochs, logger, args, result_sub_dir, lr_policy)

    # for epoch in range(args.start_epoch, args.epochs + args.warmup_epochs):
    #     lr_policy(epoch)  # adjust learning rate
    #
    #     trainer()
    #
    #     prec1, DI, DEO = val(model, device, test_loader, criterion, args)  # testloader
    #     prec1 = prec1.item()
    #     is_best = prec1 > best_prec1
    #     best_prec1 = max(prec1, best_prec1)
    #     save_checkpoint(
    #         {
    #             "epoch": epoch + 1,
    #             "arch": args.arch,
    #             "state_dict": model.state_dict(),
    #             "best_prec1": best_prec1,
    #             "optimizer": optimizer.state_dict(),
    #         },
    #         args,
    #         result_dir=os.path.join(result_sub_dir, "checkpoint"),
    #         epoch=epoch,
    #         save_dense=args.save_dense,
    #     )
    #
    #     is_best_deo = DEO < best_DEO
    #     best_DEO = min(DEO, best_DEO)
    #     is_best_DI = DI > best_DI
    #     best_DI = max(DI, best_DI)
    #
    #     logger.info(
    #         f"Epoch {epoch}, val-method {args.val_method}, validation accuracy {prec1:.2f}, best_prec {best_prec1:.2f}, DEO {DEO:.4f}, best_DEOM {best_DEO:.4f}, DI {DI:.4f}, best_DI {best_DI:.4f}"
    #     )


if __name__ == "__main__":
    main()