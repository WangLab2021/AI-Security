import os
import time

from sklearn.metrics import classification_report
from tqdm import tqdm

import mindspore
import mindspore as ms
import mindspore.nn as nn
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
from mindspore import ops, Tensor, context
from download import download
from datetime import datetime

context.set_context(mode=context.GRAPH_MODE, device_target="CPU", save_graphs=False)

from tools import *
import models_ms.resnet20 as resnet20
import models_ms.resnet as resnet


class SubDataset:

    def __init__(self, indexes, whole_set, catogaries, transform_py=None):
        self.indexes = indexes
        self.whole_set = whole_set  # 这是一个包含 (image, label) 元组的列表
        self.transform = transform_py
        self.catogaries = catogaries

    def __getitem__(self, index: int):
        global_index = self.indexes[index]
        image, label = self.whole_set[global_index]
        # 注意：MindSpore 的数据转换在 .map() 中完成，但为保持逻辑，此处保留
        if self.transform is not None:
            image = self.transform(image)
        return image, label_transfer(self.catogaries, label)

    def __len__(self):
        return len(self.indexes)


def accuracy(output, target, g_catogaries, tc_idx=0, topk=(1,)):
    maxk = max(topk)
    batch_size = target.shape[0]

    _, pred = ops.TopK(sorted=True)(output, maxk)
    pred = pred.T

    # MindSpore没有expand_as，使用broadcast_to
    target_broadcast = ops.broadcast_to(target.reshape(1, -1), pred.shape)
    correct = ops.equal(pred, target_broadcast)

    # 转换到Numpy进行后续处理
    y_true = target.asnumpy()
    y_pred = pred.asnumpy()  # 取top1的预测
    report = (
        [label_transfer_inverse(g_catogaries[tc_idx], y) for y in y_true],
        [label_transfer_inverse(g_catogaries[tc_idx], y) for y in y_pred[0]],  # 修正：只取top1
    )

    res = []
    for k in topk:
        # MindSpore算子替换
        correct_k = correct[:k, :].reshape(-1).astype(mindspore.float32).sum(0)
        res.append(correct_k * (100.0 / batch_size))
    return res, report


def save_checkpoint(state, is_best, filename='checkpoint/resnet20.ckpt'):
    model_params = state['state_dict']
    # MindSpore的save_checkpoint直接保存参数列表
    mindspore.save_checkpoint(model_params, filename)


def validate(val_loader, model, criterion, epoch, g_catogaries, g_teacher_index):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # 切换到验证模式
    model.set_train(False)

    end = time.time()
    report_data = [[], []]
    pbar = tqdm(val_loader, total=len(val_loader), desc=f"Validating Epoch {epoch+1}", unit="batch")
    for i, (image, target) in enumerate(val_loader.create_tuple_iterator()):
        output = model(image)
        loss = criterion(output, target.astype(mindspore.int32))

        # 转换为Float32以保证精度计算稳定性
        output = output.astype(mindspore.float32)
        loss = loss.astype(mindspore.float32)

        # 度量准确率和记录损失
        (prec1, *_), r = accuracy(output, target, g_catogaries=g_catogaries, tc_idx=g_teacher_index)
        report_data[0].extend(r[0])
        report_data[1].extend(r[1])

        losses.update(loss.asnumpy().item(), image.shape[0])
        top1.update(prec1.asnumpy().item(), image.shape[0])

        batch_time.update(time.time() - end)
        end = time.time()
        pbar.set_postfix(
            loss=round(losses.val, 5),
            top1=round(top1.val, 5),
        )
        pbar.update(1)
    pbar.close()
    logger.success(
        "\n".join(
            [
                "=" * 70,
                f"Validation Epoch {epoch + 1} finished",
                str(classification_report(report_data[0], report_data[1])),
                "=" * 70,
            ]
        )
    )
    # if epoch % args.printfreq == args.printfreq - 1:
    #     logger.info("Validate Epoch:", epoch + 1)
    #     logger.info("=" * 70 + "\n", classification_report(report_data[0], report_data[1]), "=" * 70)
    #     logger.info(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    #     logger.info("\n\n")

    # 切换回训练模式
    model.set_train(True)
    return top1.avg


# ==========================================================
# 6. `train` 函数转换
# ==========================================================
def train(train_loader, model, criterion, optimizer, epoch, g_catogaries, g_teacher_index):
    """
    Run one train epoch in MindSpore.
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # 切换到训练模式
    model.set_train(True)

    # ---------- MindSpore 训练核心 ----------
    def forward_fn(data, label):
        logits = model(data)
        loss = criterion(logits, label)
        return loss, logits

    grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    def train_step(data, label):
        (loss, logits), grads = grad_fn(data, label)
        loss = ops.depend(loss, optimizer(grads))  # 确保梯度计算依赖于logits
        # optimizer(grads)
        return loss, logits

    # ----------------------------------------

    end = time.time()
    report_data = [[], []]
    pbar = tqdm(train_loader, total=len(train_loader), desc=f"Training Epoch {epoch+1}", unit="batch")
    for i, (image, target) in enumerate(train_loader.create_tuple_iterator()):
        data_time.update(time.time() - end)

        # 执行单步训练
        loss, output = train_step(image, target.astype(mindspore.int32))

        # 转换为Float32以保证精度计算稳定性
        output = output.astype(mindspore.float32)
        loss_val = loss.astype(mindspore.float32)

        # 度量准确率和记录损失
        (prec1, *_), r = accuracy(output, target, g_catogaries, tc_idx=g_teacher_index)
        report_data[0].extend(r[0])
        report_data[1].extend(r[1])

        losses.update(loss_val.asnumpy().item(), image.shape[0])
        top1.update(prec1.asnumpy().item(), image.shape[0])

        batch_time.update(time.time() - end)
        end = time.time()

        pbar.set_postfix(
            loss=round(losses.val, 5),
            top1=round(top1.val, 5),
            lr=(
                optimizer.learning_rate.numpy()
                if hasattr(optimizer.learning_rate, 'numpy')
                else optimizer.learning_rate
            ),
        )
        pbar.update(1)
    pbar.close()
    logger.success(
        "\n".join(
            [
                "=" * 70,
                f"Training Epoch {epoch + 1} finished",
                str(classification_report(report_data[0], report_data[1])),
                "=" * 70,
            ]
        )
    )

    # if epoch % args.printfreq == args.printfreq - 1:
    #     logger.info('Training Epoch:', epoch + 1)
    #     logger.info("=" * 70 + "\n", classification_report(report_data[0], report_data[1]), "=" * 70 + "\n")


def main(args):
    best_prec1 = 0.0
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    catogaries = args.classes
    g_catogaries = args.classes
    g_teacher_index = 0

    # download = True
    dataset_path = Path('./data/MNIST_Data')
    dataset_files = [
        'test/t10k-images-idx3-ubyte',
        'test/t10k-labels-idx1-ubyte',
        'train/train-images-idx3-ubyte',
        'train/train-labels-idx1-ubyte',
    ]

    if not all([(dataset_path / f).absolute().exists() for f in dataset_files]):
        dataset_path.mkdir(parents=True, exist_ok=True)
        mnist_url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/MNIST_Data.zip"
        download(mnist_url, dataset_path.parent.as_posix(), kind="zip", replace=True)

    raw_train_set = ds.MnistDataset((dataset_path / 'train').as_posix(), shuffle=False)
    raw_valid_set = ds.MnistDataset((dataset_path / 'test').as_posix(), shuffle=False)

    tmp_image = next(raw_train_set.create_tuple_iterator(output_numpy=True))[0]
    image_shape = tmp_image.shape
    logger.info(f"Sample image shape: {image_shape} - {tmp_image.dtype} - {tmp_image.min()} ~ {tmp_image.max()}")

    # 为了使用索引，需要将数据集加载到内存中，这对于小数据集是可行的
    logger.info("Loading raw datasets into memory for indexing...")
    raw_train_set = list(raw_train_set.create_tuple_iterator(output_numpy=True))
    raw_valid_set = list(raw_valid_set.create_tuple_iterator(output_numpy=True))
    logger.info("Loading complete.")

    train_index_dict = build_indexes(raw_train_set)
    train_sets = split_data_by_classes(train_index_dict, catogaries, dataset_size=args.trainset)
    val_index_dict = build_indexes(raw_valid_set)
    val_sets = split_data_by_classes(val_index_dict, catogaries, mode='val', dataset_size=args.trainset)

    teacher_models = ['ResNet20', 'ResNet32', 'ResNet44', 'ResNet56', 'ResNet156']
    model_map = {
        0: resnet20.ResNet20,
        # 0: resnet.ResNet18,
        1: resnet20.ResNet32,
        2: resnet20.ResNet44,
        3: resnet20.ResNet56,
        4: resnet20.ResNet56,
    }

    for teacher_index in range(len(catogaries)):
        if args.teacher_idx >= 0 and teacher_index != args.teacher_idx:
            continue
        logger.info(f"\n{'='*20}\nTraining teacher {teacher_index} | Model: {teacher_models[teacher_index]}\n{'='*20}")

        g_teacher_index = teacher_index
        num_classes = len(catogaries[teacher_index])
        model = model_map.get(teacher_index, resnet20.ResNet56)(num_classes=num_classes, inp_channels=1)

        current_train_set = train_sets[teacher_index]
        current_val_set = val_sets[teacher_index]

        # 定义MindSpore数据处理流程
        transform_list = [vision.ToTensor(), vision.Normalize(mean=(0.1307,), std=(0.3081,), is_hwc=False)]

        # 创建训练数据加载器
        train_dataset_generator = SubDataset(current_train_set, raw_train_set, catogaries[teacher_index])
        train_loader = ds.GeneratorDataset(train_dataset_generator, ["image", "label"], shuffle=True)
        train_loader = train_loader.map(
            operations=transform_list, input_columns="image", num_parallel_workers=args.workers
        )
        train_loader = train_loader.batch(args.batch_size)
        tmp_image = next(train_loader.create_tuple_iterator(output_numpy=True))[0]
        logger.info(
            f"Sample train image shape: {tmp_image.shape} - {tmp_image.dtype} - {tmp_image.min()} ~ {tmp_image.max()}"
        )

        # 创建验证数据加载器
        val_dataset_generator = SubDataset(current_val_set, raw_valid_set, catogaries[teacher_index])
        val_loader = ds.GeneratorDataset(val_dataset_generator, ["image", "label"], shuffle=False)
        val_loader = val_loader.map(operations=transform_list, input_columns="image", num_parallel_workers=args.workers)
        val_loader = val_loader.batch(args.batch_size)

        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        lr = args.lr * 0.1 if args.arch in ['ResNet1202', 'ResNet110'] else args.lr
        # MindSpore的学习率调度器，近似 MultiStepLR
        milestones = [100, 150]
        # learning_rates = [lr * r for r in [0.1, 0.01]]
        # lr_scheduler = nn.piecewise_constant_lr(milestones, learning_rates)
        # if args.start_epoch > 0:
        #     lr_scheduler = lr_scheduler[args.start_epoch :]
        # optimizer.learning_rate = lr_scheduler
        optimizer = nn.SGD(
            model.trainable_params(), learning_rate=lr, momentum=args.momentum, weight_decay=args.weight_decay
        )

        if args.evaluate:
            validate(val_loader, model, criterion, 0, g_catogaries, g_teacher_index)
            return

        for epoch in range(args.start_epoch, args.epochs):
            # 训练一个epoch
            train(train_loader, model, criterion, optimizer, epoch, g_catogaries, g_teacher_index)

            # MindSpore学习率调度器在epoch后更新
            local_lr = lr
            for m in milestones:
                if epoch > m:
                    local_lr *= 0.1
            if local_lr != optimizer.learning_rate:
                ops.assign(optimizer.learning_rate, ms.Tensor(local_lr, ms.float32))
            # optimizer.assignadd(local_lr)  # 更新学习率
            # optimizer.learning_rate = local_lr

            # lr_scheduler.step()

            # 在验证集上评估
            prec1 = validate(val_loader, model, criterion, epoch, g_catogaries, g_teacher_index)

            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)

            # 保存检查点
            dirs = args.save_dir
            if not os.path.exists(dirs):
                os.makedirs(dirs)

            # 获取模型参数用于保存
            param_dict = model.parameters_dict()
            save_checkpoint(
                {'state_dict': list(param_dict.values()), 'best_prec1': best_prec1},
                is_best,
                filename=os.path.join(dirs, f'Teacher_{teacher_index}_even.ckpt'),
            )

        # break


if __name__ == '__main__':
    log_file = Path("./logs", Path(__file__).stem, datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".log")
    setup_logger(log_file=log_file)

    logger.info("Training Teacher [ensemble] with MindSpore")
    logger.info('--------args----------')
    arg_parser = get_train_teacher_argparser(
        model_names=["ResNet20", "ResNet32", "ResNet44", "ResNet56", "ResNet110"]
    )
    arg_parser.add_argument("--teacher_idx", type=int, default=-1, help="Index of the teacher model to train")
    args = arg_parser.parse_args()
    args.workers = max(1, args.workers)  # MindSpore不支持负数工作线程
    for k, v in vars(args).items():
        logger.info('%s: %s' % (k, v))
    logger.info('----------------------')
    main(args)
