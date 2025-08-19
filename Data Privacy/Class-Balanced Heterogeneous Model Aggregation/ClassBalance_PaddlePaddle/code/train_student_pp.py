import os
from datetime import datetime

import paddle
import paddle.vision.datasets as datasets
import paddle.vision.transforms as transforms
import paddle.nn.functional as F

import models_pp.resnet20 as resnet20
import models_pp.resnet as resnet
import paddle.optimizer as optim
import numpy as np
import random

from tools import *


# 重写模型加载/保存函数
def load_checkpoint(checkpoint_path, model, optimizer=None):
    """使用 PaddlePaddle API 加载模型参数。"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"File doesn't exist {checkpoint_path}")

    # 使用 paddle.load 加载
    params_dict = paddle.load(checkpoint_path)
    logger.info(f"Loaded checkpoint from {checkpoint_path}")

    # 使用 model.set_state_dict()
    model.set_state_dict(params_dict['state_dict'])

    if optimizer:
        if 'optim_dict' in params_dict:
            optimizer.set_state_dict(params_dict['optim_dict'])
        else:
            Warning("No optimizer state found in checkpoint, skipping optimizer loading.")

    return params_dict


def save_checkpoint(state, checkpoint, name):
    """使用 PaddlePaddle API 保存模型参数。"""
    # 推荐使用 .pdparams 扩展名
    filepath = os.path.join(checkpoint, name + '.pdparams')
    if not os.path.exists(checkpoint):
        logger.info(f"Checkpoint Directory does not exist! Making directory {checkpoint}")
        os.makedirs(checkpoint)

    # 使用 paddle.save 保存
    paddle.save(state, filepath)
    logger.info(f"Checkpoint saved to {filepath}")


def loss_fn_kd(outputs, labels, teacher_outputs, params, categories):
    """知识蒸馏损失函数 (PaddlePaddle 版本)。"""
    alpha = params.alpha
    T = params.temperature

    ce_loss = paddle.nn.functional.cross_entropy(outputs, labels)

    kd_loss = 0

    for i in range(len(categories)):
        category = categories[i]
        t_output = teacher_outputs[i]

        outputs = F.softmax(outputs / T, axis=1)
        index = list(map(int, category))  # map(function, iterable)
        index = paddle.to_tensor(list(map(int, category)), dtype='int64')
        s_t_outputs = paddle.index_select(outputs, index, axis=1)

        sum_u_k = paddle.sum(s_t_outputs, axis=1, keepdim=True)  # sum_u_k:学生模型在这几个类别上的预测的总和
        log_sum_u_k = paddle.log(sum_u_k)

        u_l = paddle.log(s_t_outputs)  # |L_i|维   pa

        tmp_right = u_l - log_sum_u_k

        t_output = F.softmax(t_output / T, axis=1)

        res = paddle.multiply(t_output, tmp_right)  # 两个|L_i|维向量相乘
        res = paddle.sum(res) * -1

        kd_loss += res

    ce_loss = 10 * ce_loss
    total_loss = kd_loss + ce_loss
    return total_loss, kd_loss, ce_loss


def train(model, teacher_models, optimizer, dataloader, args, categ):
    meter = AverageMeter()
    
    model.train()

    # 教师模型进入评估模式
    for t_model in teacher_models:
        t_model.eval()

    total_correct = 0
    total_samples = 0

    from tqdm import tqdm

    pbar = tqdm(dataloader, total=len(dataloader), desc="Training", unit="batch")

    for i, (train_batch, labels_batch) in enumerate(dataloader):
        # 无需手动 .cuda() 或 Variable()
        optimizer.clear_grad()
        output_teacher_batch = []

        # 使用 paddle.no_grad()
        with paddle.no_grad():
            for t_model in teacher_models:
                output_t_batch = t_model(train_batch)
                output_teacher_batch.append(output_t_batch)

        output_batch = model(train_batch)
        loss, kd_loss, ce_loss = loss_fn_kd(output_batch, labels_batch, output_teacher_batch, args, categ)

        # 梯度清零、反向传播、更新
        loss.backward()
        optimizer.step()

        preds = paddle.argmax(output_batch, axis=1)
        preds = preds.flatten()  # 确保预测是1D的
        labels_batch = labels_batch.flatten()  # 确保标签是1D的
        num_right = paddle.sum((preds == labels_batch)).item()
        num_batch = labels_batch.shape[0]

        total_correct += num_right
        total_samples += num_batch

        pbar.set_postfix(
            loss="{} - {} - {}".format(
                round(loss.numpy().item(), 5), round(kd_loss.numpy().item(), 5), round(ce_loss.numpy().item(), 5)
            ),
            acc="{}({})".format(round(num_right / num_batch, 2), round(total_correct / total_samples, 2)),
            lr=round(optimizer.get_lr(), 5),
        )
        pbar.update(1)
        meter.update(np.array([l.numpy().item() for l in [loss, kd_loss, ce_loss]]))

    pbar.close()

    acc = total_correct / total_samples    
    logger.info("training acc: {} loss: {}".format(round(acc, 5), meter.avg))

    return acc


def evaluate(model, dataloader, args):
    """
    使用 PaddlePaddle 框架实现的模型评估函数。
    """
    count_n = 0

    # 修改点 1: 模型评估模式切换
    model.eval()

    num_sample = 0
    num_right = [0] * 10
    num_sample_classes = [0] * 10

    from tqdm import tqdm

    pbar = tqdm(dataloader, total=len(dataloader), desc="Evaluating", unit="batch")

    with paddle.no_grad():
        for i, (data_batch, labels_batch) in enumerate(dataloader):
            labels_batch = labels_batch.flatten()  # 确保标签是1D的
            output_batch = model(data_batch)

            output_batch = output_batch.numpy()
            labels_batch = labels_batch.numpy()

            outputs = np.argmax(output_batch, axis=1)
            outputs, labels_batch = outputs.flatten(), labels_batch.flatten()
            count_n += np.sum(outputs == labels_batch)
            num_sample += len(labels_batch)

            num_right_batch = []
            num_sample_batch = []

            # 假设 classes 变量已定义
            classes = list(range(10))  # 示例定义

            for j in range(len(classes)):
                num_right_class = np.count_nonzero(
                    (outputs == labels_batch) & (outputs == classes[j])
                )  # 一个batch中每个类别分类正确的次数
                num_sample_class = np.sum(labels_batch == classes[j])  # 一个batch中每个类别的样本的个数
                # logger.info(f"Class {classes[j]}: {num_right_class} correct out of {num_sample_class} samples")
                num_right_batch.insert(j, num_right_class)  # 一个batch中所有类别分类正确的次数
                num_sample_batch.insert(j, num_sample_class)

            num_right = [a + b for a, b in zip(num_right, num_right_batch)]
            num_sample_classes = [a + b for a, b in zip(num_sample_classes, num_sample_batch)]
            pbar.set_postfix(
                acc="{}({})".format(
                    round(sum(num_right_batch) / sum(num_sample_batch), 2),
                    round(sum(num_right) / sum(num_sample_classes), 2),
                ),
            )
            pbar.update(1)  # 更新进度条
    pbar.close()

    safe_num_sample_classes = [i if i != 0 else 1 for i in num_sample_classes]

    acc = count_n / num_sample if num_sample > 0 else 0.0
    acc_classes = [float(a / b) for a, b in zip(num_right, safe_num_sample_classes)]

    logger.info(
        "evaluation acc:{} \nevaluation acc of classes:{}".format(round(acc, 5), [round(x, 5) for x in acc_classes])
    )
    return acc


if __name__ == '__main__':
    args = get_train_student_argparser().parse_args()
    log_file = Path("./logs", Path(__file__).stem, datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".log")
    setup_logger(log_file=log_file)
    
    use_gpu = args.use_gpu and paddle.is_compiled_with_cuda()
    args.use_gpu = use_gpu
    device = 'gpu' if use_gpu else 'cpu'
    paddle.set_device(device)
    logger.info(f"Using device: {device}")

    # 设置随机种子
    random.seed(230)
    np.random.seed(230)
    paddle.seed(230)

    # --- 数据加载 ---
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_set = datasets.FashionMNIST(mode='train', download=True, transform=transform)
    test_set = datasets.FashionMNIST(mode='test', download=True, transform=transform)
    logger.info(f"Train set size: {len(train_set)}, Test set size: {len(test_set)}")
    logger.info(f"Picture size: {train_set[0][0].shape} data range: {train_set[0][0].min()} - {train_set[0][0].max()}")
    train_loader = paddle.io.DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers)
    test_loader = paddle.io.DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers)

    # --- 加载教师模型 ---
    save_path = Path(args.save_dir)
    if not save_path.exists():
        raise FileNotFoundError(f"Save directory {args.save_dir} does not exist.")
    teacher_models = []
    catogaries = args.classes
    logger.info(f"Categories for teachers: {catogaries}")
    teacher_models_name = ['ResNet20', 'ResNet32', 'ResNet44', 'ResNet56', 'ResNet56']
    model_map = {i: getattr(resnet20, name) for i, name in enumerate(teacher_models_name)}
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    for index in range(len(teacher_models_name)):
        num_classes = len(catogaries[index])
        model = model_map[index](num_classes=num_classes, inp_channels=1)
        ckpt_path = save_path / f'Teacher_{index}_even.pdparams'
        load_checkpoint(ckpt_path.as_posix(), model)
        teacher_models.append(model)

    # --- 学生模型定义 ---
    student_model = resnet.ResNet18()

    # --- 优化器和学习率调度器 ---
    scheduler = optim.lr.StepDecay(learning_rate=args.learning_rate, step_size=150, gamma=0.1)
    optimizer = optim.Adam(parameters=student_model.parameters(), learning_rate=scheduler, weight_decay=5e-4)

    # --- 训练循环 ---
    best_acc = 0.0
    best_epoch = 0

    save_student_path = save_path / 'student_epochs_paddle'
    for epoch in range(args.epochs):
        logger.info(f'Epoch: {epoch + 1}, Current LR: {optimizer.get_lr():.6f}')

        train_acc = train(student_model, teacher_models, optimizer, train_loader, args, catogaries)
        test_acc = evaluate(student_model, test_loader, args)

        # 在每个 epoch 后手动调用 scheduler.step()
        scheduler.step()

        save_checkpoint(
            {'epoch': epoch + 1, 'state_dict': student_model.state_dict()},
            checkpoint=save_student_path,
            name="last",
        )

        if test_acc >= best_acc:
            logger.info(f"New best accuracy: {test_acc:.5f} (previously {best_acc:.5f})")
            best_acc = test_acc
            best_epoch = epoch + 1
            save_checkpoint(
                {'epoch': epoch + 1, 'state_dict': student_model.state_dict()},
                checkpoint=save_student_path,
                name='bset',
            )

        logger.info(f"best acc so far: {best_acc:.5f} at epoch: {best_epoch}\n")
