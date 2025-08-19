import os
import random
from mindspore import nn, Tensor, context
import mindspore as ms
import mindspore.ops as ops
import mindspore.dataset.vision as vision
import mindspore.dataset as ds
import numpy as np
from datetime import datetime


from tools import *
import models_ms.resnet20 as resnet20
import models_ms.resnet as resnet

context.set_context(mode=context.GRAPH_MODE, device_target="CPU", save_graphs=False)


# 重写模型加载/保存函数
def load_checkpoint(checkpoint_path, model: nn.Cell, optimizer: nn.Optimizer = None):
    checkpoint_file = checkpoint_path if checkpoint_path.endswith('.ckpt') else checkpoint_path + '.ckpt'

    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"File doesn't exist: {checkpoint_file}")

    param_dict = ms.load_checkpoint(checkpoint_file)
    logger.info(f"Loaded checkpoint from {checkpoint_file}")
    ms.load_param_into_net(model, param_dict)

    if optimizer:
        opt_prefix = "optimizer."
        optim_state = {k[len(opt_prefix) :]: v for k, v in param_dict.items() if k.startswith(opt_prefix)}
        if optim_state:
            ms.load_param_into_net(optimizer, optim_state)
        else:
            logger.info("Warning: No optimizer state found in checkpoint, skipping optimizer loading.")

    return param_dict


def save_checkpoint(state: dict, checkpoint: str, name: str):
    filepath = os.path.join(checkpoint, name + '.ckpt')

    if not os.path.exists(checkpoint):
        logger.info(f"Checkpoint Directory does not exist! Making directory {checkpoint}")
        os.makedirs(checkpoint)
    model_params = state['state_dict']
    # MindSpore的save_checkpoint直接保存参数列表
    ms.save_checkpoint(model_params, filepath)
    logger.info(f"Checkpoint saved to {filepath}")


def loss_fn_kd(outputs: Tensor, labels: Tensor, teacher_outputs: list, params, categories: list):

    alpha = params.alpha
    T = params.temperature

    # Cross Entropy Loss
    outputs_sm = ops.softmax(outputs, axis=1)
    if outputs_sm is None or ms.ops.isnan(outputs_sm).any():
        with open('error_log.txt', 'a') as f:
            f.write("Softmax outputs are NaN or None, check your model outputs.\n{}\n{}".format(outputs, outputs_sm))
        raise ValueError("Softmax outputs are NaN or None, check your model outputs.")
    ce_loss = ops.cross_entropy(outputs_sm, labels.astype(ms.int64), reduction='mean')
    if ce_loss is None or ms.ops.isnan(ce_loss).any():
        pass
        raise ValueError("Cross entropy loss is NaN or None, check your model outputs and labels.")
    kd_loss = 0

    for i in range(len(categories)):
        category = categories[i]
        t_output = teacher_outputs[i]  # 教师模型的输出

        # 学生模型的输出进行 softmax 并缩放
        outputs = ops.softmax(outputs / T, axis=1)
        index = Tensor(list(map(int, category)), dtype=ms.int32)

        s_t_outputs = ops.index_select(outputs, 1, index)

        sum_u_k = ops.sum(s_t_outputs, dim=1, keepdim=True)  # shape: [batch_size, 1]
        log_sum_u_k = ops.log(sum_u_k)
        u_l = ops.log(s_t_outputs)  # 防止除零
        tmp_right = u_l - log_sum_u_k  # shape: [batch_size, len(category)]
        t_output = ops.softmax(t_output / T, axis=1)  # 教师模型输出进行 softmax 并缩放
        res = ops.mul(t_output, tmp_right)  # 计算知识蒸馏损失
        res = ops.sum(res) * -1  # 求和并取负值

        if ms.ops.isnan(res).any() or ms.ops.isinf(res).any():
            pass
            raise ValueError("Knowledge distillation loss is NaN or Inf, check your model outputs and labels.")

        kd_loss += res

    ce_loss = 10 * ce_loss
    total_loss = kd_loss + ce_loss
    return total_loss, kd_loss, ce_loss


def train(model, teacher_model, optimizer, loss_fn_kd, dataloader, args, categ):
    meter = AverageMeter()
    count_n = 0
    teacher_model_eval = []
    # 设置成训练模式
    model.set_train(True)
    num_sample = 0

    for t_model in teacher_model:
        teacher_model_eval.append(t_model.set_train(False))

    def train_forward_fn(data, label, t_logits):
        logits = model(data)
        loss, kd_loss, ce_loss = loss_fn_kd(logits, label, t_logits, args, categ)
        return loss, kd_loss, ce_loss, logits

    grad_fn = ms.value_and_grad(train_forward_fn, grad_position=None, weights=optimizer.parameters, has_aux=True)

    def train_step_fn(data, label, t_logits):
        (loss, kd_loss, ce_loss, logits), grads = grad_fn(data, label, t_logits)
        loss = ops.depend(loss, optimizer(grads))
        return loss, kd_loss, ce_loss, logits

    from tqdm import tqdm

    pbar = tqdm(dataloader, total=len(dataloader), desc="Training", unit="batch")

    for i, (train_batch, labels_batch) in enumerate(pbar):
        # 在MindSpore中数据已经自动转换为Tensor，不需要Variable

        output_teacher_batch = []
        # 在MindSpore中使用no_grad需要手动设置
        for t_model in teacher_model_eval:
            output_t_batch = ops.stop_gradient(t_model(train_batch))
            output_teacher_batch.append(output_t_batch)

        # grad_fn = ms.value_and_grad(train_forward_fn, grad_position=None, weights=model.trainable_params())
        # (loss, output_batch), grads = grad_fn(train_batch, labels_batch, output_teacher_batch)
        # optimizer(grads)
        loss, kd_loss, ce_loss, output_batch = train_step_fn(train_batch, labels_batch, output_teacher_batch)

        # accuracy calculation
        output_batch = output_batch.asnumpy()
        labels_batch = labels_batch.asnumpy()

        outputs = np.argmax(output_batch, axis=1)
        num_right = np.sum(outputs == labels_batch)
        num_batch = labels_batch.shape[0]  # 获取当前批次大小
        count_n += num_right  # 一个batch中预测输出与gt label一致的次数
        b = args.batch_size

        # logger.info(f"Batch {i + 1}/{len(dataloader)}: {num_right} correct out of {num_batch} samples")

        num_sample += len(labels_batch)

        pbar.set_postfix(
            loss="{} - {} - {}".format(
                round(loss.asnumpy().item(), 5), round(kd_loss.asnumpy().item(), 5), round(ce_loss.asnumpy().item(), 5)
            ),  # 当前batch的损失值
            acc="{}({})".format(
                round(num_right / num_batch, 2), round(count_n / num_sample, 2)
            ),  # 当前batch的准确率和所有batch的预测正确率
            lr=round(optimizer.learning_rate.asnumpy().item(), 5),
        )
        pbar.update(1)
        meter.update(np.array([l.asnumpy().item() for l in [loss, kd_loss, ce_loss]]))

    pbar.close()

    acc = count_n / num_sample  # 所有batch的预测正确次数之和
    logger.info("training acc: {} loss: {}".format(round(acc, 5), meter.avg))

    return acc


def evaluate(model, dataloader, args):
    count_n = 0
    model.set_train(False)  # 设置为评估模式
    num_sample = 0
    num_right = [0] * 10
    num_sample_classes = [0] * 10
    classes = list(range(10))  # 假设是10分类任务

    from tqdm import tqdm

    pbar = tqdm(dataloader, total=len(dataloader), desc="Evaluating", unit="batch")

    for i, (data_batch, labels_batch) in enumerate(pbar):
        # MindSpore中数据已经是Tensor，不需要Variable和.cuda()
        output_batch = model(data_batch)

        # 转换为numpy计算指标
        output_batch = output_batch.asnumpy()
        labels_batch = labels_batch.asnumpy()

        outputs = np.argmax(output_batch, axis=1)
        count_n += np.sum(outputs == labels_batch)
        num_sample += len(labels_batch)

        num_right_batch = []
        num_sample_batch = []
        for j in range(len(classes)):
            # 计算每个类别的正确分类数和样本数
            num_right_class = np.count_nonzero((outputs == labels_batch) & (outputs == classes[j]))
            num_sample_class = np.sum(labels_batch == classes[j])
            num_right_batch.append(num_right_class)
            num_sample_batch.append(num_sample_class)

        # 累积所有batch的结果
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

    # 计算总体准确率和各类别准确率
    acc = count_n / num_sample
    acc_classes = [a / b if b != 0 else 0 for a, b in zip(num_right, num_sample_classes)]

    logger.info(
        "evaluation acc:{} \nevaluation acc of classes:{}".format(round(acc, 5), [round(x, 5) for x in acc_classes])
    )

    return acc


if __name__ == '__main__':

    args = get_train_student_argparser().parse_args()
    log_file = Path("./logs", Path(__file__).stem, datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".log")
    setup_logger(log_file=log_file)

    use_gpu = ms.context.get_context("device_target") == "GPU"
    ms.context.set_context(device_target='GPU' if use_gpu else 'CPU')
    logger.info(f"Using device: {'gpu' if use_gpu else 'cpu'}")

    # 设置随机种子
    random.seed(230)
    np.random.seed(230)
    ms.set_seed(230)

    transform_list = [vision.ToTensor(), vision.Normalize(mean=(0.1307,), std=(0.3081,), is_hwc=False)]

    dataset_path = Path('./data/MNIST_Data')
    train_set = ds.MnistDataset((dataset_path / 'train').as_posix(), shuffle=False)
    test_set = ds.MnistDataset((dataset_path / 'test').as_posix(), shuffle=False)
    # 获取训练集的第一个样本
    for spl in train_set.create_dict_iterator():
        image = spl['image']
        logger.info("Picture size: {}".format(image.shape))
        logger.info("Data range: {} - {}".format(image.min().asnumpy(), image.max().asnumpy()))
        break

    train_loader = ds.GeneratorDataset(train_set, ["image", "label"], shuffle=True)
    train_loader = train_loader.map(
        operations=transform_list, input_columns="image", num_parallel_workers=args.num_workers
    )
    train_loader = train_loader.batch(args.batch_size)

    test_loader = ds.GeneratorDataset(test_set, ["image", "label"], shuffle=False)
    test_loader = test_loader.map(
        operations=transform_list, input_columns="image", num_parallel_workers=args.num_workers
    )
    test_loader = test_loader.batch(args.batch_size)

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
        ckpt_path = save_path / f'Teacher_{index}_even'
        load_checkpoint(ckpt_path.as_posix(), model)
        teacher_models.append(model)

    # --- 学生模型定义 ---
    student_model = resnet.ResNet18(num_classes=10, inp_channels=1)
    # student_model = resnet20.ResNet20(num_classes=10, inp_channels=1)
    # student_model(Tensor(np.zeros((1, 1, 32, 32)), dtype=ms.float32))  # 构建模型，自动初始化参数
    pass

    # --- 优化器和学习率调度器 ---
    optimizer = nn.Adam(student_model.trainable_params(), learning_rate=args.learning_rate, weight_decay=5e-4)

    # --- 训练循环 ---
    best_acc = 0.0
    best_epoch = 0
    save_student_path = save_path / 'student_epochs_ms'
    for epoch in range(args.epochs):
        logger.info(f'Epoch: {epoch + 1}, Current LR: {float(optimizer.get_lr()):.6f}')

        train_acc = train(student_model, teacher_models, optimizer, loss_fn_kd, train_loader, args, catogaries)
        test_acc = evaluate(student_model, test_loader, args)

        # 在每个 epoch 后手动调用 scheduler.step()
        # scheduler.step()
        if epoch % 150 == 0 and epoch > 0:
            ops.assign(optimizer.learning_rate, optimizer.learning_rate * 0.1)

        save_checkpoint(
            {'epoch': ms.Parameter(epoch + 1), 'state_dict': student_model.state_dict()},
            checkpoint=save_student_path,
            name='last',
        )

        if test_acc >= best_acc:
            logger.info(f"New best accuracy: {test_acc:.5f} (previously {best_acc:.5f})")
            best_acc = test_acc
            best_epoch = epoch + 1
            save_checkpoint(
                {'epoch': ms.Parameter(epoch + 1), 'state_dict': student_model.state_dict()},
                checkpoint=save_student_path,
                name='bset',
            )

        logger.info(f"best acc so far: {best_acc:.5f} at epoch: {best_epoch}\n")
