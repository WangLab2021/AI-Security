import os
import time
from collections import defaultdict
from pathlib import Path
from datetime import datetime

from tqdm import tqdm
from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import Sequence
from tensorflow.data import AUTOTUNE


import models_tf.resnet20 as resnet20
from tools import *


class SubDataset(Sequence):
    def __init__(self, indexes, whole_set_images, whole_set_labels, catogaries, transform=None):
        super().__init__()
        self.indexes = indexes
        self.whole_set_images = whole_set_images
        self.whole_set_labels = whole_set_labels
        self.transform = transform
        self.catogaries = catogaries

    def __getitem__(self, index: int):
        global_index = self.indexes[index]
        # TensorFlow 数据集通常将图像和标签分开处理
        image, label = self.whole_set_images[global_index], self.whole_set_labels[global_index]

        if self.transform is not None:
            image = self.transform(image)

        # label 本身就是整数，无需 .item()
        return image, label_transfer(self.catogaries, label)

    def __len__(self):
        return len(self.indexes)


def save_checkpoint(ckpt, manager, is_best, filename='checkpoint.ckpt'):
    # 使用 tf.train.CheckpointManager 来保存检查点
    # ckpt.save(filename)
    manager.save()
    logger.info(f"Checkpoint saved to {os.path.dirname(filename)}")
    if is_best:
        # Keras 的 CheckpointManager 会自动处理 "best" 的情况
        # 你可以通过 monitor 'val_accuracy' 等指标来只保存最好的模型
        logger.info(f"New best model saved.")


def accuracy(output, target, g_catogaries, tc_idx=0, topk=(1,)):
    maxk = max(topk)
    batch_size = target.shape[0]

    # 使用 tf.math.top_k
    _, pred = tf.math.top_k(output, k=maxk, sorted=True)
    pred = tf.transpose(pred)

    # TensorFlow 中 target 通常已经是 int64 或 int32，可以直接比较
    # 如果需要，使用 tf.cast
    target_cast = tf.cast(target, pred.dtype)
    correct = tf.equal(pred, tf.reshape(target_cast, [1, -1]))

    y_true = target.numpy().flatten()
    y_pred = pred.numpy().flatten()

    report = (
        [label_transfer_inverse(g_catogaries[tc_idx], y) for y in y_true],
        [label_transfer_inverse(g_catogaries[tc_idx], y) for y in y_pred],
    )

    res = []
    for k in topk:
        # correct_k = tf.cast(correct[:k], 'float32').sum(0)
        correct_k = tf.reduce_sum(tf.cast(correct[:k], 'float32'))
        res.append(correct_k * (100.0 / batch_size))
    return res, report


# 使用 tf.function 装饰器以获得更好的性能
# @tf.function
def train_step(input_tensor, target, model, criterion, optimizer):
    with tf.GradientTape(persistent=True) as tape:
        output = model(input_tensor, training=True)
        loss = criterion(target, output)
        # 如果使用混合精度，需要处理 loss scaling
        # scaled_loss = optimizer.get_scaled_loss(loss)

    # gradients = tape.gradient(scaled_loss, model.trainable_variables)
    gradients = tape.gradient(loss, model.trainable_variables)
    # optimizer.apply_gradients(zip(optimizer.get_unscaled_gradients(gradients), model.trainable_variables))
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    del tape  # 清理梯度带以释放内存
    return output, loss


def train(args, train_loader, model, criterion, optimizer, epoch, g_catogaries, g_teacher_index, dataset_len):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # model.train() 在 TensorFlow 中通过 training=True 参数控制
    end = time.time()
    report_data = [[], []]

    pbar = tqdm(train_loader, total=dataset_len // args.batch_size + 1, desc=f"Training Epoch {epoch+1}", unit="batch")

    for i, (input_tensor, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        output, loss = train_step(input_tensor, target, model, criterion, optimizer)
        losses.update(loss.numpy(), input_tensor.shape[0])

        # 计算精度
        (prec1, *_), r = accuracy(output, target, g_catogaries, g_teacher_index)
        report_data[0].extend(r[0])
        report_data[1].extend(r[1])
        top1.update(prec1.numpy(), input_tensor.shape[0])

        batch_time.update(time.time() - end)
        end = time.time()

        pbar.set_postfix(
            loss="{}({})".format(round(float(losses.val), 5), round(float(losses.avg), 5)),
            top1="{}({})".format(round(float(top1.val), 2), round(float(top1.avg), 2)),
            lr=(
                optimizer.learning_rate.numpy()
                if hasattr(optimizer.learning_rate, 'numpy')
                else optimizer.learning_rate
            ),
        )
        pbar.update(1)

        if i % args.printfreq == 0:
            current_lr = optimizer.learning_rate
            if callable(current_lr):
                current_lr = current_lr(optimizer.iterations)

            # logger.info(
            #     'Epoch: [{0}][{1}/{2}]\t'
            #     'current lr {lr:.5e}\t'
            #     'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #     'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            #     'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            #     'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
            #         epoch + 1,
            #         i + 1,
            #         # len(train_loader),
            #         60000,  # 假设每个 epoch 有 60000 个样本
            #         lr=current_lr,
            #         batch_time=batch_time,
            #         data_time=data_time,
            #         loss=losses,
            #         top1=top1,
            #     )
            # )
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
    #     logger.info(f'Training Epoch: {epoch+1}')
    #     logger.info(
    #         "=" * 70 + "\n", classification_report(report_data[0], report_data[1], zero_division=0), "=" * 70 + "\n"
    #     )


# 使用 tf.function 装饰器以获得更好的性能
@tf.function
def validate_step(input_tensor, target, model, criterion):
    output = model(input_tensor, training=False)
    loss = criterion(target, output)
    return output, loss


def validate(args, val_loader, model, criterion, epoch, g_catogaries, g_teacher_index, dataset_len):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # model.eval() 在 TensorFlow 中通过 training=False 参数控制
    end = time.time()
    report_data = [[], []]

    pbar = tqdm(val_loader, total=dataset_len // args.batch_size + 1, desc=f"Validating Epoch {epoch+1}", unit="batch")

    for i, (input_tensor, target) in enumerate(val_loader):
        output, loss = validate_step(input_tensor, target, model, criterion)

        (prec1, *_), r = accuracy(output, target, g_catogaries, g_teacher_index)
        report_data[0].extend(r[0])
        report_data[1].extend(r[1])

        losses.update(loss.numpy(), input_tensor.shape[0])
        top1.update(prec1.numpy(), input_tensor.shape[0])

        batch_time.update(time.time() - end)
        end = time.time()

        pbar.set_postfix(
            loss="{}({})".format(round(float(losses.val), 5), round(float(losses.avg), 5)),
            top1="{}({})".format(round(float(top1.val), 2), round(float(top1.avg), 2)),
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
    #     logger.info(f"Validate Epoch: {epoch+1}")
    #     logger.info("=" * 70 + "\n", classification_report(report_data[0], report_data[1], zero_division=0), "=" * 70)
    #     logger.info(f' * Prec@1 {top1.avg:.3f}\n\n')

    return top1.avg


def main(args):
    best_prec1 = 0

    # 设置设备
    use_gpu = args.use_gpu and bool(tf.config.list_physical_devices('GPU'))
    args.use_gpu = use_gpu
    if use_gpu:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                logger.info(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                logger.info(e)
            strategy = tf.distribute.MirroredStrategy()
            logger.info(f'Using device: GPU with {strategy.num_replicas_in_sync} replicas')
        else:
            use_gpu = args.use_gpu = False
            logger.warning("No GPU found, falling back to CPU.")
    if not use_gpu:
        strategy = tf.distribute.get_strategy()  # 默认策略 (CPU)
        logger.info("Using device: CPU")

    # 混合精度设置
    if args.half:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    catogaries = args.classes
    logger.info(str(catogaries))
    # 使用 tf.keras.datasets 加载数据
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    # Keras 数据集返回 numpy 数组，标签是扁平的
    y_train, y_test = y_train.flatten(), y_test.flatten()

    tmp_image = next(iter(x_train))
    image_shape = tmp_image.shape
    logger.info(f"Sample image shape: {image_shape} - {tmp_image.dtype} - {tmp_image.min()} ~ {tmp_image.max()}")
    
    train_index_dict = build_indexes(zip(x_train, map(int, y_train)))
    train_sets = split_data_by_classes(train_index_dict, catogaries, dataset_size=args.trainset)
    val_index_dict = build_indexes(zip(x_test, map(int, y_test)))
    val_sets = split_data_by_classes(val_index_dict, catogaries, mode='val', dataset_size=args.trainset)

    model_map = {
        0: resnet20.ResNet20,
        1: resnet20.ResNet32,
        2: resnet20.ResNet44,
        3: resnet20.ResNet56,
        4: resnet20.ResNet56,
    }
    teacher_models = [c.__name__ for c in model_map.values()]
    logger.info(f"Teacher Models: {teacher_models}")

    for teacher_index in range(len(catogaries)):
        logger.info(
            "\n".join(
                [
                    '=' * 20,
                    f"Training teacher {teacher_index} Model {teacher_models[teacher_index]}",
                    f"Catogaries : {catogaries[teacher_index]}",
                    '=' * 20,
                ]
            )
        )
        if teacher_index not in model_map:
            raise ValueError(f"Expect teacher index in {list(model_map)}, but got {teacher_index}")

        logger.info("Clearing previous session to avoid memory issues...")

        # 清理之前的会话以释放内存
        tf.keras.backend.clear_session()

        num_classes = len(catogaries[teacher_index])
        dummy_input = tf.zeros((1, 28, 28, 1), dtype=tf.float32)  # Fashion-MNIST 输入形状

        # 在分布式策略范围内创建模型和优化器
        with strategy.scope():

            model = model_map.get(teacher_index, resnet20.ResNet56)(num_classes=num_classes, inp_channels=1)
            _ = model(dummy_input, training=False)
            model.summary()
            criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

            if args.arch in ['ResNet1202', 'ResNet110']:
                logger.info(f"Change lr from {args.lr} to {args.lr *0.1}")

            lr = args.lr if args.arch not in ['ResNet1202', 'ResNet110'] else args.lr * 0.1
            lr_scheduler = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                boundaries=[
                    100 * len(train_sets[teacher_index]) // args.batch_size,
                    150 * len(train_sets[teacher_index]) // args.batch_size,
                ],
                values=[lr, lr * 0.1, lr * 0.01],
            )
            optimizer = tf.keras.optimizers.SGD(
                learning_rate=lr_scheduler, momentum=args.momentum, weight_decay=args.weight_decay
            )
            if args.half:
                optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

        # 定义数据预处理
        def preprocess(image, label):
            image = tf.cast(image, tf.float32) / 255.0
            image = (image - 0.1307) / 0.3081
            # TensorFlow 需要通道维度
            image = tf.expand_dims(image, axis=-1)
            return image, label

        current_train_set_indices = train_sets[teacher_index]
        train_dataset_len = len(current_train_set_indices)
        train_dataset_gen = SubDataset(current_train_set_indices, x_train, y_train, catogaries[teacher_index])
        train_dataset_len = len(train_dataset_gen)

        train_loader = tf.data.Dataset.from_generator(
            lambda: train_dataset_gen,
            output_signature=(tf.TensorSpec(shape=(28, 28), dtype=tf.float32), tf.TensorSpec(shape=(), dtype=tf.int32)),
        )
        train_loader = (
            train_loader.map(preprocess, num_parallel_calls=AUTOTUNE)
            .shuffle(buffer_size=10000)
            .batch(args.batch_size)
            .prefetch(buffer_size=AUTOTUNE)
        )
        tmp_image = np.array(next(iter(train_loader))[0])
        logger.info(
            f"Sample train image shape: {tmp_image.shape} - {tmp_image.dtype} - {tmp_image.min()} ~ {tmp_image.max()}"
        )

        current_val_set_indices = val_sets[teacher_index]
        val_dataset_gen = SubDataset(current_val_set_indices, x_test, y_test, catogaries[teacher_index])
        val_dataset_len = len(val_dataset_gen)

        val_loader = tf.data.Dataset.from_generator(
            lambda: val_dataset_gen,
            output_signature=(tf.TensorSpec(shape=(28, 28), dtype=tf.float32), tf.TensorSpec(shape=(), dtype=tf.int32)),
        )
        val_loader = (
            val_loader.map(preprocess, num_parallel_calls=AUTOTUNE)
            .batch(args.batch_size)
            .prefetch(buffer_size=AUTOTUNE)
        )
        logger.info(f"Dataset size: Train {train_dataset_len} Val {val_dataset_len}")

        # 设置 Checkpoint
        save_path = Path(args.save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
        manager = tf.train.CheckpointManager(
            ckpt,
            directory=(save_path / f'Teacher_{teacher_index}_even').absolute().as_posix(),
            # checkpoint_name=f'',
            max_to_keep=1,
        )

        if args.evaluate:
            # 如果需要，可以从 checkpoint 恢复模型
            # ckpt.restore(manager.latest_checkpoint)
            validate(args, val_loader, model, criterion, 0, catogaries, teacher_index, val_dataset_len)
            return

        for epoch in range(args.start_epoch, args.epochs):
            train(args, train_loader, model, criterion, optimizer, epoch, catogaries, teacher_index, train_dataset_len)
            prec1 = validate(args, val_loader, model, criterion, epoch, catogaries, teacher_index, val_dataset_len)

            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)

            # CheckpointManager 会处理保存逻辑
            save_checkpoint(ckpt, manager, is_best, filename=save_path.absolute().as_posix())

        logger.success("done!!!!!!!!!!!!!!!!!!!!!")


if __name__ == '__main__':
    log_file = Path("./logs", Path(__file__).stem, datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".log")
    setup_logger(log_file=log_file)

    logger.info("Training Teacher [ensemble] with TensorFlow")
    logger.info('--------args----------')
    # 确保 get_train_teacher_argparser 返回的参数与 TensorFlow 代码兼容
    args = get_train_teacher_argparser(
        model_names=["ResNet20", "ResNet32", "ResNet44", "ResNet56", "ResNet110"]
    ).parse_args()
    # 为 TF 添加 momentum 参数
    if not hasattr(args, 'momentum'):
        args.momentum = 0.9  # SGD 默认值
    for k, v in vars(args).items():
        logger.info(f'{k}: {v}')
    logger.info('----------------------')
    main(args)
