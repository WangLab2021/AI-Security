import tensorflow as tf
import os
from tensorflow.keras import optimizers, losses
import numpy as np
import random
from pathlib import Path
from datetime import datetime

import models_tf.resnet20 as resnet20
import models_tf.resnet as resnet

from tools import *


def load_checkpoint(checkpoint_directory, model, optimizer=None):
    logger.info(f"Attempting to load checkpoint from directory: {checkpoint_directory}")
    if optimizer is not None:
        ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    else:
        ckpt = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(ckpt, checkpoint_directory, max_to_keep=5)
    if manager.latest_checkpoint:
        logger.info(f"Restoring from latest checkpoint: {manager.latest_checkpoint}")
        # .expect_partial() 允许在加载的权重和模型不完全匹配时也能成功恢复（例如，只恢复模型权重而不恢复优化器状态）。
        status = ckpt.restore(manager.latest_checkpoint).expect_partial()
        status.assert_existing_objects_matched()
    else:
        logger.info(f"No checkpoint found in '{checkpoint_directory}'. Initializing model from scratch.")
    return ckpt


def save_checkpoint(ckpt_manager):
    """使用 TensorFlow CheckpointManager 保存模型参数。"""
    path = ckpt_manager.save()
    logger.info(f"Checkpoint saved to {path}")


def loss_fn_kd(outputs, labels, teacher_outputs, args, categories, use_gpu):
    alpha = args.alpha
    T = args.temperature

    ce_loss_fn = losses.SparseCategoricalCrossentropy(from_logits=True)
    ce_loss = ce_loss_fn(labels, outputs)

    kd_loss = 0.0

    for i in range(len(categories)):
        category = categories[i]
        t_output = teacher_outputs[i]

        outputs = tf.nn.softmax(outputs / T, axis=1)
        index = tf.convert_to_tensor(list(map(int, category)), dtype=tf.int32)

        s_t_outputs = tf.gather(outputs, index, axis=1)
        sum_u_k = tf.reduce_sum(s_t_outputs, axis=1, keepdims=True)
        log_sum_u_k = tf.math.log(sum_u_k + 1e-10)
        u_l = tf.math.log(s_t_outputs)
        tmp_right = u_l - log_sum_u_k
        t_output = tf.nn.softmax(t_output / T, axis=1)
        res = tf.reduce_sum(tf.multiply(t_output, tmp_right)) * -1

        kd_loss += res

    ce_loss = 10 * ce_loss
    total_loss = kd_loss + ce_loss
    return total_loss, kd_loss, ce_loss


# @tf.function
def train_step(model, teacher_models, optimizer, loss_fn_kd, train_batch, labels_batch, args, categ, use_gpu=False):
    # 教师模型的前向传播
    output_teacher_batch = []
    for t_model in teacher_models:
        output_t_batch = t_model(train_batch, training=False)
        output_teacher_batch.append(output_t_batch)

    with tf.GradientTape() as tape:
        # 学生模型的前向传播
        output_batch = model(train_batch, training=True)

        loss, kd_loss, ce_loss = loss_fn_kd(output_batch, labels_batch, output_teacher_batch, args, categ, use_gpu)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # 计算准确率 (在 graph 模式下使用 TensorFlow 操作)
    # tf.argmax 返回 int64，需要和 labels_batch 的类型匹配
    predicted_labels = tf.argmax(output_batch, axis=1, output_type=labels_batch.dtype)
    num_correct = tf.reduce_sum(tf.cast(predicted_labels == labels_batch, dtype=tf.int32))

    # 返回损失和正确预测的数量，以便在循环外聚合
    return loss, kd_loss, ce_loss, num_correct


def train(model, teacher_models, optimizer, loss_fn_kd, dataloader, args, categ, use_gpu):
    meter = AverageMeter()

    count_n = 0
    teacher_model_eval = []
    num_sample = 0

    for t in teacher_models:
        teacher_model_eval.append(t)
    from tqdm import tqdm

    pbar = tqdm(dataloader, total=len(dataloader), desc="Evaluating", unit="batch")

    for train_batch, labels_batch in dataloader:
        # 调用编译好的 train_step 函数
        loss, kd_loss, ce_loss, num_correct = train_step(
            model, teacher_models, optimizer, loss_fn_kd, train_batch, labels_batch, args, categ, use_gpu
        )

        count_n += num_correct
        num_batch = tf.shape(labels_batch)[0]  # 获取当前批次大小
        num_sample += num_batch  # 获取当前批次大小

        pbar.set_postfix(
            loss="{} - {} - {}".format(
                round(loss.numpy().item(), 5), round(kd_loss.numpy().item(), 5), round(ce_loss.numpy().item(), 5)
            ),
            acc="{}({})".format(round(float(num_correct / num_batch), 2), round(float(count_n / num_sample), 2)),
            lr=round(optimizer.learning_rate.numpy(), 5),
        )
        pbar.update(1)  # 更新进度条
        meter.update(np.array([l.numpy().item() for l in [loss, kd_loss, ce_loss]]))
    pbar.close()
    # 计算最终准确率
    # .numpy() 用于从 Tensor 中获取数值
    acc = count_n / num_sample
    logger.info("training acc: {} loss: {}".format(round(acc, 5), meter.avg))

    return acc


def evaluate(model, dataloader, classes, args, use_gpu):
    count_n = 0
    num_sample = 0
    num_classes = len(classes)
    num_right = [0] * num_classes
    num_sample_classes = [0] * num_classes
    from tqdm import tqdm

    pbar = tqdm(dataloader, total=len(dataloader), desc="Evaluating", unit="batch")

    for data_batch, labels_batch in dataloader:
        output_batch = model(data_batch, training=False)
        output_batch_np = output_batch.numpy()
        labels_batch_np = labels_batch.numpy()

        # 核心的 NumPy 计算逻辑保持不变
        outputs = np.argmax(output_batch_np, axis=1)
        num_currect = np.sum(outputs == labels_batch_np)
        count_n += num_currect
        num_batch = labels_batch_np.shape[0]  # 获取当前批次大小
        num_sample += num_batch

        num_right_batch = []
        num_sample_batch = []
        for j in range(num_classes):
            num_right_class = np.count_nonzero((outputs == labels_batch_np) & (outputs == classes[j]))
            num_sample_class = np.sum(labels_batch_np == classes[j])
            num_right_batch.append(num_right_class)  # 使用 append 更常见，但 insert 也可以
            num_sample_batch.append(num_sample_class)

        num_right = [a + b for a, b in zip(num_right, num_right_batch)]
        num_sample_classes = [a + b for a, b in zip(num_sample_classes, num_sample_batch)]
        pbar.set_postfix(
            acc="{}({})".format(
                round(float(num_currect / num_batch), 2),
                round(float(count_n / num_sample), 2),
            ),
        )
        pbar.update(1)  # 更新进度条
    pbar.close()

    acc = count_n / num_sample if num_sample > 0 else 0.0
    acc_classes = [(a / b) if b > 0 else 0.0 for a, b in zip(num_right, num_sample_classes)]

    logger.info(
        "evaluation acc:{} \nevaluation acc of classes:{}".format(round(acc, 5), [round(x, 5) for x in acc_classes])
    )
    return acc


if __name__ == '__main__':
    args = get_train_student_argparser().parse_args()
    log_file = Path("./logs", Path(__file__).stem, datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".log")
    setup_logger(log_file=log_file)

    # TENSORFLOW-SPECIFIC: 设置设备
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

    # TENSORFLOW-SPECIFIC: 设置随机种子
    random.seed(230)
    np.random.seed(230)
    tf.random.set_seed(230)

    # --- 数据加载 ---
    # TENSORFLOW-SPECIFIC: 使用 tf.keras.datasets 加载数据
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    y_train, y_test = y_train.astype(np.int32), y_test.astype(np.int32)
    logger.info("Fashion MNIST dataset loaded")
    logger.info(f"Training samples: {len(x_train)}, Testing samples: {len(x_test)}")
    logger.info(f"Image shape: {x_train[0].shape}, Data range: [{np.min(x_train)}, {np.max(x_train)}]")

    # 定义预处理函数
    def preprocess(image, label):
        image = tf.expand_dims(image, axis=-1)  # 增加通道维度
        image = tf.cast(image, tf.float32) / 255.0
        image = (image - 0.1307) / 0.3081
        return image, label

    # TENSORFLOW-SPECIFIC: 使用 tf.data.Dataset 构建数据管道
    train_loader = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        # .shuffle(buffer_size=60000)
        .batch(args.batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    test_loader = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test))
        .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(args.batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    dummy_input = tf.zeros((1, 28, 28, 1), dtype=tf.float32)

    # loading teacher models
    teacher_model = []
    # model_1:134789 023689 245689 045679 023579
    # catogaries = ['134789', '023689', '245689', '045679', '023579']
    catogaries = args.classes
    logger.info(catogaries)
    teacher_models_name = ['ResNet20', 'ResNet32', 'ResNet44', 'ResNet56', 'ResNet56']
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    # --- 加载教师模型 ---
    save_path = Path(args.save_dir)
    if not save_path.exists():
        raise FileNotFoundError(f"Save directory {args.save_dir} does not exist.")
    teacher_models = []
    catogaries = args.classes
    logger.info(f"Categories for teachers: {catogaries}")
    teacher_models_name = ['ResNet20', 'ResNet32', 'ResNet44', 'ResNet56', 'ResNet56']
    model_map = {i: getattr(resnet20, name) for i, name in enumerate(teacher_models_name)}

    for index in range(len(teacher_models_name)):
        num_classes = len(catogaries[index])
        model = model_map[index](num_classes=num_classes, inp_channels=1)
        _ = model(dummy_input, training=False)  # 触发模型构建
        # model.build(input_shape=(None, 28, 28, 1))
        model.summary()
        ckpt_path = save_path / f'Teacher_{index}_even'  # 假设TF模型文件名后缀为_tf
        # TensorFlow CheckpointManager 会自动寻找目录中最新的 ckpt 文件
        load_checkpoint(str(ckpt_path), model)
        teacher_models.append(model)

    # --- 学生模型定义 ---
    num_stu_model_classes = len(classes)
    student_model = resnet.ResNet18(num_classes=num_stu_model_classes)  # 假设 ResNet18 接受 num_classes
    _ = student_model(dummy_input, training=True)  # 触发模型构建
    student_model.summary()
    # student_model.build(input_shape=(None, 28, 28, 1))

    # --- 优化器和学习率调度器 ---
    # TENSORFLOW-SPECIFIC: Keras 调度器作为参数传入优化器
    lr_schedule = optimizers.schedules.ExponentialDecay(
        initial_learning_rate=args.learning_rate,
        decay_steps=150 * len(train_loader),  # 150个epoch的步数
        decay_rate=0.1,
    )
    optimizer = optimizers.Adam(learning_rate=lr_schedule)

    # TENSORFLOW-SPECIFIC: 创建 CheckpointManager 来管理检查点保存
    # save_student_path = save_path / 'student_epochs_tf'
    ckpt = tf.train.Checkpoint(model=student_model, optimizer=optimizer)
    ckpt_manager_epoch = tf.train.CheckpointManager(ckpt, save_path / 'student_epochs_tf', max_to_keep=1)
    ckpt_manager_best = tf.train.CheckpointManager(ckpt, save_path / 'student_best_tf', max_to_keep=1)

    # --- 训练循环 ---
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(args.epochs):
        logger.info(f'Epoch: {epoch + 1}, Current LR: {optimizer.learning_rate:.6f}')

        train_acc = train(student_model, teacher_models, optimizer, loss_fn_kd, train_loader, args, catogaries, use_gpu)
        test_acc = evaluate(student_model, test_loader, classes, args, use_gpu)

        save_checkpoint(ckpt_manager_epoch)

        if test_acc >= best_acc:
            logger.info(f"New best accuracy: {test_acc:.5f} (previously {best_acc:.5f})")
            best_acc = test_acc
            best_epoch = epoch + 1
            save_checkpoint(ckpt_manager_best)

        logger.info("best acc: {} epochs: {}\n".format(best_acc, best_epoch))
