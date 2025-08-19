import time
import os
from pathlib import Path

import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, ops, context, Parameter
from mindspore import dtype as mstype
from mindspore import set_seed
from mindspore.common.initializer import initializer, Normal

import copy
from tools import logger

# context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


def flatten_grads(grads):
    res = []
    for grad in grads:
        if isinstance(grad, Tensor):
            res.append(grad)
        elif isinstance(grad, (list, tuple)):
            res.extend(flatten_grads(grad))
        elif isinstance(grad, dict):
            for key in grad:
                res.extend(flatten_grads(grad[key]))
        else:
            raise TypeError(f"Unsupported type for grad: {type(grad)}")
    return res


def to_pil_image(x: np.ndarray):
    """
    Convert a NumPy array to a PIL Image.
    This function assumes the input is in the format (C, H, W) or (H, W, C) and scales pixel values to [0, 255].
    """
    from PIL import Image

    # Ensure the input is a NumPy array
    if not isinstance(x, np.ndarray):
        raise TypeError("Input must be a NumPy array")

    # Check if the input is in (C, H, W) format and transpose to (H, W, C) if necessary
    if x.ndim == 3 and x.shape[0] == 1:  # Grayscale image
        x = x.squeeze(0)
    elif x.ndim == 3 and x.shape[0] == 3:  # RGB image
        x = np.transpose(x, (1, 2, 0))

    # Scale pixel values to [0, 255]
    x = np.clip(x * 255, 0, 255).astype(np.uint8)

    return Image.fromarray(x)


def DLA(model, gt_data, gt_label, update, device, num_classes, channel):
    model.set_train(False)
    # gt_label = gt_label.view((1, *gt_label.shape))
    dataset = '1'
    set_seed(57)
    np.random.seed(57)
    root_path = '.'
    # data_path = os.path.join(root_path, '../data').replace('\\', '/')
    save_path = os.path.join(root_path, 'results/DLG_%s' % dataset).replace('\\', '/')

    lr = 1.0
    Iteration = 3
    num_exp = 1
    if not os.path.exists('results'):
        os.mkdir('results')
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # train DLG and iDLG
    for idx_net in range(num_exp):
        net = model
        
        logger.info('running %d|%d experiment' % (idx_net, num_exp))

        for method in ['DLG']:
            logger.info('%s, Try to generate images' % (method))

            criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='none')

            def loss_fn(data, target):
                logits = net(data)
                loss = criterion(logits, target)
                return loss

            gt_data = ops.ExpandDims()(gt_data, 0)
            dy_dx = ops.grad(loss_fn, weights=net.trainable_params())(gt_data, gt_label)
            # original_dy_dx = [g for g in flatten_grads(dy_dx)]
            original_dy_dx = copy.deepcopy(dy_dx)

            # generate dummy data and label
            pat_1 = np.random.normal(size=(channel, 16, 16)).astype(np.float32)
            pat_2 = np.concatenate((pat_1, pat_1), axis=1)
            pat_4 = np.concatenate((pat_2, pat_2), axis=2)
            pat_4 = np.expand_dims(pat_4, axis=0)
            dummy_data = Parameter(ms.from_numpy(pat_4), requires_grad=True)

            adam_lr = 0.01
            optimizer = nn.Adam([dummy_data], learning_rate=adam_lr, weight_decay=0.0)

            history = []
            history_iters = []
            losses = []
            mses = []
            train_iters = []

            Iteration = 300
            logger.info('lr = {}, Iteration = {} '.format(adam_lr, Iteration))
            for iters in range(Iteration):
                # logger.info(f"Iteration {iters + 1}/{Iteration}")

                def calc_grad_diff(gx, gy):
                    if isinstance(gx, tuple) and isinstance(gy, tuple):
                        # 递归处理元组中的每个元素
                        return sum(calc_grad_diff(sub_gx, sub_gy) for sub_gx, sub_gy in zip(gx, gy))
                    else:
                        # 当元素是Tensor时，计算平方差的和
                        return ops.sum((gx - gy) ** 2)

                def get_grad_diff(dummy_data, label):
                    dummy_dy_dx = ops.grad(loss_fn, weights=net.trainable_params())(dummy_data, label)
                    grad_diff = 0
                    for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                        # grad_diff += ops.sum((gx - gy) ** 2)
                        grad_diff += calc_grad_diff(gx, gy)
                    return grad_diff

                (grad_diff), grads = ops.value_and_grad(get_grad_diff, grad_position=0)(dummy_data, gt_label)

                history.append(to_pil_image(dummy_data[0].asnumpy()))

                optimizer(tuple([grads]))

                current_loss = grad_diff.asnumpy()
                train_iters.append(iters)
                losses.append(current_loss)
                mse = float(ops.ReduceMean()((dummy_data - gt_data) ** 2).asnumpy())
                mses.append(mse)

                print_info = False
                if Iteration > 10:
                    if iters == ((Iteration // 10) * 10) or iters == 0:
                        print_info = True
                else:
                    print_info = True

                # if iters % 10 == 0 and iters != 0:
                if print_info:
                    current_time = str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))
                    logger.info(f"{current_time} {iters} loss = {current_loss:.8f}, mse = {mse:.8f}")
                    history_iters.append(iters)
            logger.info("DLA Training complete. loss = {:.8f}, mse = {:.8f}".format(np.mean(losses), np.mean(mses)))


    if np.isnan(mses[-1]) or mses[-1] >= mses[0] or mses[-1] > 1.0:
        return 1.0
    else:
        return mses[-1]
