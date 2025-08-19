import time
import os
import numpy as np
import paddle
import paddle.nn as nn
from PIL import Image  # 引入 PIL.Image 用于替代 transforms.ToPILImage
from tools import logger


def tensor_to_pil(tensor):
    """
    手动实现将 (C, H, W) 格式的 Paddle Tensor 转换为 PIL Image 的功能。
    这是 `torchvision.transforms.ToPILImage` 的替代方案。
    """
    # 1. 将 Paddle Tensor 转换为 NumPy array
    # .numpy() 会自动将数据从 GPU 移至 CPU
    arr = tensor.numpy()

    # 2. 将数据范围从 [0, 1] 缩放到 [0, 255] 并转为整数
    arr = (arr.clip(0, 1) * 255).astype('uint8')

    # 3. 转换轴的顺序，从 (C, H, W) 变为 (H, W, C) 以满足 PIL 的要求
    if arr.shape[0] == 1:  # 处理灰度图
        arr = arr.squeeze(0)  # 从 (1, H, W) 变为 (H, W)
        return Image.fromarray(arr, 'L')
    else:  # 处理彩色图
        arr = np.transpose(arr, (1, 2, 0))
        return Image.fromarray(arr, 'RGB')


def DLA(model, gt_data, gt_label, update, device, num_classes, channel):
    """
    DLA 函数的 PaddlePaddle 实现。
    """
    # paddle.set_device(device) # 在函数外部或开始时设置全局设备

    model.eval()

    # 注意: 在 PyTorch 中，gt_label.view(1, *gt_label.size()) 只是创建了一个视图但没有赋值，
    # 实际上没有效果。因此在 Paddle 中可以直接忽略这一行。

    dataset = '1'
    paddle.seed(57)  # 对应 torch.manual_seed
    np.random.seed(57)  # 确保 NumPy 随机数生成器也被设置

    root_path = '.'
    data_path = os.path.join(root_path, '../data').replace('\\', '/')
    save_path = os.path.join(root_path, 'results/DLG_%s' % dataset).replace('\\', '/')

    lr = 1.0
    Iteration = 3
    num_exp = 1

    # 根据你的要求，使用自定义函数替代 ToPILImage
    # tp = transforms.Compose([transforms.ToPILImage()])

    if not os.path.exists('results'):
        os.mkdir('results')
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    ''' train DLG and iDLG '''
    for idx_net in range(num_exp):
        net = model
        logger.info('running %d|%d experiment' % (idx_net, num_exp))

        for method in ['DLG']:
            logger.info('%s, Try to generate images' % (method))

            criterion = nn.CrossEntropyLoss()

            # --- 计算原始梯度 ---
            # 关键改动: torch.unsqueeze(..., dim=0) -> paddle.unsqueeze(..., axis=0)
            gt_data = paddle.unsqueeze(gt_data, axis=0)
            out = net(gt_data)
            y = criterion(out, gt_label)

            # 关键改动: torch.autograd.grad -> paddle.grad (API名称和基本用法相同)
            dy_dx = paddle.grad(y, net.parameters())
            original_dy_dx = [_.detach().clone() for _ in dy_dx]

            # --- 生成虚拟数据和标签 ---
            # pat_1 = paddle.rand([channel, 16, 16])
            # # 关键改动: torch.cat(..., dim=1) -> paddle.concat(..., axis=1)
            # pat_2 = paddle.concat([pat_1, pat_1], axis=1)
            # pat_4 = paddle.concat([pat_2, pat_2], axis=2)

            # dummy_data = paddle.unsqueeze(pat_4, axis=0)
            pat_1 = np.random.normal(size=(channel, 16, 16)).astype(np.float32)
            pat_2 = np.concatenate((pat_1, pat_1), axis=1)
            pat_4 = np.concatenate((pat_2, pat_2), axis=2)
            pat_4 = np.expand_dims(pat_4, axis=0)
            # pat_4 = np.transpose(pat_4, (0, 2, 3, 1))  # Convert to HWC format
            dummy_data = paddle.to_tensor(pat_4, dtype='float32', stop_gradient=False)
            # 关键改动: .requires_grad_(True) -> .stop_gradient = False
            # dummy_data.stop_gradient = False

            # 关键改动: Optimizer 参数顺序不同！
            # Torch: optim.LBFGS([dummy_data,], lr=lr)
            # Paddle: optimizer.LBFGS(learning_rate=lr, parameters=[dummy_data])
            optimizer = paddle.optimizer.LBFGS(learning_rate=lr, parameters=[dummy_data], max_iter=300)

            history = []
            losses = []
            mses = []
            train_iters = []
            logger.info('lr = {}, Iteration = {} '.format(lr, Iteration))

            for iters in range(Iteration):
                # logger.info(f"Iteration {iters + 1}/{Iteration}")

                def closure():
                    # 关键改动: optimizer.zero_grad() -> optimizer.clear_grad()
                    optimizer.clear_grad()

                    pred = net(dummy_data)
                    dummy_loss = criterion(pred, gt_label)

                    # create_graph=True 在 paddle.grad 中同样支持
                    dummy_dy_dx = paddle.grad(dummy_loss, net.parameters(), create_graph=False)

                    original_dy_dx2 = [update[i] for i in range(len(original_dy_dx))]

                    grad_diff = 0
                    for gx, gy in zip(dummy_dy_dx, original_dy_dx2):
                        grad_diff += paddle.sum((gx - gy) ** 2)  # 建议使用 paddle.sum

                    grad_diff.backward()
                    return grad_diff

                # 使用我们自定义的 tensor_to_pil 函数
                history.append(tensor_to_pil(dummy_data[0].detach().clone()))

                # optimizer.step(closure) 的用法在两个框架中对于 LBFGS 是相同的
                optimizer.step(closure)

                # .item() 的用法相同
                current_loss = closure().item()
                train_iters.append(iters)
                losses.append(current_loss)

                # 关键改动: torch.mean -> paddle.mean
                mse = paddle.mean((dummy_data - gt_data) ** 2).item()
                mses.append(mse)

                if iters % 10 == 0 and iters != 0:
                    current_time = str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))
                    logger.info(f"{current_time} {iters} loss = {current_loss:.8f}, mse = {mse:.8f}")
            logger.info("DLA Training complete. loss = {:.8f}, mse = {:.8f}".format(np.mean(losses), np.mean(mses)))

    logger.info("DLA Training complete.")
    if np.isnan(mses[-1]) or mses[-1] >= mses[0] or mses[-1] > 1.0:
        return 1.0
    else:
        return mses[-1]
