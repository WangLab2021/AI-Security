from pathlib import Path
import time
import os
import numpy as np
import tensorflow as tf
from PIL import Image

from tools import logger


def DLA(model, gt_data, gt_label, update, device, num_classes, channel):
    logger.info("Running DLA with TensorFlow")
    # Set random seed for reproducibility
    tf.random.set_seed(57)
    np.random.seed(57)

    dataset = '1'
    root_path = '.'
    # data_path = os.path.join(root_path, '../data').replace('\\', '/')
    save_path = Path(root_path, 'results/DLG_%s' % dataset)
    save_path.mkdir(parents=True, exist_ok=True)
    # save_path = os.path.join(root_path, 'results/DLG_%s' % dataset).replace('\\', '/')

    lr = 1.0
    Iteration = 3
    num_exp = 1

    def tensor_to_pil(tensor):
        """Convert tensor to PIL Image"""
        # Ensure tensor is in [0, 1] range and convert to numpy
        tensor_np = tf.clip_by_value(tensor, 0, 1).numpy()
        # Convert to uint8
        tensor_np = (tensor_np * 255).astype(np.uint8)
        # If tensor has batch dimension, squeeze it
        if len(tensor_np.shape) == 4:
            tensor_np = tensor_np[0]
        # Convert from CHW to HWC format if needed
        if tensor_np.shape[0] == channel:
            tensor_np = np.transpose(tensor_np, (1, 2, 0))
        # Handle grayscale
        if tensor_np.shape[-1] == 1:
            tensor_np = tensor_np.squeeze(-1)
        return Image.fromarray(tensor_np)

    ''' train DLG '''
    for idx_net in range(num_exp):
        net = model
        logger.info('running %d|%d experiment' % (idx_net, num_exp))

        for method in ['DLG']:
            logger.info('%s, Try to generate images' % (method))
            # Cross entropy loss function
            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

            # Ensure gt_data has batch dimension
            if len(gt_data.shape) == 3:
                gt_data = tf.expand_dims(gt_data, axis=0)

            # Ensure gt_label is the right shape
            if len(gt_label.shape) == 0:
                gt_label = tf.expand_dims(gt_label, axis=0)

            # Compute original gradient
            with tf.GradientTape() as tape:
                out = net(gt_data, training=False)
                y = loss_fn(gt_label, out)

            # Get gradients with respect to model parameters
            original_dy_dx = tape.gradient(y, net.trainable_variables)

            # # Use provided update gradients instead of computed ones if available
            # if update is not None:
            #     original_dy_dx = update

            # Generate dummy data pattern
            # pat_1 = tf.random.uniform([channel, 16, 16])
            # pat_2 = tf.concat([pat_1, pat_1], axis=1)
            # pat_4 = tf.concat([pat_2, pat_2], axis=2)
            # pat_4 = tf.transpose(pat_4, perm=[1, 2, 0])  # Convert to HWC format
            # dummy_data = tf.expand_dims(pat_4, axis=0)
            pat_1 = np.random.normal(size=(channel, 16, 16)).astype(np.float32)
            pat_2 = np.concatenate((pat_1, pat_1), axis=1)
            pat_4 = np.concatenate((pat_2, pat_2), axis=2)
            pat_4 = np.expand_dims(pat_4, axis=0)
            pat_4 = np.transpose(pat_4, (0, 2, 3, 1))  # Convert to HWC format
            dummy_data = tf.convert_to_tensor(pat_4, dtype=tf.float32)

            # Convert to Variable for optimization
            dummy_data = tf.Variable(dummy_data, trainable=True)
            adam_lr = 0.01
            # L-BFGS optimizer (using Adam as TF doesn't have L-BFGS built-in)
            # For better L-BFGS approximation, you might want to use tensorflow-probability
            optimizer = tf.keras.optimizers.Adam(learning_rate=adam_lr)

            history = []
            history_iters = []
            losses = []
            mses = []
            train_iters = []

            logger.info('lr = {}, Iteration = {}'.format(adam_lr, Iteration))
            Iteration = 300  # Set to 300 iterations as per original code
            for iters in range(Iteration):
                # logger.info(f"Iteration {iters + 1}/{Iteration}")

                with tf.GradientTape(persistent=True) as tape:
                    # Forward pass with dummy data
                    pred = net(dummy_data, training=False)
                    dummy_loss = loss_fn(gt_label, pred)

                    # Compute gradients of dummy loss
                    dummy_dy_dx = tape.gradient(dummy_loss, net.trainable_variables)

                    # Compute gradient difference
                    grad_diff = tf.constant(0.0)
                    for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                        if gx is not None and gy is not None:
                            grad_diff += tf.reduce_sum(tf.square(gx - gy))

                # Compute gradients with respect to dummy_data
                grads = tape.gradient(grad_diff, dummy_data)

                # Apply gradients
                optimizer.apply_gradients([(grads, dummy_data)])

                # Record history
                history.append(tensor_to_pil(dummy_data[0]))

                current_loss = grad_diff.numpy()
                train_iters.append(iters)
                losses.append(current_loss)

                # Compute MSE
                mse = tf.reduce_mean(tf.square(dummy_data - gt_data)).numpy()
                mses.append(mse)

                print_info = False
                if Iteration > 10:
                    if iters == ((Iteration // 10) * 10) or iters == 0:
                        print_info = True
                else:
                    print_info = True

                # if iters % 10 == 0 and iters != 0:
                # logger.info(print_info)
                if print_info:
                    current_time = str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))
                    logger.info(f"{current_time} {iters} loss = {current_loss:.8f}, mse = {mse:.8f}")
                    history_iters.append(iters)
            logger.info("DLA Training complete. loss = {:.8f}, mse = {:.8f}".format(np.mean(losses), np.mean(mses)))

    logger.info("DLA Training complete.")
    # Return final MSE or 1.0 if optimization failed
    if np.isnan(mses[-1]) or mses[-1] >= mses[0] or mses[-1] > 1.0:
        return 1.0
    else:
        return mses[-1]
