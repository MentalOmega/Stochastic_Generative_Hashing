import datetime
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib import gridspec
from tensorflow import keras
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework import function
from setting import *
from Comparison_of_Manifold_Learning_methods import *


@function.Defun(dtype, dtype, dtype, dtype)
def DoublySNGrad(logits, epsilon, dprev, dpout):
    '''
    函数名的意思便是，连续的sign的梯度
    给定ξ（epsilon）便可用论文2.4 Reparametrization via Stochastic Neur中提到的方法来进行重参数的采样
    '''
    prob = 1.0 / (1 + tf.exp(-logits))
    yout = (tf.sign(prob - epsilon) + 1.0) / 2.0

    # unbiased
    dlogits = prob * (1 - prob) * (dprev + dpout)

    depsilon = dprev
    return dlogits, depsilon

    # 这里应该是使用了TensorFlow中的梯度重写


@function.Defun(dtype, dtype, grad_func=DoublySNGrad)
def DoublySN(logits, epsilon):
    prob = 1.0 / (1 + tf.exp(-logits))
    yout = (tf.sign(prob - epsilon) + 1.0) / 2.0
    return yout, prob


class VAE_stoc_neuron:
    def __init__(self):
        print("初始化自编码器")
        self.encoder_model = self.encoder()
        self.encoder_model.summary()
        self.decoder_model = self.decoder()
        self.decoder_model.summary()

    def cycle(self, input):
        input = tf.reshape(input, [-1, dim_input])
        hencode = self.encoder_model(input)
        hepsilon = tf.ones(shape=tf.shape(hencode), dtype=dtype) * .5
        yout, pout = DoublySN(hencode, hepsilon)
        # yout = tf.convert_to_tensor(yout,tf.float32)
        yout = tf.reshape(yout, [-1, dim_hidden])
        print(yout.shape)
        # yout.set_shape([batch_size,dim_hidden])
        output = self.decoder_model(yout)
        output = tf.reshape(output, [-1, image_h, image_w])
        return output

    def encoder(self):
        input = keras.layers.Input([dim_input])
        fc = keras.layers.Dense(dim_hidden)(input)
        output = fc
        return keras.models.Model(input, output)

    def decoder(self):
        input = keras.layers.Input([dim_hidden])
        fc = keras.layers.Dense(dim_input)(input)
        output = fc
        return keras.models.Model(input, output)


class dataset:
    def __init__(self, data):
        self.dataset = tf.data.Dataset.from_tensor_slices(
            (data)).repeat().batch(batch_size).shuffle(buffer_size=128)
        self.iterator = self.dataset.make_initializable_iterator()
        self.next_element = self.iterator.get_next()


def trian(train_model):
    vae = VAE_stoc_neuron()
    (train_images, train_labels), (test_images,
                                   test_labels) = keras.datasets.mnist.load_data()
    train_images = train_images/255.0
    test_images = test_images/255.0
    train_images_dataset = dataset(train_images)

    x = tf.placeholder(dtype, [batch_size, image_w, image_h])
    xout = vae.cycle(x)
    monitor = tf.nn.l2_loss(xout - x, name=None)
    loss = monitor

    optimizer = tf.train.AdamOptimizer(learning_rate)
    # optimizer = tf.train.RMSPropOptimizer(learning_rate)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)

    init = tf.global_variables_initializer()

    # Tensorboard visualization
    tf.summary.scalar(name='Cycle Loss', tensor=monitor)
    summary_op = tf.summary.merge_all()
    global_step = 0
    # Saving the model
    saver = tf.train.Saver()
    print("计算图构建完毕")
    with tf.Session() as sess:
        sess.run(init)
        print("变量初始化完毕")
        if train_model:
            if not os.path.exists(results_path + folder_name):
                os.mkdir(results_path + folder_name)
                os.mkdir(tensorboard_path)
                os.mkdir(save_path)
            sess.run(train_images_dataset.iterator.initializer)
            # writer = tf.summary.FileWriter(logdir=tensorboard_path)
            for epoch in range(n_epochs):
                n_batches = int(len(train_images) / batch_size)
                for _ in range(n_batches):
                    batch = sess.run(train_images_dataset.next_element)
                    sess.run([train_op], feed_dict={x: batch})
                    if _ % 50 == 0:
                        summary, cycle_loss = sess.run(
                            [summary_op, monitor], feed_dict={x: batch})
                        # writer.add_summary(summary,global_step)
                        print("Epoch: {}, iteration: {}".format(epoch, _))
                        print("Cycle loss: {}".format(cycle_loss))
                    global_step += 1
            print("保存路径:"+save_path)
            saver.save(sess, save_path=save_path, global_step=global_step)
        else:
            plt.figure(figsize=(60, 60))
            img = np.hstack(train_images[:show_img_num])
            plt.imshow(img, cmap=plt.cm.binary)
            plt.axis('off')
            plt.grid('off')
            plt.show()
            all_results = os.listdir(results_path)
            all_results.sort()
            print(all_results)
            saver.restore(sess, save_path=tf.train.latest_checkpoint(
                results_path + '/' + all_results[-1] + '/save/'))
            img = sess.run(xout, feed_dict={x: train_images[:batch_size]})
            Comparison_of_Manifold_Learning_methods(img,train_labels[:batch_size])
            img = np.hstack(img[:show_img_num])
            print(img.shape)
            plt.figure(figsize=(60, 60))
            plt.imshow(img, cmap=plt.cm.binary)
            plt.axis('off')
            plt.grid('off')
            plt.show()


if __name__ == "__main__":
    trian(False)
#     trian(True)
