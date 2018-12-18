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
        input = tf.reshape(input, [-1, image_w,image_h,1])
        hencode = self.encoder_model(input)
        hepsilon = tf.ones(shape=tf.shape(hencode), dtype=dtype) * .5
        yout, pout = DoublySN(hencode, hepsilon)
        # yout = tf.convert_to_tensor(yout,tf.float32)
        yout = tf.reshape(yout, [-1, dim_hidden])
        self.latent_code = yout
        # yout.set_shape([batch_size,dim_hidden])
        print(yout.shape)
        output = self.decoder_model(yout)
        output = tf.reshape(output, [-1, image_h, image_w])
        return output

    def encoder(self):
        input = keras.layers.Input([image_h,image_w,image_d])
        # x = keras.layers.Conv2D(z_dim//16, kernel_size=(5,5), strides=(2,2), padding='SAME')(input)
        x = keras.layers.Conv2D(16, kernel_size=(2,2), strides=(2,2), padding='SAME')(input) #(14,14)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LeakyReLU(0.2)(x)
        x = keras.layers.Conv2D(32, kernel_size=(2,2), strides=(2,2), padding='SAME')(x) #(7,7)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LeakyReLU(0.2)(x)
        x = keras.layers.Conv2D(64, kernel_size=(2,2), strides=(2,2), padding='SAME')(x) # (4,4)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LeakyReLU(0.2)(x)
        x = keras.layers.Conv2D(dim_hidden, kernel_size=(2,2), strides=(2,2), padding='SAME')(x) # (2 2 2)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LeakyReLU(0.2)(x)
        output = keras.layers.GlobalAveragePooling2D()(x) # (2)
        return keras.models.Model(input, output)

    def decoder(self):
        input = keras.layers.Input([dim_hidden])
        x = keras.layers.Dense(2*2*dim_hidden)(input) # 为了化成和(2 2 z_dim)一样的大小，相当于逆GlobalAveragePooling2D
        x = keras.layers.Reshape([2,2,dim_hidden])(x)
        x = keras.layers.Conv2DTranspose(64, kernel_size=(2,2), strides=(2,2), padding='SAME')(x)#(4,4)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.Conv2DTranspose(32, kernel_size=(2,2), strides=(2,2), padding='SAME')(x)#(8,8)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.Conv2DTranspose(8, kernel_size=(2,2), strides=(3,3), padding='SAME')(x)#(24,24)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.Conv2DTranspose(image_d, kernel_size=(5,5), strides=(1,1), padding='VALID')(x)#(28*28*1)
        output = keras.layers.Activation("relu")(x)
        return keras.models.Model(input, output)


class dataset:
    def __init__(self, data):
        self.dataset = tf.data.Dataset.from_tensor_slices(
            (data)).repeat().batch(batch_size).shuffle(buffer_size=128)
        self.iterator = self.dataset.make_initializable_iterator()
        self.next_element = self.iterator.get_next()


def train(train_model):
    vae = VAE_stoc_neuron()
    (train_images, train_labels), (test_images,
                                   test_labels) = keras.datasets.mnist.load_data()
    train_images = train_images/255.0 #图片归一化，加快训练
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
            all_results = os.listdir(results_path)
            all_results.sort()
            print(all_results)
            saver.restore(sess, save_path=tf.train.latest_checkpoint(
                results_path + '/' + all_results[-1] + '/save/'))
            print("原始图片的降维可视化分布")
            Comparison_of_Manifold_Learning_methods(train_images[:batch_size].reshape(-1,image_h*image_w),train_labels[:batch_size])
            
            img,latent_code = sess.run([xout,vae.latent_code], feed_dict={x: train_images[:batch_size]})
            print("latent code的降维可视化分布")
            Comparison_of_Manifold_Learning_methods(latent_code,train_labels[:batch_size])
            print("原始图片"+" >"*10)
            plt.figure(figsize=(60, 60))
            origin_img = np.hstack(train_images[:show_img_num])
            plt.imshow(origin_img, cmap=plt.cm.binary)
            plt.axis('off')
            plt.grid('off')
            plt.show()
            img = np.hstack(img[:show_img_num])
            print("重构图片"+" >"*10)
            plt.figure(figsize=(60, 60))
            plt.imshow(img, cmap=plt.cm.binary)
            plt.axis('off')
            plt.grid('off')
            plt.show()


if __name__ == "__main__":
    train(False)
#     trian(True)
