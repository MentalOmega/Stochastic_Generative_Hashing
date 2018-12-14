import tensorflow as tf
import time
dim_input = 28 * 28
dim_hidden= 64
batch_size = 500
learning_rate = 1e-2
max_iter = 5000
image_h = 28
image_w = 28

alpha = 1e-3
beta = 1e-3

dtype = tf.float32

results_path = r"./result"
folder_name = time.strftime("%Y-%m-%d %Hh%Mm%Ss", time.localtime())
tensorboard_path = results_path + r"/{}/tensorboard".format(folder_name)
save_path = results_path + r"/{}/save".format(folder_name)

n_epochs = 10