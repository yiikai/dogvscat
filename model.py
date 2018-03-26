#!/usr/bin/python3
import tensorflow as tf
import matplotlib.image as mpimg # mpimg 用于读取图片
import cv2 # 机器视觉库，安装请用pip3 install opencv-python
import numpy as np # 数值计算库
import os # 系统库
from random import shuffle # 随机数据库 
from tqdm import tqdm # 输出进度库
import matplotlib.pyplot as plt # 常用画图库
import tensorflow as tf
from scipy import misc

train_dir = './train'
img_size = 50
batch_size = 100
epochs = 20

def getlabel(img):
    word_label = img.split('.')[0]
    if word_label == 'dog':
        label = 0
    else:
        label = 1
    return label


def create_train_set(dir):
    training_data = []
    training_label = []
    for img in tqdm(os.listdir(train_dir)):
        y_ = getlabel(img)
        path = os.path.join(train_dir, img)
        img = cv2.imread(path)  # 读入RGB
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        x_ = cv2.resize(img,(img_size,img_size),interpolation=cv2.INTER_CUBIC)  # 将图片变成统一大小
        #print(x_.shape)
        training_data.append(x_)
        training_label.append(y_)
    return training_data,training_label


from sklearn.model_selection import train_test_split
x_data,y_data = create_train_set(train_dir)
print(y_data[0:2])
sess = tf.Session()
#y_data = sess.run(tf.one_hot(y_data,2,dtype=tf.float32))

train_x,test_x,train_y,test_y = train_test_split(x_data,y_data,test_size=0.2)
print(len(train_x))

train_x = np.array(train_x).reshape(-1,img_size,img_size,3)
test_x = np.array(test_x).reshape(-1,img_size,img_size,3)
train_y = np.array(train_y)
test_y = np.array(test_y)
print(train_x.shape)
print(test_x.shape)
print(test_y.shape)

def create_weight(shape):
    var = tf.Variable(tf.truncated_normal(shape))
    return var

img_hodler = tf.placeholder(tf.float32, [None, img_size, img_size, 3])
img_lable = tf.placeholder(tf.int32, [None])
dropout = tf.placeholder(tf.float32)

def create_conv_weight_bias(f_w,f_h,f_c,filters):
    var = tf.Variable(tf.truncated_normal([f_w,f_h,f_c,filters],stddev = 0.1,dtype=tf.float32))
    bias = tf.Variable(tf.zeros([filters],dtype=tf.float32))
    return var,bias

def create_weight_bias(w,h):
    weight = tf.Variable(tf.truncated_normal([w,h],stddev=0.1,dtype=tf.float32))
    bias = tf.Variable(tf.zeros([h],dtype=tf.float32))
    return weight,bias

conv_weight1,conv_bias1 = create_conv_weight_bias(2,2,3,64)
conv_weight2,conv_bias2 = create_conv_weight_bias(2,2,64,32)

def create_conv_net(input_data):
    conv1 = tf.nn.conv2d(input_data,conv_weight1,strides=[1,1,1,1],padding="SAME")
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1,conv_bias1))
    max_pool1 = tf.nn.max_pool(relu1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

    conv2 = tf.nn.conv2d(max_pool1,conv_weight2,strides=[1,1,1,1],padding="SAME")
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2,conv_bias2))
    max_pool2 = tf.nn.max_pool(relu2,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
    print("max_pool2 shape: ",max_pool2.shape)
    final_conv_shape = tf.contrib.layers.flatten(max_pool2) 
    print("max_pool2 flaten shape: ",final_conv_shape.shape)

    output_input_size1 = final_conv_shape.shape[1].value
    full_weight1,full_bias1 = create_weight_bias(output_input_size1,200)
    full_weight1 = tf.nn.dropout(full_weight1,dropout)
    full1 = tf.nn.relu(tf.matmul(final_conv_shape,full_weight1) + full_bias1)
    print("full1 shape: ",full1.shape)
    output_input_size2 = full1.shape[1].value
    full_weight2,full_bias2 = create_weight_bias(output_input_size2,2)
    full_weight2 = tf.nn.dropout(full_weight2,dropout)
    full2 = tf.nn.relu(tf.matmul(full1,full_weight2) + full_bias2)
    print("full2 shape: ",full2.shape)
    output_input_size3 = full2.shape[1].value
    full_weight3,full_bias3 = create_weight_bias(output_input_size2,2)
    full3 = tf.matmul(full1,full_weight3) + full_bias3
    print("full2 shape: ",full3.shape)
    return full3

def create_loss(output,labels):
    loss_func = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=output)
    loss = tf.reduce_mean(loss_func)
    return loss

def create_opter(loss,learning_rate):
    opt = tf.train.AdamOptimizer(learning_rate)
    opt_op = opt.minimize(loss)
    return opt_op


logit = create_conv_net(img_hodler)
loss = create_loss(logit,img_lable)
top_k_op = tf.nn.in_top_k(logit, img_lable, 1)
opter = create_opter(loss,1e-3)
sess.run(tf.global_variables_initializer())
true_count = 0
step = 0
total_sample_count = epochs * 20000
j = 0
finish = 0
precision_list = []
for i in range(epochs):
    for z in range(20000//batch_size):
        if finish == 20000:
            start = 0
            finish = 0
            j = 0
            indices = np.random.permutation(train_x.shape[0]) # shape[0]表示第0轴的长度，通常是训练数据的数量
            train_x = train_x[indices]
            train_y = train_y[indices] # data_y就是标记（label）
            break
        else:
            start = j * batch_size
            finish = start + batch_size
            j+=1
            print(start,finish)
            train_data = train_x[start:finish]
            print(train_data.shape)
            train_label = train_y[start:finish]
            sess.run(opter,feed_dict={img_hodler:train_data,img_lable:train_label,dropout:0.8})
            loss_val = sess.run(loss,feed_dict={img_hodler:train_data,img_lable:train_label,dropout:1.0})
            print("loss========{:d}th:{:f}".format(i,loss_val))
    predictions = sess.run([top_k_op], feed_dict = {img_hodler:train_data,img_lable:train_label,dropout:1.0})
    true_count += np.sum(predictions)
    precision = true_count / total_sample_count
    print('precision @ 1 = %.3f' % precision)
    precision_list.append(precision)

test_loss = sess.run(loss,feed_dict={img_hodler:test_x,img_lable:test_y,dropout:1.0})
print("valid loss: ",test_loss)
