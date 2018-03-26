#!/usr/bin/python3
import tensorflow as tf
import dataset as ds
import numpy as np

epochs = 10
batch_szie = 10
dataset = ds.DataSet(224)
in_dir = '/mydata/train/'
x_data,y_data = dataset.create_train_set(in_dir)


from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y = train_test_split(x_data,y_data,test_size=0.2)
print(type(train_x))
import tensorflow.contrib.slim.nets as nets
import tensorflow.contrib.slim as slim

with tf.Graph().as_default():
    img_holder = tf.placeholder(tf.float32, [None, 224, 224, 3])
    img_label  = tf.placeholder(tf.int32, [None])
    
    logits,_ = nets.vgg.vgg_16(inputs=img_holder,num_classes=2,is_training=True)
    
    train_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=img_label))
    optimizer = tf.train.AdamOptimizer(1e-3).minimize(train_loss)

    finish = 0
    j = 0
    start = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(epochs):
            for z in range(20000//10):
                if finish == 20000:
                    finish = 0
                    start = 0
                    j = 0
                    indices = np.random.permutation(train_x.shape[0])
                    train_x = train_x[indices]
                    train_y = train_y[indices]
                    continue
                else:
                    start = j * 10
                    finish = start + 10
                    j+=1
                    train_data = train_x[start:finish]
                    train_label = train_y[start:finish]
                    sess.run(optimizer,feed_dict={img_holder:train_data,img_label:train_label})
                    loss_val = sess.run(train_loss,feed_dict={img_holder:train_data,img_label:train_label})
                    print("loss========{:d}_{:d}th:{:f}".format(i,j,loss_val))
    
