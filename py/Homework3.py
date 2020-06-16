#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)
imagSize = [28, 28]

########################################
#2. 신경망 모델 구성

#입력층, 은닉층 3개(각각 256개씩), 출력층
Input = imagSize[0] * imagSize[1]
Output = 10 # 0 ~ 9 의 10가지로 분류
Node = [Input, 256, 256, 256, Output]

X = tf.placeholder(tf.float32, [None, Input])
Y = tf.placeholder(tf.float32, [None, Output])
keep_prob = tf.placeholder(tf.float32)
batch_prob = tf.placeholder(tf.bool)

Count_Layer = len(Node)
Count_Edge = Count_Layer-1
Count_Hidden = Count_Layer-2

W = [] #가중치
b = [] #편향
for i in range(Count_Edge):
    W.append(tf.Variable(tf.random_normal([Node[i],Node[i+1]], stddev=0.01)))
    b.append(tf.Variable(tf.zeros([Node[i+1]])))

#가중치와 편향을 적용하는 함수
def Apply(m, i):
    return tf.add(tf.matmul(m,W[i]), b[i])

#활성화 함수(relu 적용) 
def Activation_Relu(m):
    return tf.nn.relu(m)

#과적합 방지(dropout)
def DropOut(m):
    return tf.nn.dropout(m, keep_prob)

def BatchNormalization(m):
    return tf.layers.batch_normalization(m, training=batch_prob)

#입력층
model = X

#은닉층 설정
for num in range(Count_Hidden):
    model = Apply(model, num)
    model = BatchNormalization(model)
    model = Activation_Relu(model)
    model = DropOut(model)

#츨력층 설정
model = Apply(model, Count_Edge-1)

#손실함수
cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=model))

#3.최적화 방법
#경사하강법(Gradient Descent Optimizer) 사용
optimizer = tf.train.GradientDescentOptimizer(learning_rate=2)

#최소로 하는 값을 찾도록 설정
train_op = optimizer.minimize(cost)


######################################## 
#신경망 모델 학습
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)

for epoch in range(15):
    total_cost = 0
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        _, cost_val = sess.run([train_op, cost], 
                               feed_dict={X : batch_xs, 
                                          Y : batch_ys, 
                                          keep_prob : 0.8, batch_prob : True})
        total_cost += cost_val
    
    print('Epoch:', '%04d' % (epoch + 1), 
          'Avg. cost=','{:.8f}'.format(total_cost/total_batch))

print('----------------------------------- 최적화 완료 -----------------------------------')

######################################## 
#결과 확인
prediction = tf.argmax(model,1)
target = tf.argmax(Y,1)
is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도:',sess.run(accuracy, 
                      feed_dict ={X : mnist.test.images, 
                                  Y : mnist.test.labels,
                                  keep_prob : 1, batch_prob : True} ))

######################################## 
#결과 확인 (matplot)
labels = sess.run(model,
                 feed_dict ={X : mnist.test.images, 
                             Y : mnist.test.labels,
                             keep_prob : 1, batch_prob : True})

for i in range(10):
    subplot = plt.figure().add_subplot(2, 5, i+1)
    subplot.set_xticks([])
    subplot.set_yticks([])
    subplot.set_title('%d' % np.argmax(labels[i]))
    subplot.imshow(mnist.test.images[i].reshape((imagSize[0],imagSize[1])), cmap=plt.cm.gray_r)

plt.show()


# In[ ]:




