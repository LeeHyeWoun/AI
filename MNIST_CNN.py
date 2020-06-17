#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
import matplotlib.pyplot as plt

print('\n----------------------------------- 데이터 입력 -----------------------------------')
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

X = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

model = X

print('\n----------------------------------- 신경망 생성 -----------------------------------')
def Conv(m, fn):
    F = tf.Variable(tf.random_normal([3, 3, m.shape[3], fn], stddev=0.01))
    m = tf.nn.conv2d(m, F, strides=[1, 1, 1, 1], padding = 'SAME')
    print('#  Convolutional Layer   : %s' % m.shape)
    return m

def ReLu(m):
    return tf.nn.relu(m)

def Max_Pooling(m):
    m = tf.nn.max_pool(m, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding = 'SAME')
    print('#  Max Pooling           : %s' % m.shape)
    return m

def Affine(m, node):
    W = tf.Variable(tf.random_normal([m.shape[1], node], stddev=0.01))
    m = tf.matmul(m, W)
    return m

#Convolutional Layers
FN = [32, 64]
for x in range(len(FN)) :
    model = Conv(model, FN[x])
    model = ReLu(model)
    model = Max_Pooling(model)

#Fully Connected Layers
model = tf.reshape(model, [-1,model.shape[1] * model.shape[2] * model.shape[3]])
Node = [256, 256]
for x in range(len(Node)) :
    model = Affine(model, Node[x])
    model = ReLu(model)
    print('#  Fully Connected Layer : %s' % model.shape)

#Dropout
model = tf.nn.dropout(model, keep_prob)

#츨력층 설정
model = Affine(model, 10)
print('#  Output Layer          : %s' % model.shape)

#손실함수
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))

#최적화
optimizer_Adam = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
optimizer_PMSProb = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(cost)


print('\n####################################### Adam #######################################')
print('\n----------------------------------- 최적화 시작 -----------------------------------')
#신경망 모델 학습
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)

history_Adam = []
for epoch in range(15):
    total_cost = 0
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape(-1, 28, 28, 1)
        _, cost_val = sess.run([optimizer_Adam, cost], feed_dict={X : batch_xs, Y : batch_ys, keep_prob : 0.8})
        total_cost += cost_val
    
    history_Adam.append(total_cost/total_batch)
    print('Epoch:', '%04d' % (epoch + 1), 
          'Avg. cost=','{:.3f}'.format(total_cost/total_batch))

print('\n----------------------------------- 최적화 완료 -----------------------------------')
#결과 확인
is_correct = tf.equal(tf.argmax(model,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('Adam으로 최적화 할 때의 정확도:',sess.run(accuracy, 
                      feed_dict ={X : mnist.test.images.reshape(-1, 28, 28, 1), 
                                  Y : mnist.test.labels,
                                  keep_prob : 1}))


print('\n\n###################################### PMSProb ######################################')
print('\n----------------------------------- 최적화 시작 -----------------------------------')

sess.run(init)

history_PMSProb = []
for epoch in range(15):
    total_cost = 0
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape(-1, 28, 28, 1)
        _, cost_val = sess.run([optimizer_PMSProb, cost], feed_dict={X : batch_xs, Y : batch_ys, keep_prob : 0.8})
        total_cost += cost_val
    
    history_PMSProb.append(total_cost/total_batch)
    print('Epoch:', '%04d' % (epoch + 1), 
          'Avg. cost=','{:.3f}'.format(total_cost/total_batch))


print('\ㅜ\n----------------------------------- 최적화 완료 -----------------------------------')
is_correct = tf.equal(tf.argmax(model,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('PMSProb으로 최적화 할 때의 정확도:',sess.run(accuracy, 
                      feed_dict ={X : mnist.test.images.reshape(-1, 28, 28, 1), 
                                  Y : mnist.test.labels,
                                  keep_prob : 1}))


print('\n\n----------------------------------- 비용 그래프 -----------------------------------')
plt.plot(history_Adam, 'b-')
plt.plot(history_PMSProb, 'r-')
plt.legend(['Adam','PMSProb'])
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.ylim([0, 0.1])
plt.xlim([0, 14])
plt.show()


# In[ ]:




