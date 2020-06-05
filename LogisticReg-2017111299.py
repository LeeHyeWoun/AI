#!/usr/bin/env python
# coding: utf-8

# In[4]:


from PIL import Image
import tensorflow as tf
import numpy as np

size_bird = 500
size_dog = 500
size_etc = 500

######################################## 
# [털, 날개]
x_data = []
for i in  range(size_bird):
    x_data.append([1,1])
for i in  range(size_dog):
    x_data.append([1,0])
for i in  range(size_etc):
    x_data.append([0,0])

# [기타, 포유류, 조류] : one-hot
y_data =[]
for i in  range(size_bird):
    y_data.append([0,0,1])
for i in  range(size_dog):
    y_data.append([0,1,0])
for i in  range(size_etc):
    y_data.append([1,0,0])

######################################## 
#신경망 모델 구성

#입력층 : [1,1],[1,0],[0,0]
#은닉층1 -> node 10개
#은닉층2 -> node 20개
#은닉층3 -> node 10개
#출력층 : [0,0,1], [0,1,0], [1,0,0]

Node = [len(x_data[0]), 10, 20, 10, len(y_data[0])]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = [] #가중치
b = [] #편향
for i in range(len(Node)-1):
    W.append(tf.Variable(tf.random_uniform([Node[i],Node[i+1]],-1,1)))
    b.append(tf.Variable(tf.zeros([Node[i+1]])))

#입력층
value = X

#가중치 W와 편향 b 적용하는 함수
def Apply (num) :
    return tf.add(tf.matmul(value,W[num]), b[num])

#활성화 함수 적용 함수 : relu
def Activation (value):
    return tf.nn.relu(value)
    
#은닉층 설정
for i in range(len(Node)-2):
    value = Apply(i)
    value = Activation(value)

#츨력층 설정
model = Apply(len(Node)-2)

#손실함수
cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=model))

#최적화 함수 : 경사하강법 적용
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

#최소로 하는 값을 찾도록 설정
train_op = optimizer.minimize(cost)


######################################## 
#신경망 모델 학습
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for step in range(100):
    sess.run(train_op, feed_dict={X:x_data, Y:y_data})
    
    if(step + 1) % 10 == 0:
        print(step+1, sess.run(cost, feed_dict={X:x_data, Y:y_data}))

######################################## 
#결과 확인
prediction = tf.argmax(model,1)
target = tf.argmax(Y,1)
print('예측값:\n', sess.run(prediction, feed_dict={X:x_data}))
print('실제값:\n', sess.run(target, feed_dict = {Y:y_data}))

is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도: %.2f' % sess.run(accuracy * 100, feed_dict ={X:x_data, Y:y_data} ))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




