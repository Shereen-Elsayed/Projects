# -*- coding: utf-8 -*-

import tensorflow as tf
import sklearn.datasets
import matplotlib.pyplot as pyt
import numpy as np
import math as m

#=================================intializations======================
Pixels = tf.placeholder(tf.float32, [None, 4096])
Weights = tf.Variable(tf.zeros([4096, 40]))
bias = tf.Variable(tf.zeros([40]))
#======= Predict y and apply actiation function softmax as it is classification
y_Predicted = tf.nn.softmax(tf.matmul(Pixels, Weights) + bias)
tf.summary.histogram("predictions", y_Predicted)

y= tf.placeholder(tf.float32, [None,40])

#calculating multi class logistic loss=======================================
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y* tf.log(y_Predicted), reduction_indices=[1]))
tf.summary.scalar('Loss',cross_entropy)

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

#tf.summary.histogram('Gradient Decent',train_step)

#======= Evaluate================================================
correct_prediction = tf.equal(tf.argmax(y_Predicted,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy',accuracy)

merged = tf.summary.merge_all()

Ginit=tf.global_variables_initializer()

#======= Create session to run the code============================
Data=sklearn.datasets.fetch_olivetti_faces(data_home=None, shuffle=True, random_state=0, download_if_missing=True)
#pyt.imshow(Data.images[100])
#All_images=Data.data
#y=Data.target
#print('target  ==',y)

with tf.Session() as sess:
	train_writer = tf.summary.FileWriter("/media/Shereen/HieldshiemMasters/Semester1/DistributedDataAnalytics/Exercises/Ex5_Solution/Logistic_Reg/Summary", sess.graph)
	
	sess.run(Ginit)
	images=Data.data
	Train_ratio=len(images)*0.9
	Train=images[:m.floor(Train_ratio)]
	Test=images[m.floor(Train_ratio):]
	target=Data.target
	y_temp=np.zeros((400,40))
	for i in range (0,400):
	    y_temp[i][target[i]]=1
	Train_target=y_temp[:m.floor(Train_ratio)]
	Test_target=y_temp[m.floor(Train_ratio):]
	for i in range(800): 
	  _,error,summary=sess.run([train_step,cross_entropy,merged], feed_dict={Pixels: Train, y: Train_target})
	  train_writer.add_summary(summary,i)
	  #print('error for epoch',error)
  


#===============run session for evaluation=========================
	print(sess.run(accuracy, feed_dict={Pixels:Test , y: Test_target}))




