import tensorflow as tf
import pickle
import os 
import numpy as np
import matplotlib.pyplot as plt
#from PIL import Image


#======= Tensorflow part==============================

SInput=tf.placeholder(tf.float32, [None, 32,32,3],name='Input')
SLabels=tf.placeholder(tf.int64, [None,1],name='Labels')
bool1=tf.placeholder(tf.bool,name='bool1')
probability = tf.get_variable("probability", dtype=tf.float32,initializer=0.2)
#settings of the CNN
bright=tf.map_fn(lambda img: tf.image.random_brightness(img,max_delta=64.0), SInput)
flip=tf.map_fn(lambda img: tf.image.random_flip_left_right(img), SInput)
rotate=tf.image.rot90(SInput,k=1)
Input=tf.cond(bool1,lambda: tf.concat([SInput,bright,flip,rotate], 0),lambda: SInput)
Labels=tf.cond(bool1,lambda: tf.concat([SLabels,SLabels,SLabels,SLabels], 0),lambda: SLabels)
# first layer convolution with ReLU activation
conv1 = tf.layers.conv2d(inputs=Input,filters=32,kernel_size=[3, 3],padding="same",activation=tf.nn.relu)
#apply max pooling on the output to downsample the output
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
norm1=tf.contrib.layers.batch_norm(inputs=pool1, center=True, scale=True, is_training=bool1)
conv2 = tf.layers.conv2d(inputs=norm1,filters=32,kernel_size=[3, 3],padding="same",activation=tf.nn.relu)
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
norm2=tf.contrib.layers.batch_norm(inputs=pool2, center=True, scale=True, is_training=bool1)
norm2_flat=tf.contrib.layers.flatten(norm2)
# Fully connected output layer
fc1=tf.layers.dense(inputs=norm2_flat, units=30,activation=tf.nn.relu)
fc1 = tf.layers.dropout(fc1, rate=probability, training=bool1)
fc2 = tf.layers.dense(inputs=fc1, units=20,activation=tf.nn.relu)
fc2 = tf.layers.dropout(fc2, rate=probability, training=bool1)
logits=tf.layers.dense(inputs=fc2, units=10)

# calculated the predictions using softmax
predictions = {"classes": tf.argmax(input=logits, axis=1),"probabilities": tf.nn.softmax(logits, name="softmax_tensor")}
#calculate loss between predicted and actual labels

loss = tf.losses.sparse_softmax_cross_entropy(labels=Labels, logits=logits)
tf.summary.scalar('Loss',loss)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
	optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001)
	#optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train_op = optimizer.minimize(loss=loss,global_step=tf.train.get_global_step())
merged = tf.summary.merge_all()
#=== calculated the accuracy=========================
#correct_prediction = tf.equal(predictions["classes"], Labels)
#evaluation=tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#evaluation=tf.metrics.accuracy(labels=Labels, predictions=tf.cast(predictions["classes"], tf.float32))
#=====================================================
Ginit=tf.global_variables_initializer()
counter=0
path='/media/Shereen/HieldshiemMasters/Semester1/DistributedDataAnalytics/Exercises/Ex6_Solution/Dataset/Training/'
path2='/media/Shereen/HieldshiemMasters/Semester1/DistributedDataAnalytics/Exercises/Ex6_Solution/Dataset/Test/test_batch'
batch_size=10000
NoOfEpochs=10
minibatch=100
def unpickle (file ) :
	with open ( file , 'rb' ) as fo :
		dict = pickle.load(fo , encoding ='bytes' )
	return dict
#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.2

with tf.Session() as sess:
	sess.run(Ginit)
	train_writer = tf.summary.FileWriter("/media/Shereen/HieldshiemMasters/Semester1/DistributedDataAnalytics/Exercises/Ex7_Solution/Summary_Opt_RMS", sess.graph)
	#loop over the training files
	for j in range(0,NoOfEpochs):
		counter=0
		Totalloss=0
		prediction_train_batch=np.asarray([],dtype=np.int32)
		labels_all=np.asarray([],dtype=np.int32)
		correct_train=0
		Loss_all=[]
		for filename in os.listdir(path):
			print('working on batch number :',counter)
			batch= unpickle (path+filename)
			counter=counter+1
			#reshape the data tp be images of 32*32*3sess.run(Ginit)
			data=np.asarray(batch[b'data'],dtype=np.float32).reshape(10000, 3, 32, 32).transpose(0,2,3,1)
			labels=np.asarray(batch[b'labels'],dtype=np.int32)
			labels=labels.reshape(10000,1)
			#print(data.shape,'  ', labels.shape)
			#==================================================
			batch_size=len(data)
			for i in range(int(batch_size/minibatch)):
				start=i*minibatch
				end=start+minibatch			
				imagerow=data[start:end]
				imagelabel=labels[start:end]
				bool2=True
			#=====================================================
				_,ls,pre,l=sess.run([train_op,loss,predictions["classes"],Labels], feed_dict={SInput:imagerow, SLabels:imagelabel,bool1:True}) 

				if i==0 and counter==1:
					labels_all=l
					prediction_train_batch=pre
				else:
					labels_all=np.concatenate((labels_all,l),axis=0)
					prediction_train_batch=np.concatenate((prediction_train_batch,pre),axis=0)

				Totalloss=Totalloss+ls
				Loss_all.append(Totalloss)

		LossAll = np.mean(Loss_all)
		summary = tf.Summary()
		summary.value.add(tag="Loss", simple_value=LossAll)
		train_writer.add_summary(summary,j)
		##calculate accuracy of training
		for z in range(0,len(labels_all)):
			if prediction_train_batch[z]==labels_all[z]:
				correct_train=correct_train+1
		Accuracy=(correct_train/len(prediction_train_batch))*100
		summaryTrain = tf.Summary()
		summaryTrain.value.add(tag="TrainAccuracy", simple_value=Accuracy)
		train_writer.add_summary(summaryTrain,j)
		print('total=',len(prediction_train_batch),'    correct=',correct_train)

		print((Totalloss/(5*(batch_size/minibatch))),' :   losss')
		#img = Image.fromarray(data[1], 'RGB')
		#img.save('my.png')
		#img.show()
		minibatch_test=1000
		correct=0
		batch_test= unpickle (path2)
		data_test=np.asarray(batch_test[b'data'],dtype=np.float32).reshape(10000, 3, 32, 32).transpose(0,2,3,1)
		labels_test=np.asarray(batch_test[b'labels'],dtype=np.int32)
		labels_test=labels_test.reshape(10000,1)
		prediction_test_batch=np.asarray([],dtype=np.int32)
		bool2=False
		for i in range(int(batch_size/minibatch_test)):
			start=i*minibatch_test
			end=start+minibatch_test				
			imagerow_test=data_test[start:end]
			imagelabel_test=labels_test[start:end] 
			prediction_per_batch=sess.run(predictions["classes"], feed_dict={Input:imagerow_test , Labels: imagelabel_test,bool1:False})
			if i==0:
				prediction_test_batch=prediction_per_batch
			else:
				prediction_test_batch=np.concatenate((prediction_test_batch,prediction_per_batch),axis=0)
			
		#print(labels_test[1:100])
		for z in range(0,len(labels_test)):
			if prediction_test_batch[z]==labels_test[z][0]:
				correct=correct+1
		#tf.summary.scalar('accuracy',correct)
		TestAccuracy=(correct/len(labels_test))*100
		summaryTest = tf.Summary()
		summaryTest.value.add(tag="TestAccuracy", simple_value=TestAccuracy)
		train_writer.add_summary(summaryTest,j)
		print('total=',len(prediction_test_batch),'    correct=',correct)



