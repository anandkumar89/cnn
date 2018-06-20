#!/home/anand/training/.env/bin/python3
import tensorflow as tf
from pandas import read_csv

data = read_csv('data/birth_life_2010.txt', sep='\t')

x = data['Birth rate']
y = data['Life expectancy']

X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')

# weight and bias, initialized to 0
w = tf.get_variable('weights', initializer=tf.constant(0.0))
b = tf.get_variable('bias', initializer=tf.constant(0.0))

# model 
Yhat = w*X + b

# loss 
loss = tf.square(Y-Yhat, name='loss')

# optimizer 
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	
	for i in range(100):
		for x_, y_ in zip(x, y):
			sess.run(optimizer, feed_dict={X:x_, Y:y_})

	w_out, b_out = sess.run([w, b])
	
	print('coefficients: w = %lf, b = %lf'%(w_out, b_out))
	
	

writer.close()
