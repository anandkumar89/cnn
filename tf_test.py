import numpy as np
import tensorflow as tf

# constants

sess = tf.Session()

a = tf.constant(10, name='const_a')
b = tf.constant(20, name='const_b')


a = a + 1

print(sess.run(a))

c_1 = a + b

a = tf.random_uniform(shape=(3,), name='const_rand_a')
b = tf.random_uniform(shape=(3,), name='const_rand_b')
c_2 = a + b



print(sess.run(c_1))
print(sess.run(c_2))
print('Note different values of random constant with different run')
print(sess.run(c_2))



