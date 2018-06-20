#!/home/anand/training/.env/bin/python3

import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# image = mpimg.imread('images/cat_pup.jpeg')
# print(image.shape)
# filter1 = tf.constant([[1,0,-1],[1,0,-1],[1,0,-1]], name='filter_vertical_edge')
# print(image[:,:,1])

'''
channel is 2-d matrix
image is 3-d matrix (28x28x1)
'''


### Read mnist data
from array import array
data = array('B')
label= array('B')

with open('MNIST_data/train-images-idx3-ubyte', 'rb') as f:
	data.fromfile(f, 28*28*60000+16)

with open('MNIST_data/train-labels-idx1-ubyte', 'rb') as f:
	label.fromfile(f, 60000+8)

# Parsing mnist data
data = np.array(data); data = data[16:]
data = np.reshape(data, (28, 28, 1, 60000),order='F')
label= np.array(label);label= label[8:]	

# Vizualization of parsed data
# a = np.block([[data[:,:,0,0].T,data[:,:,0,1].T,data[:,:,0,2].T,data[:,:,0,3].T,data[:,:,0,4].T]])
# plt.imshow(a)
# plt.title(label[:5])
# plt.show()

# a getter for mnist image and label
mnist = lambda index: (data[:,:,:,index], label[index])
def mnistShow(image):
	plt.imshow(image[:,:,0].T);
	plt.show()


def pad_image(image, px):
	x = image.shape[0]
	y = image.shape[1]
	z = image.shape[2]
	padded_image = np.ones((x+2*px, y+2*px, z)); #*100; to vizualize
	padded_image[px:-px,px:-px, :] = image;
	# plt.imshow(padded_image[:,:,0].T); plt.show()
	return padded_image

'''
too slow 
how to use tensors: object can't be assigned error
as of now takes one filter, convolves an image and returns a channel
input :
	image: 3d
	filmat: 4d (3d*nf filters)
	stride: integer value of stride
return:
	3d matrix: stacked convolution for each filter
'''
def convolve(image, filmat, stride):
	image = np.array(image, dtype=np.float32)
	x, y, z = image.shape

	if(not(isinstance(filmat, np.ndarray))):
		filmat = np.array(filmat, dtype=np.float32)

	fx, fy, fz, nf = filmat.shape

	sx = int((x-fx)/stride)
	sy = int((y-fy)/stride)
	
	img_out = np.zeros((sx, sy, nf), dtype=np.float32)
		
	for i in range(sx):
		px = i*stride
		for j in range(sy): 
			py = j*stride
			sub = image[px:px+fx, py:py+fy, :]
			img_out[i,j,:] = np.tensordot(sub,filmat, axes=3)

	return img_out



'''
	image = (h,w,c)
	bias  = (1,1,c) % added to all elements in channel, broadcasting
'''
# activates an image
def activate(image, bias, activation='relu'):
	return np.maximum(image+bias, np.zeros_like(mul))



# performs pooling operation on a image
'''
input:
	f is size of image subset to pool from
	stride - stride of pool filter
	type - max, avg pooling implemented
output: 
	3D matrix with same depth as input depth
'''
def pool(image, f, stride, type='max'):
	image = np.array(image, dtype=np.float32)
	x, y, z = image.shape

	sx = int((x-f)/stride)
	sy = int((y-f)/stride)

	img_out = np.zeros((sx, sy, z), dtype=np.float32)

	for i in range(sx):
		px = i*stride
		for j in range(sy): 
			py = j*stride
			sub = image[px:px+f, py:py+f]
			if type=='avg':
				img_out[i,j,:] = sub.mean(axis=(0,1))
			else:
				img_out[i,j,:] = sub.max(axis=(0,1))
 
	return img_out



# forward pass for fully connected
def fc_forward(a, w, b, activation):
	'''
	a = activation of prev layer (nx1)
	w = weight parameter (mxn)
	b = biases for layer (mx1)
	a_next (mx1) = w(mxn)*a(nx1) + b(mx1)
	'''
	z = np.dot(w, a)+b
	if activation=='sigmoid':
		return (z, 1/(1+np.exp(-1*z)));
	elif activation=='softmax':	
		expz = np.exp(-z);
		return (z, expz/expz.sum());
	else:
		return (z, np.maximum(z, np.zeros_like(z)));

'''
input:
	cache-tuple of required inputs
'''
def fc_backprop(al_prev, zl, al, wl_next, dl_next, activationL):
	'''
	L = -(y[j]*log(a[j]) + (2-y[j])log(1-a[j]))
	a[j] = 1/(1+exp(z[j]); z[j] = w[ij]a_[i]
	dw[ij] = (y[j]-a[j])/{aj*(1-aj)}*a[j](1-a[j])*a_[i] = (yj-aj)a_i
        https://medium.com/@erikhallstrm/backpropagation-from-the-beggining
        dl = dc/dzl
        '''
        if activationL = 'sigmoid':
            daldzl = al*(1-al)
        elif activationL = 'softmax':
            daldzl = (al-1)*exp(-zl)
        else            # default case : relu, dvt of relu is step function 
            daldzl = np.array([0 if x<0 else 1 for x in zl])

	# classification layer
        dl = np.multiply(wl_next.T.dot(dl_next), daldzl)
        dw = np.tensordot(dl, al_prev) 
        db = dl

        return (dl, dw, db)

def conv_backprop():
        '''
        (input, filter) -> convOut, (+ bias -> activate) -> activation
        '''
        
