from cnn_img_filter_tf import *
from math import *
import matplotlib.pyplot as plt


filter_ev = np.array([[1,0,-1],[3,0,-3],[1,0,-1]])
# filter_ev = np.repeat(filter_ev[:,:,np.newaxis], 3, axis=2)
filter_ev = np.reshape(filter_ev, (3,3,1))

f = 3	# dimension of filter matrix
nc= 1	# depth of image channel monochrome:1, rgb:3, rgba:4
nf= 4	# number of image filters
fil = np.random.randn(f,f,nc,nf)

img, lab = mnist(10)

p = pad_image(img, 2)

c = convolve(p, filter_ev, 1)

weigh = np.ones_like(c, dtype=np.int32)
bias  = 50;
a = activate(c, weigh, 50)

po = pool(a, 3, 1, 'max');

# plt.imshow(po); plt.show()

# fc layer
fc0 = po.reshape(-1)
fc0 = (fc0-fc0.mean())/fc0.std()
w1  = np.random.randn(100, fc0.shape[0])
b1  = np.zeros((100))
z1, fc1 = nn_pass(fc0, w1, b1, 'relu')
 
# softmax layer
print('starting sotfmax layer')
fc1 = (fc1-fc1.mean())/fc1.std()
w2 = np.random.randn(10, fc1.shape[0])
b2 = np.zeros((10))
z2, y = nn_pass(fc1, w2, b2, 'softmax')

print(y.sum())
