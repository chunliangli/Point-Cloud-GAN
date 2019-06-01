import numpy as np
import tensorflow as tf
import h5py
import pdb
from tqdm import trange


def rotate_z(theta, x):
    theta = np.expand_dims(theta, 1)
    outz = np.expand_dims(x[:,:,2], 2)
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    xx = np.expand_dims(x[:,:,0], 2)
    yy = np.expand_dims(x[:,:,1], 2)
    outx = cos_t * xx - sin_t*yy
    outy = sin_t * xx + cos_t*yy
    return np.concatenate([outx, outy, outz], axis=2)

def rotate_z_gpu(theta, x):
    theta = tf.expand_dims(theta, 1)
    outz = tf.expand_dims(x[:,:,2], 2)
    sin_t = tf.sin(theta)
    cos_t = tf.cos(theta)
    xx = tf.expand_dims(x[:,:,0], 2)
    yy = tf.expand_dims(x[:,:,1], 2)
    outx = cos_t * xx - sin_t*yy
    outy = sin_t * xx + cos_t*yy
    return tf.concat([outx, outy, outz], axis=2)
    
# def rotate_3d(thetas, x):
#     return rotate_x(thetas[:,2], rotate_y(thetas[:,1], rotate_z(thetas[:,0], x)))

def augment(x):
    bs = x.shape[0]
    thetas = np.random.uniform(-1, 1, [bs,1])*np.pi
    rotated = rotate_z(thetas, x)
    scale = np.random.uniform(size=[bs,1,3])*0.45 + 0.8
    return rotated*scale

def augment_gpu(x, y, bs):
    thetas = tf.random_uniform([bs,1], -1, 1)*tf.constant(np.pi)
    rotated = rotate_z_gpu(thetas, x)
    scale = tf.random_uniform([bs,1,3])*0.45 + 0.8
    return rotated*scale, y

def augmentd_gpu(x, y, bs):
    thetas = tf.to_float(tf.random_uniform([bs,1], 0, 8, tf.int32))*tf.constant(np.pi/4)
    rotated = rotate_z_gpu(thetas, x)
    scale = tf.random_uniform([bs,1,3])*0.45 + 0.8
    return rotated*scale, y

def standardize(x):
    clipper = np.mean(np.abs(x), (1,2), keepdims=True)
    z = np.clip(x, -100*clipper, 100*clipper)
    mean = np.mean(z, (1,2), keepdims=True)
    std = np.std(z, (1,2), keepdims=True)
    return (z-mean)/std

def standardize_gpu(x, y, num_pts):
    perm = tf.random_shuffle(tf.range(10000))
    x = tf.gather(x, perm[:num_pts], axis=1)
    #clipper = tf.reduce_mean(tf.abs(x), [1,2], keepdims=True)
    z = tf.clip_by_value(x, -20 , 20)
    mean, var = tf.nn.moments(z, [1,2], keep_dims=True)
    std = tf.sqrt(var)
    return (z - mean)/std, y


class DataLoader(object):
    def __init__(self, params, y=0, do_standardize=False, do_augmentation=False, n_obj=5):#batch_size, data, labels, shuffle=True, repeat=True:
        for key, val in params.items():
            setattr(self, key, val)
        #pdb.set_trace()
        self.y = y if isinstance(y, (list, tuple)) else [y]
        filt = np.full(self.labels.shape, False)
        for yi in self.y:
            lflt = self.labels == yi
            locs = np.where(lflt)[0]
            if len(locs) > n_obj:
                lflt[locs[n_obj]:] = False
            filt = np.logical_or(filt, lflt) #np.logical_or(self.labels==0, self.labels==7) # plane or car
        self.labels = self.labels[filt]
        self.data = self.data[filt, :, :]
        self.max_n_pt = self.data.shape[1]
       
        #self.data = self.data[:n_obj]
        n_repeat = 30000//sum(filt) #   (n_obj*len(self.y))
        self.data = np.tile(self.data, (n_repeat,1,1))
        self.labels = np.tile(self.labels, (n_repeat,))
        for i in trange(len(self.data)):
            pt_perm = np.random.permutation( self.max_n_pt )
            self.data[i] = self.data[i,pt_perm]

        self.len_data = len(self.data) 
        
        self.prep1 = (lambda x, y: standardize_gpu(x, y, self.num_points_per_object)) if do_standardize else lambda x: x
        self.prep2 = (lambda x, y: augmentd_gpu(self.prep1(x, y)[0], y, self.batch_size)) if do_augmentation else self.prep1

        data_placeholder = tf.placeholder(self.data.dtype, self.data.shape)
        labels_placeholder = tf.placeholder(self.labels.dtype, self.labels.shape)
        dataset = tf.data.Dataset.from_tensor_slices((data_placeholder, labels_placeholder))
        dataset = dataset.shuffle(30000).repeat().batch(self.batch_size).map(self.prep2, num_parallel_calls=2)
        dataset = dataset.prefetch(buffer_size=10000)
        iterator = dataset.make_initializable_iterator()

        self.sess.run(iterator.initializer, feed_dict={data_placeholder: self.data, labels_placeholder: self.labels})

        bdata, blabel = iterator.get_next()
        bdata.set_shape((self.batch_size, self.num_points_per_object, 3))
        self.next_batch = (bdata, blabel)


    def __iter__(self):
        return self.iterator()

    def iterator(self):
        while(True):
            yield self.sess.run(self.next_batch)

