import os
import h5py
import torch
import numpy as np
import tensorflow as tf
from tqdm import tqdm, trange
from data_loader import DataLoader
import pdb

import torch_model




# change this
class BaseTrainer(object):

    def __init__(self, G, D, G_inv, params):
        self.G = G
        self.D = D
        self.G_inv = G_inv

        self.gpus = [0, 1]

        self.pth_G = getattr(torch_model, self.G.__class__.__name__)(z1_dim=params['z1_dim'], z2_dim=params['z2_dim'], x_dim=params['x_dim'])
        self.pth_G_inv = getattr(torch_model, self.G_inv.__class__.__name__)(x_dim=params['x_dim'], d_dim=params['d_dim'], z1_dim=params['z1_dim'], pool=params['pool'])

        # transfer parameters to self
        for key, val in params.items(): 
            setattr(self, key, val)

        #config = tf.ConfigProto()
        #config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
        #config = tf.ConfigProto(allow_soft_placement = True)

        self.sess = tf.Session() #config=config)
        self.samples_from_target_distribution, _ = self.init_data().next_batch

        #pdb.set_trace()

        if len(self.gpus) > 1:
            outputs =  self.make_parallel(inputs=self.samples_from_target_distribution)
        else:
            outputs = self.build(self.samples_from_target_distribution)

        if self.optimizer == 'rmsprop':
            self.optimD = tf.train.RMSPropOptimizer(self.d_lr, decay=0.9, epsilon=1e-6)
            self.optimG = tf.train.RMSPropOptimizer(self.g_lr, decay=0.9, epsilon=1e-6)
            self.optimG_inv = tf.train.RMSPropOptimizer(self.inv_lr, decay=0.9, epsilon=1e-6)

        elif self.optimizer == 'adam':
            self.optimD = tf.train.AdamOptimizer(self.d_lr, beta1=0.1, beta2=0.999, epsilon=1e-3)
            self.optimG = tf.train.AdamOptimizer(self.g_lr, beta1=0.1, beta2=0.999, epsilon=1e-3)
            self.optimG_inv = tf.train.AdamOptimizer(self.inv_lr, beta1=0.1, beta2=0.999, epsilon=1e-3)

        elif self.optimizer == 'sgd':
            self.optimD = tf.train.GradientDescentOptimizer(self.d_lr)
            self.optimG = tf.train.GradientDescentOptimizer(self.g_lr)
            self.optimG_inv = tf.train.GradientDescentOptimizer(self.inv_lr)

        self.update_ops(outputs)


    def make_parallel(self, **kwargs):
        in_splits = {}
        num_gpus = len(self.gpus)
        for k, v in kwargs.items():
            in_splits[k] = tf.split(v, num_gpus)
    
        out_split = []
        for i in self.gpus:
            with tf.device(tf.DeviceSpec(device_type="GPU", device_index=i)):
                with tf.variable_scope(tf.get_variable_scope(),
                                       reuse=(i != self.gpus[0])):
                    #pdb.set_trace()
                    out_split.append(self.build(**{k: v[i] for k, v in in_splits.items()}))
    
        outputs = zip(*out_split)
        outputs = [tf.reduce_mean(output) for output in outputs]
    
        return outputs



    def generate_noise(self):
        return tf.random_normal([self.batch_size//len(self.gpus), self.num_points_per_object, self.z2_dim])


    def build(self, inputs):
        raise NotImplementedError

    def update_ops(self, outputs):
        self.lossD, lossD_cons, self.lossG = outputs
        self.trainD = self.optimD.minimize(lossD_cons, var_list=self.D.parameters(), colocate_gradients_with_ops=True)
        self.trainG = self.optimG.minimize(self.lossG, var_list=self.G.parameters(), colocate_gradients_with_ops=True)
        self.trainG_inv = self.optimG_inv.minimize(self.lossG, var_list=self.G_inv.parameters(), colocate_gradients_with_ops=True)


    def init_data(self):
        """
        :params fp: filepath
        """
        with h5py.File(self.data_file, 'r') as f:
            trd= np.array(f['tr_cloud'])
            trl = np.array(f['tr_labels'])
            ted = np.array(f['test_cloud'])
            tel = np.array(f['test_labels'])
            train_params = {'data': trd, 'labels': trl, 'shuffle': True, 
                            'repeat': True, 'num_points_per_object': self.num_points_per_object, 
                            'batch_size' : self.batch_size, 'sess': self.sess}
            tr_loader = DataLoader(train_params, y=self.obj, do_standardize=True, n_obj=self.n_obj, do_augmentation=True)
        return tr_loader


    def train(self):
        f = open(os.path.join(self.out_dir,'log.txt'), 'a')
        g = None # open(os.path.join(self.out_dir,'out.txt'), 'w')

        train_writer = tf.summary.FileWriter( os.path.join(self.out_dir,'tb'), self.sess.graph)
        merge = tf.summary.merge_all()

        self.sess.run(tf.global_variables_initializer())
        
        #pdb.set_trace()
        iters = trange(self.num_iters, desc="Dloss: 0.00000 Gloss: 0.000000 ", file = g, ncols=120)
        for i in iters:

            if i % 50 != 0:
            
                # D part
                for j in range(self.critic_steps):
                    self.sess.run(self.trainD)

                # G part
                self.sess.run([self.trainG, self.trainG_inv])

            else:
                # Display loss statistics
                D_loss = 0.0
                for j in range(self.critic_steps):
                    D_loss_j, _ = self.sess.run([self.lossD, self.trainD])
                    D_loss += D_loss_j
                D_loss /= self.critic_steps
                G_loss, _, _ = self.sess.run([self.lossG, self.trainG, self.trainG_inv])

                iters.set_description("Dloss: {0:0.6f} Gloss: {1:0.6f} ".format(D_loss, G_loss))
                #print('Iter:{0}, Dloss: {1}, Gloss: {2}'.format(i, D_loss, G_loss))
                tqdm.write('Iter:{0}, Dloss: {1}, Gloss: {2}'.format(i, D_loss, G_loss), file=g)
                try:
                    f.write('Iter:{0}, Dloss: {1}, Gloss: {2}: \n'.format(i, D_loss, G_loss))
                    f.flush()
                except:
                    print('Could not write log')

            # Save model
            if i % 1000 == 0:
                try:
                    self.G.transfer_to_pytorch_model(self.sess, self.pth_G)
                    self.G_inv.transfer_to_pytorch_model(self.sess, self.pth_G_inv)
                    torch.save(self.pth_G, self.out_dir + '/G_network_{0}.pth'.format(i))
                    torch.save(self.pth_G_inv, self.out_dir + '/G_inv_network_{0}.pth'.format(i))
                except:
                    print('Could not save at iter {}'.format(i))

        f.close()
        #g.close()
