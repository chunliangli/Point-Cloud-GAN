import numpy as np
import tensorflow as tf
import pdb

from base_trainer import BaseTrainer
from structural_losses.tf_approxmatch import approx_match, match_cost


class ApproxFisherTrainer(BaseTrainer):

    def __init__(self, G, D, G_inv, params):
        self.alpha = tf.Variable(0.0, trainable=False)
        super(ApproxFisherTrainer, self).__init__(G, D, G_inv, params)

    def build(self, inputs):
        rho = 1e-6

        #get z1
        real_x = inputs
        z1 = tf.expand_dims(self.G_inv(real_x), axis=1)

        #real part
        err_real = tf.squeeze(self.D(real_x, tf.stop_gradient(z1)))
        E_P_f2 = tf.reduce_mean( tf.square(err_real) , axis=1)

        #fake part
        z2 = self.generate_noise()
        fake_x = self.G(z1, z2)
        err_fake = tf.squeeze(self.D(fake_x, tf.stop_gradient(z1)))
        E_Q_f2 = tf.reduce_mean( tf.square(err_fake) , axis=1)

        #Constraint
        constraint = 1 - 0.5*(E_P_f2 + E_Q_f2)

        #loss
        lossD = tf.reduce_mean(err_real) - tf.reduce_mean(err_fake)
        mean_cons = tf.reduce_mean(constraint)
        lossD_cons = lossD - self.alpha*mean_cons + 0.5*rho*tf.reduce_mean(tf.square(constraint))
        match = approx_match(fake_x, real_x)
        lossG = tf.reduce_mean(err_fake) + 20*tf.reduce_mean(match_cost(fake_x, real_x, match))/self.num_points_per_object

        return [lossD, lossD_cons, lossG, mean_cons]


    def update_ops(self, outputs):
        rho = 1e-6

        #train ops
        self.lossD, lossD_cons, self.lossG, mean_cons = outputs
        self.trainD = tf.group(self.optimD.minimize(lossD_cons, var_list=self.D.parameters(), colocate_gradients_with_ops=True), self.alpha.assign_sub(rho*mean_cons))
        self.trainG = self.optimG.minimize(self.lossG, var_list=self.G.parameters(), colocate_gradients_with_ops=True)
        self.trainG_inv = self.optimG_inv.minimize(self.lossG, var_list=self.G_inv.parameters(), colocate_gradients_with_ops=True)
