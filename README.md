# Point-Cloud-GAN
The code for Point Cloud GAN (https://arxiv.org/abs/1810.05795). 

For the hierarchical sampling, you have to train the other GAN for the latent code. You can use any existing implementation on github, such as WGAN-GP. 
For running, you have to 

(1) compile code in ./structural_losses

(2) prepare data. Please modify data_loader.py for your own format accordingly. 

(3) run with python main.py --config=config/sandwich_shape.yml

Please note that, the code is in tensorflow but we save the model in pytorch format. However, the implementation for Softplus activation is different in Tensorflow and Pytorch. So you have to modify your pytorch source code accordingly, or you can replace them with ReLU.  Training on all modelnet40 data is slow. We used 4 V100 to train  and it still took us a few days. For multi-gpu, please check base_trainer.py. The code provided here is using gpu 0 and gpu 1. You can modify it accordingly. 


If you find our paper or implementation useful in your research, please cite:

<table border="0" cellspacing="15" cellpadding="0">
<tbody>
<tr>
<td>
<pre>			
@article{li2018point,
  title={Point cloud gan},
  author={Li, Chun-Liang and Zaheer, Manzil and Zhang, Yang and Poczos, Barnabas and Salakhutdinov, Ruslan},
  journal={arXiv preprint arXiv:1810.05795},
  year={2018}
}
</pre>
</tbody>
</table>

#Acknowledgement
Part of our code is from https://github.com/optas/latent_3d_points
