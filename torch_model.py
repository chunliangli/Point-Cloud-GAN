import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import pdb

class PermEqui1_max(nn.Module):
  def __init__(self, in_dim, out_dim):
    super(PermEqui1_max, self).__init__()
    self.Gamma = nn.Linear(in_dim, out_dim)

  def forward(self, x):
    xm, _ = x.max(1, keepdim=True)
    x = self.Gamma(x-xm)
    return x


class PermEqui2_max(nn.Module):
  def __init__(self, in_dim, out_dim):
    super(PermEqui2_max, self).__init__()
    self.Gamma = nn.Linear(in_dim, out_dim)
    self.Lambda = nn.Linear(in_dim, out_dim, bias=False)

  def forward(self, x):
    xm, _ = x.max(1, keepdim=True)
    xm = self.Lambda(xm) 
    x = self.Gamma(x)
    x = x - xm

    return x


class PermEqui2_mean(nn.Module):
  def __init__(self, in_dim, out_dim):
    super(PermEqui2_mean, self).__init__()
    self.Gamma = nn.Linear(in_dim, out_dim)
    self.Lambda = nn.Linear(in_dim, out_dim, bias=False)

  def forward(self, x):
    xm = x.mean(1, keepdim=True)
    xm = self.Lambda(xm) 
    x = self.Gamma(x)
    x = x - xm

    return x


class G_inv_Tanh(nn.Module):

  def __init__(self, x_dim, d_dim, z1_dim, pool = 'mean'):
    super(G_inv_Tanh, self).__init__()
    self.d_dim = d_dim
    self.x_dim = x_dim
    self.z1_dim = z1_dim
    self.pool = pool

    if pool == 'max':
        self.phi = nn.Sequential(
          PermEqui2_max(self.x_dim, self.d_dim),
          nn.Tanh(),
          PermEqui2_max(self.d_dim, self.d_dim),
          nn.Tanh(),
          PermEqui2_max(self.d_dim, self.d_dim),
          nn.Tanh(),
        )
    elif pool == 'max1':
        self.phi = nn.Sequential(
          PermEqui1_max(self.x_dim, self.d_dim),
          nn.Tanh(),
          PermEqui1_max(self.d_dim, self.d_dim),
          nn.Tanh(),
          PermEqui1_max(self.d_dim, self.d_dim),
          nn.Tanh(),
        )
    elif pool == 'mean':
        self.phi = nn.Sequential(
          PermEqui2_mean(self.x_dim, self.d_dim),
          nn.Tanh(),
          PermEqui2_mean(self.d_dim, self.d_dim),
          nn.Tanh(),
          PermEqui2_mean(self.d_dim, self.d_dim),
          nn.Tanh(),
        )
    
    self.ro = nn.Sequential(
       nn.Linear(self.d_dim, self.d_dim),
       nn.Tanh(),
       nn.Linear(self.d_dim, self.z1_dim),
    )
    print(self) 
    self.faster_parameters = [p for p in self.parameters()]

  def forward(self, x):
    phi_output = self.phi(x)
    sum_output, _ = phi_output.max(1)
    ro_output = self.ro(sum_output)
    return ro_output


class G_inv(nn.Module):

  def __init__(self, x_dim, d_dim, z1_dim, pool = 'mean'):
    super(G_inv, self).__init__()
    self.d_dim = d_dim
    self.x_dim = x_dim
    self.z1_dim = z1_dim
    self.pool = pool

    if pool == 'max':
        self.phi = nn.Sequential(
          PermEqui2_max(self.x_dim, self.d_dim),
          nn.Softplus(),
          PermEqui2_max(self.d_dim, self.d_dim),
          nn.Softplus(),
          PermEqui2_max(self.d_dim, self.d_dim),
          nn.Softplus(),
        )
    elif pool == 'max1':
        self.phi = nn.Sequential(
          PermEqui1_max(self.x_dim, self.d_dim),
          nn.Softplus(),
          PermEqui1_max(self.d_dim, self.d_dim),
          nn.Softplus(),
          PermEqui1_max(self.d_dim, self.d_dim),
          nn.Softplus(),
        )
    elif pool == 'mean':
        self.phi = nn.Sequential(
          PermEqui2_mean(self.x_dim, self.d_dim),
          nn.Softplus(),
          PermEqui2_mean(self.d_dim, self.d_dim),
          nn.Softplus(),
          PermEqui2_mean(self.d_dim, self.d_dim),
          nn.Softplus(),
        )

    self.ro = nn.Sequential(
      nn.Linear(self.d_dim, self.d_dim),
      nn.Softplus(),
      nn.Linear(self.d_dim, self.z1_dim),
    ) 
    print(self)
    self.faster_parameters = [p for p in self.parameters()]

  def forward(self, x):
    phi_output = self.phi(x)
    sum_output, _ = phi_output.max(1)
    ro_output = self.ro(sum_output)
    return ro_output


class D(nn.Module):

  def __init__(self, x_dim, z1_dim, d_dim, o_dim=1):
    super(D, self).__init__()
    self.d_dim = d_dim
    self.x_dim = x_dim
    self.z1_dim = z1_dim

    self.fc = nn.Linear(self.z1_dim, self.d_dim)
    self.fu = nn.Linear(self.x_dim, self.d_dim, bias=False)
    self.f = nn.Sequential(
      nn.Softplus(),
      nn.Linear(self.d_dim, self.d_dim),
      nn.Softplus(),
      nn.Linear(self.d_dim, self.d_dim),
      nn.Softplus(),
      nn.Linear(self.d_dim, self.d_dim),
      nn.Softplus(),
      nn.Linear(self.d_dim, o_dim),
    )
    print(self)
    self.faster_parameters = [p for p in self.parameters()]
    

  def forward(self, x, z1):
    y = self.fc(z1) + self.fu(x)
    return self.f(y) 


class skipD(nn.Module):

  def __init__(self, x_dim, z1_dim, d_dim, o_dim=1):
    super(skipD, self).__init__()
    self.d_dim = d_dim
    self.x_dim = x_dim
    self.z1_dim = z1_dim
    #hid_d = 5*(z1_dim+z2_dim)
    hid_d = max(1024, 2*z1_dim)

    self.fc = nn.Linear(self.z1_dim, hid_d)
    self.fu = nn.Linear(self.x_dim, hid_d, bias=False)
    self.part1 = nn.Sequential(
      nn.Softplus(),
      nn.Linear(hid_d, hid_d),
      nn.Softplus(),
      nn.Linear(hid_d, hid_d - self.z1_dim),
    )

    self.sc = nn.Linear(self.z1_dim, hid_d)
    self.su = nn.Linear(hid_d - self.z1_dim, hid_d, bias=False)
    self.part2 = nn.Sequential(
      nn.Softplus(),
      nn.Linear(hid_d, hid_d),
      nn.Softplus(),
      nn.Linear(hid_d, hid_d - self.z1_dim)
    )

    self.tc = nn.Linear(self.z1_dim, hid_d)
    self.tu = nn.Linear(hid_d - self.z1_dim, hid_d, bias=False)
    self.part3 = nn.Sequential(
      nn.Softplus(),
      nn.Linear(hid_d, hid_d),
      nn.Softplus(),
      nn.Linear(hid_d, o_dim)
    )
    print(self)
    self.faster_parameters = [p for p in self.parameters()]

  def forward(self, x, z1):
    y = self.fc(z1) + self.fu(x)
    output = self.part1(y)
    y2 = self.sc(z1) + self.su(output)
    output1 = self.part2( y2 )
    y3 = self.tc(z1) + self.tu(output1)
    output2 = self.part3( y3)
    return output2


class G(nn.Module):

  def __init__(self, x_dim, z1_dim, z2_dim):
    super(G, self).__init__()
    self.z1_dim = z1_dim
    self.z2_dim = z2_dim
    self.x_dim = x_dim
    hid_d = max(250, 2*z1_dim)
    #hid_d = z1_dim+z2_dim

    self.fc = nn.Linear(self.z1_dim, hid_d)
    self.fu = nn.Linear(self.z2_dim, hid_d, bias=False)
    self.main = nn.Sequential(
      nn.Softplus(),
      nn.Linear(hid_d, hid_d),
      nn.Softplus(),
      nn.Linear(hid_d, hid_d),
      nn.Softplus(),
      nn.Linear(hid_d, hid_d),
      nn.Softplus(),
      nn.Linear(hid_d, hid_d),
      nn.Softplus(),
      nn.Linear(hid_d, self.x_dim),
    )
    print(self)
    self.faster_parameters = [p for p in self.parameters()]

  def forward(self, z1, z2):
    x = self.fc(z1) + self.fu(z2)
    output = self.main(x)
    return output


class skipG(nn.Module):

  def __init__(self, x_dim, z1_dim, z2_dim):
    super(skipG, self).__init__()
    self.z1_dim = z1_dim
    self.z2_dim = z2_dim
    self.x_dim = x_dim
    #hid_d = 5*(z1_dim+z2_dim)
    hid_d = 250

    self.fc = nn.Linear(self.z1_dim, hid_d)
    self.fu = nn.Linear(self.z2_dim, hid_d, bias=False)
    self.part1 = nn.Sequential(
      nn.Softplus(),
      nn.Linear(hid_d, hid_d),
      nn.Softplus(),
      nn.Linear(hid_d, z2_dim),
    )

    self.sc = nn.Linear(self.z1_dim, hid_d)
    self.su = nn.Linear(self.z2_dim, hid_d, bias=False)
    self.part2 = nn.Sequential(
      nn.Softplus(),
      nn.Linear(hid_d, hid_d),
      nn.Softplus(),
      nn.Linear(hid_d, x_dim)
    )
    print(self)
    self.faster_parameters = [p for p in self.parameters()]

  def forward(self, z1, z2):
    x = self.fc(z1) + self.fu(z2)
    output = self.part1(x)
    x1 = self.sc(z1) + self.su(output)
    output1 = self.part2(x1)
    return output1


class ALPHA(nn.Module):

  def __init__(self, z1_dim):
    super(ALPHA, self).__init__()
    self.z1_dim = z1_dim
    hid_d = min(z1_dim + 50, 100)
    #hid_d = z1_dim+z2_dim

    self.main = nn.Sequential(
      nn.Linear(self.z1_dim, hid_d),
      nn.Softplus(),
      nn.Linear(hid_d, hid_d),
      nn.Softplus(),
      nn.Linear(hid_d, hid_d),
      nn.Softplus(),
      nn.Linear(hid_d, hid_d),
      nn.Softplus(),
      nn.Linear(hid_d, 1),
    )
    print(self)
    self.faster_parameters = [p for p in self.parameters()]

  def forward(self, x):
    output = self.main(x)
    return output


def zero_weights_init(m):
  if isinstance(m, nn.Linear):
      m.weight.data.uniform_(-5e-3, 5e-3)
      m.bias.data.fill_(0)
