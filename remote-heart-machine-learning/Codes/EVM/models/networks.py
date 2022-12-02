import torch
import functools
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler

class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
      classname = m.__class__.__name__
      if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
          if init_type == 'normal':
              init.normal_(m.weight.data, 0.0, init_gain)
          elif init_type == 'xavier':
              init.xavier_normal_(m.weight.data, gain=init_gain)
          elif init_type == 'kaiming':
              init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
          elif init_type == 'orthogonal':
              init.orthogonal_(m.weight.data, gain=init_gain)
          else:
              raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
          if hasattr(m, 'bias') and m.bias is not None:
              init.constant_(m.bias.data, 0.0)
      elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
          init.normal_(m.weight.data, 1.0, init_gain)
          init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(torch.device(gpu_ids[0]))
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_evm_cnn_network(init_type, init_gain, gpu_ids):
  net = EVM_CNN_Network()
  return init_net(net, init_type, init_gain, gpu_ids)


class EVM_CNN_Network(nn.Module):
   def __init__(self):

    super(EVM_CNN_Network, self).__init__()
    ch = 96

    model = [
              ConvBlock(3, ch, (5, 5), 1, 1),
              DWConvBlock(ch, (3, 3), 1, 0),
              PWConvBlock(ch, ch, (1, 1), 1),
              DWConvBlock(ch, (3, 3), 2),
              PWConvBlock(ch, ch, (1, 1), 1),
              DWConvBlock(ch, (3, 3), 2),
              PWConvBlock(ch, int(ch * (4/3)), (1, 1), 1)
     ]

    ch = int(ch * (4/3))

    for i in range(2):
      model += [
              DWConvBlock(ch, (3, 3), 2),
              PWConvBlock(ch, ch, (1, 1), 1)
      ]

    pool_size = (2, 2)

    model += [
              nn.AvgPool2d(pool_size, 1),
              nn.Flatten(),
              nn.Linear(int(ch * 3), int(ch * (3/2))),
              nn.Dropout2d(0.6),
              nn.Linear(int(ch * (3/2)), 30)
            ]

    self.model = nn.Sequential(*model)

   def forward(self, input):
      return self.model(input)

class ConvBlock(nn.Module):
  def __init__(self, input_channels, output_channels, kernel_size, stride, padding, groups = 1):
    super(ConvBlock, self).__init__()
    conv_block = [  
                        nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding = padding, groups = groups, bias = False),
                        nn.BatchNorm2d(output_channels),
                        nn.ReLU(True)
    ]

    self.conv_block = nn.Sequential(*conv_block)

  def forward(self, input):
    return self.conv_block(input)

class DWConvBlock(nn.Module):
  def __init__(self, dim, kernel_size, stride, padding = 1):
    super(DWConvBlock, self).__init__()
    self.dw_conv_block = ConvBlock(input_channels = dim, output_channels = dim, kernel_size = kernel_size, padding = padding, stride = stride, groups = dim)
    

  def forward(self, input):
    return self.dw_conv_block(input)

class PWConvBlock(nn.Module):
  def __init__(self, input_channels, output_channels, kernel_size, stride, padding = 0):
    super(PWConvBlock, self).__init__()
    pw_conv_clock = [
                          nn.Conv2d(input_channels, output_channels, stride = stride, kernel_size = kernel_size, padding = padding, bias = False),
                          nn.BatchNorm2d(output_channels),
                          nn.ReLU(True)
    ]

    self.pw_conv_clock = nn.Sequential(*pw_conv_clock)

  def forward(self, input):
     return self.pw_conv_clock(input)





