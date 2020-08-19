import torch
import torch.nn as nn

class conv_block(nn.Module):

  def __init__(self, input_size,  output_size, filter=3, stride=1, padding=1, act=nn.ReLU, reps=2, norm='mvn'):
    super(conv_block, self).__init__()
    in_conv = input_size
    blocks = nn.ModuleList()
    for i in range(reps):
      layers = []
      layers.append(nn.Conv2d(in_conv, output_size, filter, stride, padding=padding))
      if norm=='batchnorm':
        layers.append(nn.BatchNorm2d(output_size))
      elif norm=='mvn':
        layers.append(MVN())
      layers.append(act())
      blocks.append(nn.Sequential(*layers))
      in_conv = output_size
    self.blocks = blocks
  
  def forward(self, x):
    for block in self.blocks:
      x = block(x)
    return x

class SegmentLoss(nn.Module):
    def init(self):
        super(diceLoss, self).init()
    def forward(self, pred, target):
      smooth = 1.
      return 1 - self.dice_coef(pred, target, smooth=smooth)

    def dice_coef(self, pred, target, smooth = 1e-3):
      iflat = pred.contiguous().view(-1)
      tflat = target.contiguous().view(-1)
      intersection = (iflat * tflat).sum()
      A_sum = torch.sum(iflat * iflat)
      B_sum = torch.sum(tflat * tflat)
      return (2. * intersection + smooth) / (A_sum + B_sum + smooth)

    def jaccard_coef(self, pred, target, smooth = 1e-3):
      iflat = pred.contiguous().view(-1)
      tflat = target.contiguous().view(-1)
      intersection = (iflat * tflat).sum()
      A_sum = torch.sum(iflat * iflat)
      B_sum = torch.sum(tflat * tflat)
      return (intersection + smooth) / (A_sum + B_sum - intersection + smooth)

 class MVN(nn.Module):
  def __init__(self, esp=1e-6):
    super(MVN, self).__init__()
    self.esp=esp
  def forward(self, x):
    mean = torch.mean(x, dim=(2,3), keepdim=True)
    std = torch.std(x, dim=(2,3), keepdim=True)
    x = (x-mean)/(std+self.esp)
    return x

class Unet_based(nn.Module):

  def __init__(self, n_class, norm=None, model_name='unet'):
    super(Unet_based, self).__init__()
    self.model_name = model_name
    self.n_class = n_class
    self.bd = nn.ModuleList([
                             conv_block(3,64,reps=2,norm=norm),
                             conv_block(64,128,reps=2,norm=norm),
                             conv_block(128,256,reps=2,norm=norm),
                             conv_block(256,512,reps=2,norm=norm),
                             conv_block(512,512,reps=2,norm=norm)
    ])
    self.maxpool = nn.MaxPool2d(2,2)
    self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    self.bu = nn.ModuleList([
                             conv_block(512+512,512,reps=2,norm=norm),
                             conv_block(256+512,256,reps=2,norm=norm),
                             conv_block(128+256,128,reps=2,norm=norm),
                             conv_block(64+128,64,reps=2,norm=norm)
    ])
    self.attgate = nn.ModuleList([
                                  Attention_block(512,512, 512),
                                  Attention_block(512,256,256),
                                  Attention_block(256,128,128),
                                  Attention_block(128,64,64),
    ])
    if n_class ==1:
      self.last_conv = nn.Sequential(nn.Conv2d(64,n_class,1), nn.Sigmoid())
    else:
      self.last_conv = nn.Sequential(nn.Conv2d(64,n_class,1), nn.Softmax())
  
  def forward(self, x):
    convs=[]
    for i, block in enumerate(self.bd):
      if i< len(self.bd)-1:
        convs.append(block(x))
        x = self.maxpool(convs[-1])
      else:
        convs.append(block(x))
        x = convs[-1]
    len_conv = len(convs)
    for i, block in enumerate(self.bu):
      if self.model_name == 'unet':
      	x = self.upsample(x)
  		x = torch.cat([x, convs[len_conv-i-2]], dim=1)
      else:
      	g = x
      	x = self.upsample(x)
      	g_sig = self.attgate[i](g, convs[len_conv-i-2])
      	x = torch.cat([x, g_sig], dim=1)
      x = block(x)
    x = self.last_conv(x)
    return x