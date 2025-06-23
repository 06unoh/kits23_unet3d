import torch
from torch import nn

class SimpleUNet3D(nn.Module):
  def __init__(self,in_channels=1,out_channels=1):
    super().__init__()
    def double_conv(in_c,out_c):
      return nn.Sequential(
          nn.Conv3d(in_channels=in_c,out_channels=out_c,kernel_size=3, padding=1),
          nn.BatchNorm3d(out_c),
          nn.ReLU(),
          nn.Dropout3d(0.2),
          nn.Conv3d(out_c,out_c,kernel_size=3, padding=1),
          nn.BatchNorm3d(out_c),
          nn.ReLU()
      )

    self.dconv_down1=double_conv(in_channels,16)
    self.dconv_down2=double_conv(16,64)
    self.dconv_down3=double_conv(64,128)
    self.dconv_down4=double_conv(128,256)

    self.maxpool=nn.MaxPool3d(2)

    self.upsample=nn.Upsample(scale_factor=2, mode='trilinear',align_corners=True)

    self.dconv_up3=double_conv(256+128,128)
    self.dconv_up2=double_conv(128+64,64)
    self.dconv_up1=double_conv(64+16,16)

    self.last_conv=nn.Conv3d(16,out_channels,1)

  def forward(self, x):
    conv1=self.dconv_down1(x)
    x=self.maxpool(conv1)
    conv2=self.dconv_down2(x)
    x=self.maxpool(conv2)
    conv3=self.dconv_down3(x)
    x=self.maxpool(conv3)
    conv4=self.dconv_down4(x)

    x=self.upsample(conv4)
    x=torch.cat([x, conv3], dim=1)
    x=self.dconv_up3(x)

    x=self.upsample(x)
    x=torch.cat([x, conv2], dim=1)
    x=self.dconv_up2(x)

    x=self.upsample(x)
    x=torch.cat([x, conv1], dim=1)
    x=self.dconv_up1(x)

    x=self.last_conv(x)
    return x