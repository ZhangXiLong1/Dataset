import torch
from torch import nn

# --------------------------
# Common functions
# --------------------------
class reflect_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, pad=1):
        super(reflect_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(pad),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=0)
        )
    def forward(self, x):
        return self.conv(x)

def gradient(input):
    filter1 = nn.Conv2d(1, 1, 3, padding=1, bias=False)
    filter2 = nn.Conv2d(1, 1, 3, padding=1, bias=False)
    filter1.weight.data = torch.tensor([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]]).reshape(1,1,3,3).cuda()
    filter2.weight.data = torch.tensor([[1.,2.,1.],[0.,0.,0.],[-1.,-2.,-1.]]).reshape(1,1,3,3).cuda()
    g1 = filter1(input)
    g2 = filter2(input)
    return torch.abs(g1) + torch.abs(g2)

def clamp(value, min=0., max=1.):
    return torch.clamp(value, min, max)

# --------------------------
# PIAFusion
# --------------------------
def CMDAF(vi_feature, ir_feature):
    sigmoid = nn.Sigmoid()
    gap = nn.AdaptiveAvgPool2d(1)
    sub_vi_ir = vi_feature - ir_feature
    vi_ir_div = sub_vi_ir * sigmoid(gap(sub_vi_ir))
    sub_ir_vi = ir_feature - vi_feature
    ir_vi_div = sub_ir_vi * sigmoid(gap(sub_ir_vi))
    vi_feature += ir_vi_div
    ir_feature += vi_ir_div
    return vi_feature, ir_feature

def Fusion(vi_out, ir_out):
    return torch.cat([vi_out, ir_out], dim=1)

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.vi_conv1 = nn.Conv2d(1,16,1)
        self.ir_conv1 = nn.Conv2d(1,16,1)
        self.vi_conv2 = reflect_conv(16,16,3,1,1)
        self.ir_conv2 = reflect_conv(16,16,3,1,1)
        self.vi_conv3 = reflect_conv(16,32,3,1,1)
        self.ir_conv3 = reflect_conv(16,32,3,1,1)
        self.vi_conv4 = reflect_conv(32,64,3,1,1)
        self.ir_conv4 = reflect_conv(32,64,3,1,1)
        self.vi_conv5 = reflect_conv(64,128,3,1,1)
        self.ir_conv5 = reflect_conv(64,128,3,1,1)
    def forward(self, y_vi_image, ir_image):
        act = nn.LeakyReLU()
        vi_out = act(self.vi_conv1(y_vi_image))
        ir_out = act(self.ir_conv1(ir_image))
        vi_out, ir_out = CMDAF(act(self.vi_conv2(vi_out)), act(self.ir_conv2(ir_out)))
        vi_out, ir_out = CMDAF(act(self.vi_conv3(vi_out)), act(self.ir_conv3(ir_out)))
        vi_out, ir_out = CMDAF(act(self.vi_conv4(vi_out)), act(self.ir_conv4(ir_out)))
        vi_out, ir_out = act(self.vi_conv5(vi_out)), act(self.ir_conv5(ir_out))
        return vi_out, ir_out

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = reflect_conv(256,256,3,1,1)
        self.conv2 = reflect_conv(256,128,3,1,1)
        self.conv3 = reflect_conv(128,64,3,1,1)
        self.conv4 = reflect_conv(64,32,3,1,1)
        self.conv5 = nn.Conv2d(32,1,1)
    def forward(self,x):
        act = nn.LeakyReLU()
        x = act(self.conv1(x))
        x = act(self.conv2(x))
        x = act(self.conv3(x))
        x = act(self.conv4(x))
        return x

class PIAFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    def forward(self, y_vi_image, ir_image):
        vi_out, ir_out = self.encoder(y_vi_image, ir_image)
        fused = Fusion(vi_out, ir_out)
        fused = self.decoder(fused)
        return fused
