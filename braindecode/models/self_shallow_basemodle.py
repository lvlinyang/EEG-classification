import numpy as np
from torch import nn
from torch.nn import init,Module,Sequential
import torch
from braindecode.models.base import BaseModel
from braindecode.torch_ext.modules import Expression, AvgPool2dWithConv
from braindecode.torch_ext.functions import safe_log, square
from braindecode.torch_ext.functions import identity
from braindecode.torch_ext.util import np_to_var
from torch.nn.functional import interpolate


def _squeeze_final_output(x):
    assert x.size()[3] == 1
    x = x[:, :, :, 0]
    if x.size()[2] == 1:
        x = x[:, :, 0]
    return x

def _transpose_time_to_spat(x):
    return x.permute(0, 1, 3,2)

class NewConv2d(BaseModel):
    def __init__(self):
        self.__dict__.update(locals())
        super(NewConv2d, self).__init__()
        del self.self
    def createNetwor(self,FInput,Foutput,k,s,p=0,b=False):
        model = nn.Sequential(nn.Conv2d(FInput,Foutput,kernel_size=k,stride=s,padding=p,bias=b))
        return model

class NewBN2d(BaseModel):
    def __init__(self):
        self.__dict__.update(locals())
        super(NewConv2d, self).__init__()
        del self.self
    def createNetwor(self,num):
        model = nn.Sequential(nn.BatchNorm2d(num))
        return model

class NewDCT2d(BaseModel):
    def __init__(self):
        self.__dict__.update(locals())
        super(NewDCT2d, self).__init__()
        del self.self
    def createNetwor(self,FInput,Foutput,k,s):
        model = nn.Sequential(nn.ConvTranspose2d(FInput,Foutput,kernel_size=k,stride=s))
        return model

class NewPool2d(BaseModel):
    def __init__(self):
        self.__dict__.update(locals())
        super(NewPool2d, self).__init__()
        del self.self
    def createNetwor(self,k,s):
        model = nn.Sequential(nn.AvgPool2d( kernel_size=k , stride=s))
        return model

class MFF (Module):
    def __init__(self):
        super(MFF, self).__init__()

        self.convm1 = NewConv2d.createNetwor(self,32,32,k=3,s=1,p=1)

        self.convm2 = NewConv2d.createNetwor(self,96,32,1,1)

        self.convm3 = NewConv2d.createNetwor(self,96,40,1,1)

    def corss_connection(self,L_0,L_1,L_2):
        L0_batch, L0_C, L0_Height, L0_Width = L_0.size()
        L1_batch, L1_C, L1_Height, L1_Width = L_1.size()
        L2_batch, L2_C, L2_Height, L2_Width = L_2.size()

        L2_0 = interpolate(L_2,size=[L0_Height,L0_Width],mode='nearest')
        L2_1 = interpolate(L_2,size=[L1_Height,L1_Width],mode="nearest")

        L1_2 = interpolate(L_1, size=[L2_Height, L2_Width], mode="nearest")
        L1_0 = interpolate(L_1, size=[L0_Height, L0_Width], mode="nearest")

        L0_1 = interpolate(L_0, size=[L1_Height, L1_Width], mode="nearest")
        L0_2 = interpolate(L_0, size=[L2_Height, L2_Width], mode="nearest")


        L00 = torch.cat((L_0,L1_0,L2_0),dim=1)
        L11 = torch.cat((L_1,L0_1,L2_1),dim=1)
        L22 = torch.cat((L_2,L1_2,L0_2),dim=1)

        f0 = self.convm2(L00)
        f1 = self.convm2(L11)
        f2 = self.convm2(L22)

        f1_0 = interpolate(f1, size=[L0_Height, L0_Width], mode="nearest")
        f2_0 = interpolate(f2, size=[L0_Height, L0_Width], mode="nearest")

        F = torch.cat((f0,f1_0,f2_0),dim=1)
        F = self.convm3(F)

        return F

    def forward(self, x):
        # scale1_size = [x.size(2), x.size(3)]
        scale2_size = [35, 11]
        scale3_size = [18 , 7]

        wave_scale_1 = x
        wave_scale_2 = interpolate(x, size=scale2_size, mode='bilinear', align_corners=True)
        wave_scale_3 = interpolate(x, size=scale3_size, mode='bilinear', align_corners=True)
        wave_scale_2 = self.convm1(wave_scale_2)
        wave_scale_3 = self.convm1(wave_scale_3)

        mff = self.corss_connection(wave_scale_1,wave_scale_2,wave_scale_3)
        return mff

class Waveguide (Module):
    def __init__(self):
        super(Waveguide, self).__init__()

        self.conv2_1 = NewConv2d.createNetwor(self,22,32,(5,5),(1,1))
        self.pool2_1 = NewPool2d.createNetwor(self, (100,1) , (15,1))

        self.conv2_2 = NewConv2d.createNetwor(self,40,40,(1,18),1)

        self.mff = MFF()


    def forward(self, x):

        wave2_1 = self.conv2_1(x)
        wavep2_2 = self.pool2_1(wave2_1)

        wave_mff = self.mff(wavep2_2)

        wave2_2 = self.conv2_2(wave_mff)
        # wave2_3 = self.conv2_3(wave2_2)

        return wave2_2

class SelfShallow (Module):
    def __init__(self,
                 conv_nonlin=square,
                 pool_nonlin=safe_log):
        self.__dict__.update(locals())
        # del self.self
        super(SelfShallow,self).__init__()

        # self.conv = Sequential()
        self.transpose = Expression(_transpose_time_to_spat)
        self.conv1_1 = nn.Sequential(NewConv2d.createNetwor(self,1, 40, (25, 1), (1, 1)))
        self.conv1_2 = nn.Sequential(NewConv2d.createNetwor(self,40, 40, (1, 22), (1, 1)),
                                     NewBN2d.createNetwor(self,40),
                                     Expression(conv_nonlin))

        self.pool1_1 = nn.Sequential(NewPool2d.createNetwor(self,(75, 1), (15, 1)),

                                     Expression(pool_nonlin),nn.Dropout(p = 0.5))

        self.classfier = nn.Sequential(NewConv2d.createNetwor(self,40, 4, (69,1), (1,1)),
                                       nn.LogSoftmax(dim = 1),
                                       Expression(_squeeze_final_output))

        self.guide = Waveguide()
        # self.recon = Reconstruction()
        self.deconv4 = NewDCT2d.createNetwor(self,40,40,(1,22),(1,1))
        self.conv4_1 = NewConv2d.createNetwor(self,80,40,1,1)
        self.conv5_1 = NewConv2d.createNetwor(self,80,40,1,1)
        self.deconv5 = NewDCT2d.createNetwor(self,40,1,(25,1),(1,1))

        ######################################### 将超参数alpha添加到nn.modle中学习.
        # self.alpha_1 = nn.Parameter(torch.FloatTensor([0.1]).cuda(),requires_grad=True)
        # self.alpha_2 = nn.Parameter(torch.FloatTensor([0.1]).cuda(),requires_grad=True)
        # self.loss_1 = nn.MSELoss()
        # self.loss_2 = nn.MSELoss()
        ##################################################
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         #nn.init.xavier_normal_(m.weight)
        #         #nn.init.constant_(m.bias.data, 0.0)
        #         nn.init.kaiming_normal_(m.weight, mode='fan_in')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        #     elif isinstance(m, nn.ConvTranspose2d):
        #         #nn.init.xavier_normal_(m.weight)
        #         nn.init.kaiming_normal_(m.weight, mode='fan_in')
        #         nn.init.constant_(m.bias.data, 0.0)

    def forward(self,x):
        x = self.transpose(x)

        x_raw = x[:,22,:,:]
        x_raw = x_raw[:,None,:,:]
        x_wave = x[:,:22,:,:]

        batch, C, Height, Width = x_raw.size()

        x_guide = self.guide(x_wave)

        x1_1 = self.conv1_1(x_raw)
        x1_2 = self.conv1_2(x1_1)
        xpool1_1 = self.pool1_1(x1_2)
        output = self.classfier(xpool1_1)

        # x_recon = self.recon(x_guide)
        upsample = interpolate(x_guide, size=[1101, 1],mode='nearest')
        deconv4 = torch.cat((upsample,x1_2),dim=1)
        deconv4 = self.conv4_1(deconv4)
        deconv4 = self.deconv4(deconv4)
        reconstruction = torch.cat((deconv4, x1_1), dim=1)
        reconstruction = self.conv5_1(reconstruction)
        reconstruction = self.deconv5(reconstruction)

        feature1 = xpool1_1.reshape(batch,-1)
        feature2 = x_guide.reshape(batch,-1)
        x_raw = x_raw.reshape(batch,-1)
        reconstruction = reconstruction.reshape(batch,-1)

        # print("self.alpha_1: ",self.alpha_1)
        # print("self.alpha_2: ",self.alpha_2)
        # loss = self.alpha_1*self.loss_1(feature1,feature2)+self.alpha_2*self.loss_2(x_raw,reconstruction)
        #
        # return output,loss
        return output,feature1,feature2,x_raw,reconstruction

if __name__ == '__main__':

    input = torch.rand(32,23,22,1125)
    model = SelfShallow()
    out = model(input)


    print("hello lly")

