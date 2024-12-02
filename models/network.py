import os
import copy
import torch
import torchvision
import numpy as np
import torch.nn as nn
from src.models import create_model
from .ffc import FFCResNetGenerator
import torch.nn.functional as F

from omegaconf import OmegaConf
import yaml


class ConvWithActivation(nn.Module):
    def __init__(self, conv_type, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, activation='relu'):
        super(ConvWithActivation, self).__init__()

        if conv_type == 'conv':
            conv_func = nn.Conv2d 
            self.conv2d = conv_func(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
            self.conv2d = nn.utils.spectral_norm(self.conv2d)
        elif conv_type == 'deconv':
            conv_func = nn.ConvTranspose2d
            self.conv2d = nn.utils.spectral_norm(conv_func(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias))
        self.activation = get_activation(activation)

        for m in self.modules():
            if isinstance(m, conv_func):
                nn.init.kaiming_normal_(m.weight)
        
    def forward(self, x):
        x = self.conv2d(x)
        x = self.activation(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1, use_spectral_norm=True):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            nn.utils.spectral_norm(nn.Conv2d(in_channels=dim, out_channels=256, kernel_size=3, padding=0, dilation=dilation, bias=not use_spectral_norm)),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            nn.utils.spectral_norm(nn.Conv2d(in_channels=256, out_channels=dim, kernel_size=3, padding=0, dilation=1, bias=not use_spectral_norm)),
            # nn.InstanceNorm2d(dim, track_running_stats=False),
        )

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        cnum = 32
        self.global_dis = nn.Sequential(
            ConvWithActivation('conv', 3, 2*cnum, 4, 2, padding=get_pad(256, 5, 2)),
            ConvWithActivation('conv', 2*cnum, 4*cnum, 4, 2, padding=get_pad(128, 5, 2)),
            ConvWithActivation('conv', 4*cnum, 8*cnum, 4, 2, padding=get_pad(64, 5, 2)),
            ConvWithActivation('conv', 8*cnum, 8*cnum, 4, 2, padding=get_pad(32, 5, 2)),
            ConvWithActivation('conv', 8*cnum, 8*cnum, 4, 2, padding=get_pad(16, 5, 2)),
            ConvWithActivation('conv', 8*cnum, 8*cnum, 4, 2, padding=get_pad(8, 5, 2)),            
        )
        self.local_dis = copy.deepcopy(self.global_dis)
        self.fusion = nn.Sequential(
            nn.Conv2d(16*cnum, 1, kernel_size=4),
            nn.Sigmoid()
        )
    
    def forward(self, input, mask):
        global_feat = self.global_dis(input)
        local_feat = self.local_dis(input * (1 - mask))

        concat_feat = torch.cat([global_feat, local_feat], 1)
        # fused_prob = self.fusion(concat_feat) * 2 - 1
        fused_prob = self.fusion(concat_feat).view(input.size()[0], -1) * 2 - 1
        return fused_prob

class VGG16(nn.Module):
    def __init__(self, pretrained_path):
        super(VGG16, self).__init__()
        vgg16 = torchvision.models.vgg16()
        vgg16.load_state_dict(torch.load(pretrained_path))

        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        for i in range(3):
            for param in getattr(self, f'enc_{i+1:d}').parameters():
                param.requires_grad = False 

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, f'enc_{i+1:d}')
            results.append(func(results[-1]))
        return results[1:]

class PixelDecoder(nn.Module):
    def __init__(self, return_intermediate=True):
        super(PixelDecoder, self).__init__()

        self.deconv_list = nn.ModuleList([ 
            ConvWithActivation('deconv', 512, 256, 3, 2, 1),
            ConvWithActivation('deconv', 512, 256, 3, 2, 1),
            ConvWithActivation('deconv', 512, 128, 3, 2, 1),
            ConvWithActivation('deconv', 256, 64, 3, 2, 1),
        ])
        # self.erase_conv = nn.Conv2d(64, 3, 3, 1, 1)
        self.dc3 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, padding=0),
            )

        self.lateral_connection_list = nn.ModuleList([
            build_lateral_connection(256),
            build_lateral_connection(256),
            build_lateral_connection(256),
            build_lateral_connection(128),
        ])

        if return_intermediate:
            self.intermediate_conv1 = nn.Conv2d(256, 3, 3, 1, 1)
            self.intermediate_conv2 = nn.Conv2d(128, 3, 3, 1, 1)

        self.return_intermediate = return_intermediate


        self.feature_fusion_1 = FusionLayer(256,256)
        self.feature_fusion_2 = FusionLayer(128,128)
        self.feature_fusion_3 = FusionLayer(64,64)

    def forward(self, im_f, inter_hs, middle_feat, decode_feat, skip_features):
        outputs = []; mask = None

        bs, c, h, w = im_f.shape

        x = self.deconv_list[0](torch.cat((im_f, self.lateral_connection_list[0](skip_features[-1])),1))

        # import pdb;pdb.set_trace()
        x = self.deconv_list[1](torch.cat((x, self.lateral_connection_list[1](skip_features[-2])),1)) 

        x = self.feature_fusion_1(x, decode_feat[0])
        outputs.append(self.intermediate_conv1(x))

        x = self.deconv_list[2](torch.cat((x, self.lateral_connection_list[2](skip_features[-3])),1))

        x = self.feature_fusion_2(x, decode_feat[1]) 
        outputs.append(self.intermediate_conv2(x))

        x = self.deconv_list[3](torch.cat((x, self.lateral_connection_list[3](skip_features[-4])),1)) 

        x = self.feature_fusion_3(x, decode_feat[2])
        outputs.append(self.dc3(x))

        if self.return_intermediate:
            return outputs
        else:
            return outputs[-1]

class ConvWithActivation_str(torch.nn.Module):
    """
    SN convolution for spetral normalization conv
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(ConvWithActivation_str, self).__init__()
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv2d = torch.nn.utils.spectral_norm(self.conv2d)
        self.activation = activation
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
    def forward(self, input):
        x = self.conv2d(input)
        if self.activation is not None:
            return self.activation(x)
        else:
            return x

class DeConvWithActivation(torch.nn.Module):
    """
    SN convolution for spetral normalization deconv
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, output_padding=0, dilation=1, groups=1, bias=True, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(DeConvWithActivation, self).__init__()
        self.conv2d = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding,output_padding, dilation, groups, bias)
        self.conv2d = torch.nn.utils.spectral_norm(self.conv2d)
        self.activation = activation
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
    def forward(self, input):
        x = self.conv2d(input)
        if self.activation is not None:
            return self.activation(x)
        else:
            return x

class Downsample_connect(nn.Module):
    def __init__(self, in_channels, out_channels, same_shape=True, **kwargs):
        super(Downsample_connect,self).__init__()
        self.same_shape = same_shape
        strides = 1 if same_shape else 2
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,stride=strides)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        # self.conv2 = torch.nn.utils.spectral_norm(self.conv2)
        if not same_shape:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1,
            # self.conv3 = nn.Conv2D(channels, kernel_size=3, padding=1,
                                 stride=strides)
            # self.conv3 = torch.nn.utils.spectral_norm(self.conv3)
        self.batch_norm2d = nn.BatchNorm2d(out_channels)

    def forward(self,x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        if not self.same_shape:
            x = self.conv3(x)
        out = self.batch_norm2d(out + x)
        # out = out + x
        return F.relu(out)

class StructureGen(nn.Module):
    def __init__(self, n_in_channel=3):
        super(StructureGen, self).__init__()
        #downsample
        self.conv1 = ConvWithActivation_str(4,32,kernel_size=4,stride=2,padding=1)
        self.conva = ConvWithActivation_str(32,32,kernel_size=3, stride=1, padding=1)
        self.convb = ConvWithActivation_str(32,64, kernel_size=4, stride=2, padding=1)
        self.res1 = Downsample_connect(64,64)
        self.res2 = Downsample_connect(64,64)
        self.res3 = Downsample_connect(64,128,same_shape=False)
        self.res4 = Downsample_connect(128,128)
        self.res5 = Downsample_connect(128,256,same_shape=False)
       # self.nn = ConvWithActivation(256, 512, 3, 1, dilation=2, padding=get_pad(64, 3, 1, 2))
        self.res6 = Downsample_connect(256,256)
        self.res7 = Downsample_connect(256,512,same_shape=False)
        self.res8 = Downsample_connect(512,512)
        self.conv2 = ConvWithActivation_str(512,512,kernel_size=1)

        #upsample
        self.deconv1 = DeConvWithActivation(512,256,kernel_size=3,padding=1,stride=2,output_padding=1)
        self.deconv2 = DeConvWithActivation(256*2,128,kernel_size=3,padding=1,stride=2,output_padding=1)
        self.deconv3 = DeConvWithActivation(128*2,64,kernel_size=3,padding=1,stride=2,output_padding=1)
        self.deconv4 = DeConvWithActivation(64*2,32,kernel_size=3,padding=1,stride=2,output_padding=1)
        self.deconv5 = DeConvWithActivation(64,3,kernel_size=3,padding=1,stride=2,output_padding=1)

        #lateral connection 
        self.lateral_connection1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, padding=0,stride=1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1,stride=1),
            nn.Conv2d(512, 512, kernel_size=3, padding=1,stride=1),
            nn.Conv2d(512, 256, kernel_size=1, padding=0,stride=1),)
        self.lateral_connection2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, padding=0,stride=1),
            nn.Conv2d(128, 256, kernel_size=3, padding=1,stride=1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1,stride=1),
            nn.Conv2d(256, 128, kernel_size=1, padding=0,stride=1),)
        self.lateral_connection3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, padding=0,stride=1),
            nn.Conv2d(64, 128, kernel_size=3, padding=1,stride=1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1,stride=1),
            nn.Conv2d(128, 64, kernel_size=1, padding=0,stride=1),)
        self.lateral_connection4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=1, padding=0,stride=1),
            nn.Conv2d(32, 64, kernel_size=3, padding=1,stride=1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1,stride=1),
            nn.Conv2d(64, 32, kernel_size=1, padding=0,stride=1),)  

    def forward(self, x):
        x = self.conv1(x)
        x = self.conva(x)
        con_x1 = x
       # import pdb;pdb.set_trace()
        x = self.convb(x)
        x = self.res1(x)
        con_x2 = x
        x = self.res2(x)
        x = self.res3(x)
        con_x3 = x
        x = self.res4(x)
        x = self.res5(x)
        con_x4 = x
        x = self.res6(x)

       # import pdb;pdb.set_trace()
        x = self.res7(x)
        x = self.res8(x)
        x = self.conv2(x)
        #upsample
        x = self.deconv1(x)

        x = torch.cat([self.lateral_connection1(con_x4), x], dim=1)
        x = self.deconv2(x)
        x = torch.cat([self.lateral_connection2(con_x3), x], dim=1)
        x = self.deconv3(x)
        xo1 = x
        x = torch.cat([self.lateral_connection3(con_x2), x], dim=1)
        x = self.deconv4(x)
        xo2 = x
        x = torch.cat([self.lateral_connection4(con_x1), x], dim=1)
        x = self.deconv5(x)  
        x_o_unet = x       
        return x_o_unet 
    
class FusionLayer(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super(FusionLayer, self).__init__()

        self.structure_gate = nn.Sequential(
            nn.Conv2d(in_channels=in_channels + in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        self.texture_gamma = nn.Parameter(torch.zeros(1))

        self.attn = SimAM()

    def forward(self, image_feature, inpaint_feature):

        energy = torch.cat((image_feature, inpaint_feature), dim=1)
        energy = self.attn(energy)

        fusion_f = self.structure_gate(energy)

        texture_feature = image_feature + self.texture_gamma * (fusion_f * inpaint_feature)

        return texture_feature


class SimAM(torch.nn.Module):
    def __init__(self, channels = None, e_lambda = 1e-4):
        super(SimAM, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):

        b, c, h, w = x.size()
        
        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2,3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2,3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)

def build_lateral_connection(input_dim):
    return nn.Sequential(
        nn.Conv2d(input_dim, input_dim, 1, 1, 0),
        nn.Conv2d(input_dim, input_dim*2, 3, 1, 1),
        nn.Conv2d(input_dim*2, input_dim*2, 3, 1, 1),
        nn.Conv2d(input_dim*2, input_dim, 1, 1, 0)
    )

def get_activation(type):
    if type == 'leaky relu':
        return nn.LeakyReLU(0.2, inplace=True)
    elif type == 'relu':
        return nn.ReLU(inplace=True)
    elif type == 'sigmoid':
        return nn.Sigmoid()
    else:
        raise NotImplementedError

def get_pad(in_,  ksize, stride, atrous=1):
    out_ = np.ceil(float(in_)/stride)
    return int(((out_ - 1) * stride + atrous*(ksize-1) + 1 - in_)/2)

def build_feature_extractor(args):
    pretrained_path = os.path.join(args.code_dir, 'pretrained/vgg16-397923af.pth')
    return VGG16(pretrained_path)

def build_discriminator(args):
    return Discriminator()

def build_ffc(args):
    with open('./ffc_model/ffc.yaml', 'r') as f:
        ffc_config = OmegaConf.create(yaml.safe_load(f))
    ffc_inpaint = FFCResNetGenerator(**ffc_config.generator)
    state_gen = torch.load('./ffc_model/best.ckpt', map_location='cpu')
    pretrained_model = state_gen['state_dict']
    model_dict = ffc_inpaint.state_dict()
    new_dict = {k.split('generator.')[-1]: v for k, v in pretrained_model.items() if k.split('generator.')[-1] in model_dict.keys()}
    model_dict.update(new_dict)
    ffc_inpaint.load_state_dict(model_dict, strict=True)
    return ffc_inpaint

def build_pixel_decoder(args):
    return PixelDecoder()

def build_structure(args):
    model = StructureGen()
    state_gen = torch.load('Structure_Gen.pth', map_location='cpu')
    model.load_state_dict(state_gen)    
    return model