"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import numpy as np

import torch
from torch import nn
from torch import autograd

from blocks import LinearBlock, Conv2dBlock, ResBlocks, ActFirstResBlock, ResidualOctBlock_basic, ActFirst_no_normalization, oct_conv7x7, norm_conv7x7,oct_conv5x5,oct_conv4x4, norm_conv4x4, oct_conv3x3, norm_conv3x3, Oct_conv_norm, conv_norm, Oct_conv_reLU, Oct_conv_lreLU, Oct_conv_up  
import pdb


def assign_adain_params(adain_params, model):
    # assign the adain_params to the AdaIN layers in model
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
            mean = adain_params[:, :m.num_features]
            std = adain_params[:, m.num_features:2*m.num_features]
            m.bias = mean.contiguous().view(-1)
            m.weight = std.contiguous().view(-1)
            if adain_params.size(1) > 2*m.num_features:
                adain_params = adain_params[:, 2*m.num_features:]


def get_num_adain_params(model):
    # return the number of AdaIN parameters needed by the model
    num_adain_params = 0
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
            num_adain_params += 2*m.num_features
    return num_adain_params


##################################################################################
# Discriminator
##################################################################################
class GPPatchMcResDis_yaxing(nn.Module):
    def __init__(self, hp):
        super(GPPatchMcResDis, self).__init__()
        assert hp['n_res_blks'] % 2 == 0, 'n_res_blk must be multiples of 2'
        self.n_layers = hp['n_res_blks'] // 2
        nf = hp['nf']
        #cnn_f = [Conv2dBlock(3, nf, 7, 1, 3,
        #                     pad_type='reflect',
        #                     norm='none',
        #                     activation='none')]
        oct_conv_on = True
        conv7x7 = oct_conv7x7 if oct_conv_on else norm_conv7x7
        conv4x4 = oct_conv4x4 if oct_conv_on else norm_conv4x4
        conv3x3 = oct_conv3x3 if oct_conv_on else norm_conv3x3
        conv1x1 = oct_conv1x1 if oct_conv_on else norm_conv1x1
        norm_func = Oct_conv_norm if oct_conv_on else conv_norm
        act_func = Oct_conv_reLU if oct_conv_on else nn.ReLU
        avgpool = Oct_avgpool if oct_conv_on else nn.AvgPool2d
        reflePad = Oct_reflectionPad if oct_conv_on else nn.ReflectionPad2d()

        # initialize padding: for input image
        self.pad1 = nn.ReflectionPad2d(padding=3)
        # first layer Conv-64, original paper only use conv
        self.conv1 = conv7x7(in_planes=3, out_planes=nf, padding=3, type="first")

        # loop1 
        # first resblock128
        # second resblock128: firstly conduct conv1*1, then 2 conv3*3, final avgpool
        # conv1*1:padding is 0, note: original code is ActFirst which firstly perform conv1*1, then 2 conv3*3
        self.res1 = ActFirst_no_normalization(conv_dim=nf, oct_conv_on = oct_conv_on, norm='none')
        self.conv2 = conv1x1(in_planes=nf, out_planes=nf*2, padding=0, type="normal")
        self.res2 = ActFirst_no_normalization(conv_dim=nf*2, oct_conv_on = oct_conv_on, norm='none')
        self.pad2 = reflePad(padding=1)
        self.avgp1 = avgpool(kernel_size=3, stride=2)

        # loop2 
        self.res3 = ActFirst_no_normalization(conv_dim=nf*2, oct_conv_on = oct_conv_on, norm='none')
        self.conv3 = conv1x1(in_planes=nf*2, out_planes=nf*4, padding=0, type="normal")
        self.res4 = ActFirst_no_normalization(conv_dim=nf*4, oct_conv_on = oct_conv_on, norm='none')
        self.pad3 = reflePad(padding=1)
        self.avgp2 = avgpool(kernel_size=3, stride=2)

        # loop3 
        self.res5 = ActFirst_no_normalization(conv_dim=nf*4, oct_conv_on = oct_conv_on, norm='none')
        self.conv4 = conv1x1(in_planes=nf*4, out_planes=nf*8, padding=0, type="normal")
        self.res6 = ActFirst_no_normalization(conv_dim=nf*8, oct_conv_on = oct_conv_on, norm='none')
        self.pad4 = reflePad(padding=1)
        self.avgp3 = avgpool(kernel_size=3, stride=2)
        # loop4 
        self.res7 = ActFirst_no_normalization(conv_dim=nf*8, oct_conv_on = oct_conv_on, norm='none')
        self.conv5 = conv1x1(in_planes=nf*8, out_planes=nf*16, padding=0, type="normal")
        self.res8 = ActFirst_no_normalization(conv_dim=nf*16, oct_conv_on = oct_conv_on, norm='none')
        self.pad5 = reflePad(padding=1)
        self.avgp4 = avgpool(kernel_size=3, stride=2)
        # final block 
        self.res9 = ActFirst_no_normalization(conv_dim=nf*16, oct_conv_on = oct_conv_on, norm='none')
        self.res10 = ActFirst_no_normalization(conv_dim=nf*16, oct_conv_on = oct_conv_on, norm='none')
        # extract layer: to merget tow braches 
        self.leakre1 =Oct_conv_lreLU(negative_slope=0.2, inplace=False)
        self.cnn_f = conv1x1(in_planes=nf*16, out_planes=nf*16, padding=0, type="last")
        self.leakre2 = nn.LeakyReLU(0.2, inplace=False)
        self.cnn_c = norm_conv1x1(in_planes=nf*16, out_planes=hp['num_classes'], padding=0)

    def forward(self, x, y, alpha_in, alpha_out):
        assert(x.size(0) == y.size(0))
        # feature
        #output = self.pad1(x)
        output = self.conv1(x, alpha_in=alpha_in, alpha_out=alpha_out)
        # loop1 
        output = self.res1(output, alpha_in=alpha_in, alpha_out=alpha_out) 
        output = self.conv2(output, alpha_in=alpha_in, alpha_out=alpha_out) 
        output = self.res2(output, alpha_in=alpha_in, alpha_out=alpha_out) 
        output = self.pad2(output) 
        output = self.avgp1(output) 

        # loop2 
        output = self.res3(output, alpha_in=alpha_in, alpha_out=alpha_out) 
        output = self.conv3(output, alpha_in=alpha_in, alpha_out=alpha_out) 
        output = self.res4(output, alpha_in=alpha_in, alpha_out=alpha_out) 
        output = self.pad3(output) 
        output = self.avgp2(output) 

        # loop3 
        output = self.res5(output, alpha_in=alpha_in, alpha_out=alpha_out) 
        output = self.conv4(output, alpha_in=alpha_in, alpha_out=alpha_out) 
        output = self.res6(output, alpha_in=alpha_in, alpha_out=alpha_out) 
        output = self.pad4(output) 
        output = self.avgp3(output) 
        # loop4 
        output = self.res7(output, alpha_in=alpha_in, alpha_out=alpha_out) 
        output = self.conv5(output, alpha_in=alpha_in, alpha_out=alpha_out) 
        output = self.res8(output, alpha_in=alpha_in, alpha_out=alpha_out) 
        output = self.pad5(output) 
        output = self.avgp4(output) 
        # final block 
        output = self.res9(output, alpha_in=alpha_in, alpha_out=alpha_out) 
        output = self.res10(output, alpha_in=alpha_in, alpha_out=alpha_out) 
        # extract layer: to merget tow braches 
        output = self.leakre1(output)
        feat = self.cnn_f(output, alpha_in=alpha_in, alpha_out=alpha_out) 
        feat1 = self.leakre2(feat)
        out = self.cnn_c(feat1)
        index = torch.LongTensor(range(out.size(0))).cuda()
        out = out[index, y, :, :]

       # feat = self.cnn_f(x)
       # out = self.cnn_c(feat)
       # index = torch.LongTensor(range(out.size(0))).cuda()
       # out = out[index, y, :, :]
        return out, feat
    def calc_dis_fake_loss(self, input_fake, input_label, octave_alpha):
        resp_fake, gan_feat = self.forward(input_fake, input_label, alpha_in=octave_alpha, alpha_out=octave_alpha)
        total_count = torch.tensor(np.prod(resp_fake.size()),
                                   dtype=torch.float).cuda()
        fake_loss = torch.nn.ReLU()(1.0 + resp_fake).mean()
        correct_count = (resp_fake < 0).sum()
        fake_accuracy = correct_count.type_as(fake_loss) / total_count
        return fake_loss, fake_accuracy, resp_fake

    def calc_dis_real_loss(self, input_real, input_label, octave_alpha):
        resp_real, gan_feat = self.forward(input_real, input_label, alpha_in=octave_alpha, alpha_out=octave_alpha)
        total_count = torch.tensor(np.prod(resp_real.size()),
                                   dtype=torch.float).cuda()
        real_loss = torch.nn.ReLU()(1.0 - resp_real).mean()
        correct_count = (resp_real >= 0).sum()
        real_accuracy = correct_count.type_as(real_loss) / total_count
        return real_loss, real_accuracy, resp_real

    def calc_gen_loss(self, input_fake, input_fake_label, octave_alpha):
        resp_fake, gan_feat = self.forward(input_fake, input_fake_label, alpha_in=octave_alpha, alpha_out=octave_alpha)
        total_count = torch.tensor(np.prod(resp_fake.size()),
                                   dtype=torch.float).cuda()
        loss = -resp_fake.mean()
        correct_count = (resp_fake >= 0).sum()
        accuracy = correct_count.type_as(loss) / total_count
        return loss, accuracy, gan_feat

    def calc_grad2(self, d_out, x_in):
        batch_size = x_in.size(0)
        grad_dout = autograd.grad(outputs=d_out.mean(),
                                  inputs=x_in,
                                  create_graph=True,
                                  retain_graph=True,
                                  only_inputs=True)[0]
        grad_dout2 = grad_dout.pow(2)
        assert (grad_dout2.size() == x_in.size())
        reg = grad_dout2.sum()/batch_size
        return reg



class GPPatchMcResDis(nn.Module):
    def __init__(self, hp):
        super(GPPatchMcResDis, self).__init__()
        assert hp['n_res_blks'] % 2 == 0, 'n_res_blk must be multiples of 2'
        self.n_layers = hp['n_res_blks'] // 2
        nf = hp['nf']
        cnn_f = [Conv2dBlock(3, nf, 7, 1, 3,
                             pad_type='reflect',
                             norm='none',
                             activation='none')]
        for i in range(self.n_layers - 1):
            nf_out = np.min([nf * 2, 1024])
            cnn_f += [ActFirstResBlock(nf, nf, None, 'lrelu', 'none')]
            cnn_f += [ActFirstResBlock(nf, nf_out, None, 'lrelu', 'none')]
            cnn_f += [nn.ReflectionPad2d(1)]
            cnn_f += [nn.AvgPool2d(kernel_size=3, stride=2)]
            nf = np.min([nf * 2, 1024])
        nf_out = np.min([nf * 2, 1024])
        cnn_f += [ActFirstResBlock(nf, nf, None, 'lrelu', 'none')]
        cnn_f += [ActFirstResBlock(nf, nf_out, None, 'lrelu', 'none')]
        cnn_c = [Conv2dBlock(nf_out, hp['num_classes'], 1, 1,
                             norm='none',
                             activation='lrelu',
                             activation_first=True)]
        self.cnn_f = nn.Sequential(*cnn_f)
        self.cnn_c = nn.Sequential(*cnn_c)

    def forward(self, x, y, alpha_in=0., alpha_out=0.):
        assert(x.size(0) == y.size(0))
        feat = self.cnn_f(x)
        out = self.cnn_c(feat)
        index = torch.LongTensor(range(out.size(0))).cuda()
        out = out[index, y, :, :]
        return out, feat

    def calc_dis_fake_loss(self, input_fake, input_label,octave_alpha):
        resp_fake, gan_feat = self.forward(input_fake, input_label)
        total_count = torch.tensor(np.prod(resp_fake.size()),
                                   dtype=torch.float).cuda()
        fake_loss = torch.nn.ReLU()(1.0 + resp_fake).mean()
        correct_count = (resp_fake < 0).sum()
        fake_accuracy = correct_count.type_as(fake_loss) / total_count
        return fake_loss, fake_accuracy, resp_fake

    def calc_dis_real_loss(self, input_real, input_label,octave_alpha):
        resp_real, gan_feat = self.forward(input_real, input_label)
        total_count = torch.tensor(np.prod(resp_real.size()),
                                   dtype=torch.float).cuda()
        real_loss = torch.nn.ReLU()(1.0 - resp_real).mean()
        correct_count = (resp_real >= 0).sum()
        real_accuracy = correct_count.type_as(real_loss) / total_count
        return real_loss, real_accuracy, resp_real

    def calc_gen_loss(self, input_fake, input_fake_label,octave_alpha):
        resp_fake, gan_feat = self.forward(input_fake, input_fake_label)
        total_count = torch.tensor(np.prod(resp_fake.size()),
                                   dtype=torch.float).cuda()
        loss = -resp_fake.mean()
        correct_count = (resp_fake >= 0).sum()
        accuracy = correct_count.type_as(loss) / total_count
        return loss, accuracy, gan_feat

    def calc_grad2(self, d_out, x_in):
        batch_size = x_in.size(0)
        grad_dout = autograd.grad(outputs=d_out.mean(),
                                  inputs=x_in,
                                  create_graph=True,
                                  retain_graph=True,
                                  only_inputs=True)[0]
        grad_dout2 = grad_dout.pow(2)
        assert (grad_dout2.size() == x_in.size())
        reg = grad_dout2.sum()/batch_size
        return reg


class FewShotGen(nn.Module):
    def __init__(self, hp):
        super(FewShotGen, self).__init__()
        nf = hp['nf']
        nf_mlp = hp['nf_mlp']
        down_class = hp['n_downs_class']
        down_content = hp['n_downs_content']
        n_mlp_blks = hp['n_mlp_blks']
        n_res_blks = hp['n_res_blks']
        latent_dim = hp['latent_dim']
        self.enc_class_model = ClassModelEncoder(down_class,
                                                 3,
                                                 nf,
                                                 latent_dim,
                                                 norm='none',
                                                 activ='relu',
                                                 pad_type='reflect')

        self.enc_content = ContentEncoder(down_content,
                                          n_res_blks,
                                          3,
                                          nf,
                                          'in',
                                          activ='relu',
                                          pad_type='reflect')

        self.dec = Decoder(down_content,
                           n_res_blks,
                           self.enc_content.output_dim,
                           3,
                           res_norm='adain',
                           activ='relu',
                           pad_type='reflect')

        self.mlp = MLP(latent_dim,
                       get_num_adain_params(self.dec),
                       nf_mlp,
                       n_mlp_blks,
                       norm='none',
                       activ='relu')

    def forward(self, one_image, model_set, octave_alpha):
        # reconstruct an image
        pdb.set_trace()
        content, model_codes = self.encode(one_image, model_set, alpha_in=octave_alpha, alpha_out=octave_alpha)
        model_code = torch.mean(model_codes, dim=0).unsqueeze(0)
        images_trans = self.decode(content, model_code, alpha_in=octave_alpha, alpha_out=octave_alpha)
        return images_trans

    def encode(self, one_image, model_set, octave_alpha):
        # extract content code from the input image
        content = self.enc_content(one_image, alpha_in=octave_alpha, alpha_out=octave_alpha)
        # extract model code from the images in the model set
        class_codes = self.enc_class_model(model_set, alpha_in=octave_alpha, alpha_out=octave_alpha)
        class_code = torch.mean(class_codes, dim=0).unsqueeze(0)
        return content, class_code

    def decode(self, content, model_code, octave_alpha):
        # decode content and style codes to an image
        adain_params = self.mlp(model_code)
        assign_adain_params(adain_params, self.dec)
        images = self.dec(content, alpha_in=octave_alpha, alpha_out=octave_alpha)
        return images

class ClassModelEncoder(nn.Module):
    def __init__(self, downs, ind_im, dim, latent_dim, norm, activ, pad_type):
        super(ClassModelEncoder, self).__init__()
        norm=='in'
        oct_conv_on = True
        alpha_in, alpha_out = 0.5, 0.5
        conv7x7 = oct_conv7x7 if oct_conv_on else norm_conv7x7
        conv4x4 = oct_conv4x4 if oct_conv_on else norm_conv4x4
        conv3x3 = oct_conv3x3 if oct_conv_on else norm_conv3x3
        norm_func = Oct_conv_norm if oct_conv_on else conv_norm
        act_func = Oct_conv_reLU if oct_conv_on else nn.ReLU

        # first layer:keep same resolution,
        # both of alpha_in and  alpha_out does not matter 
        self.conv1 = conv7x7(in_planes=ind_im, out_planes=dim, alpha_in=alpha_in, alpha_out=alpha_out, padding=3, type="first")
        self.bn1 = norm_func(planes = dim, norm = norm)
        self.re1 = act_func(inplace=False)

        # second layer:redcue resolution hafives time
        self.conv2 = conv4x4(in_planes=dim, out_planes=dim*2, alpha_in=alpha_in, alpha_out=alpha_out, padding=1, type="normal")
        self.bn2 = norm_func(planes = dim*2, norm = norm)
        self.re2 = act_func(inplace=True)

        # three layer:redcue resolution hafives time
        self.conv3 = conv4x4(in_planes=dim*2, out_planes=dim*4, alpha_in=alpha_in, alpha_out=alpha_out, padding=1, type="last")
        self.bn3 = conv_norm(planes = dim*4, norm = norm)
        self.re3 = nn.ReLU(inplace=True)

        self.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d(1) # global average pooling
        self.Conv2d = nn.Conv2d(dim*4, latent_dim, 1, 1, 0)
        self.output_dim = dim

        #self.model = []
        #self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        #for i in range(2):
        #    self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
        #    dim *= 2
       ## for i in range(n_downsample - 2):
       ##     self.model += [Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
       # self.model += [nn.AdaptiveAvgPool2d(1)] # global average pooling
       # self.model += [nn.Conv2d(dim, style_dim, 1, 1, 0)]
       # self.model = nn.Sequential(*self.model)
       # self.output_dim = dim

#    def forward(self, x):
#        return self.model(x)

    def forward(self, x, alpha_in, alpha_out):
        # no norm
        # first layer:keep same resolution
        out=self.conv1(x, alpha_in=alpha_in, alpha_out=alpha_out)
        #out=self.bn1(out)
        out=self.re1(out)
        # second layer:redcue resolution hafives time
        out=self.conv2(out, alpha_in=alpha_in, alpha_out=alpha_out)
       # out=self.bn2(out)
        out=self.re2(out)
        # three layer:redcue resolution hafives time
        out=self.conv3(out, alpha_in=1, alpha_out=1)
       # out=self.bn3(out)
        out=self.re3(out)
        out=self.AdaptiveAvgPool2d(out) # global average pooling
        out=self.Conv2d(out)
        return out


class ClassModelEncoder_original(nn.Module):
    def __init__(self, downs, ind_im, dim, latent_dim, norm, activ, pad_type):
        super(ClassModelEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(ind_im, dim, 7, 1, 3,
                                   norm=norm,
                                   activation=activ,
                                   pad_type=pad_type)]
        for i in range(2):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1,
                                       norm=norm,
                                       activation=activ,
                                       pad_type=pad_type)]
            dim *= 2
        for i in range(downs - 2):
            self.model += [Conv2dBlock(dim, dim, 4, 2, 1,
                                       norm=norm,
                                       activation=activ,
                                       pad_type=pad_type)]
        self.model += [nn.AdaptiveAvgPool2d(1)]
        self.model += [nn.Conv2d(dim, latent_dim, 1, 1, 0)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)



class ContentEncoder(nn.Module):
    def __init__(self, downs, n_res, input_dim, dim, norm, activ, pad_type):
        super(ContentEncoder, self).__init__()
        norm=='in'
        oct_conv_on = True
        alpha_in, alpha_out = 0.5, 0.5
        conv7x7 = oct_conv7x7 if oct_conv_on else norm_conv7x7
        conv4x4 = oct_conv4x4 if oct_conv_on else norm_conv4x4
        conv3x3 = oct_conv3x3 if oct_conv_on else norm_conv3x3
        norm_func = Oct_conv_norm if oct_conv_on else conv_norm
        act_func = Oct_conv_reLU if oct_conv_on else nn.ReLU

        # first layer:keep same resolution
        # both of alpha_in and  alpha_out does not matter 
        self.conv1 = conv7x7(in_planes=input_dim, out_planes=dim, alpha_in=alpha_in, alpha_out=alpha_out, padding=3, type="first")
        self.bn1 = norm_func(planes = dim, norm = norm)
        self.re1 = act_func(inplace=False)

        # second layer:redcue resolution hafives time
        self.conv2 = conv4x4(in_planes=dim, out_planes=dim*2, alpha_in=alpha_in, alpha_out=alpha_out, padding=1, type="normal")
        self.bn2 = norm_func(planes = dim*2, norm = norm)
        self.re2 = act_func(inplace=False)

        # three layer:redcue resolution hafives time
        self.conv3 = conv4x4(in_planes=dim*2, out_planes=dim*4, alpha_in=alpha_in, alpha_out=alpha_out, padding=1, type="normal")
        self.bn3 = norm_func(planes = dim*4, norm = norm)
        self.re3 = act_func(inplace=False)
        # residual blocks
        self.Res1 = ResidualOctBlock_basic(conv_dim=dim*4, alpha_in=alpha_in, alpha_out=alpha_out, oct_conv_on = oct_conv_on, norm=norm)
        self.Res2 = ResidualOctBlock_basic(conv_dim=dim*4, alpha_in=alpha_in, alpha_out=alpha_out, oct_conv_on = oct_conv_on, norm=norm)
        self.Res3 = ResidualOctBlock_basic(conv_dim=dim*4, alpha_in=alpha_in, alpha_out=alpha_out, oct_conv_on = oct_conv_on, norm=norm)
        self.Res4 = ResidualOctBlock_basic(conv_dim=dim*4, alpha_in=alpha_in, alpha_out=alpha_out, oct_conv_on = oct_conv_on, norm=norm)
        self.Res5 = ResidualOctBlock_basic(conv_dim=dim*4, alpha_in=alpha_in, alpha_out=alpha_out, oct_conv_on = oct_conv_on, norm=norm)
        self.Res6 = ResidualOctBlock_basic(conv_dim=dim*4, alpha_in=alpha_in, alpha_out=alpha_out, oct_conv_on = oct_conv_on, norm=norm)

       # self.model = []
       # self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
       # # downsampling blocks
       # for i in range(n_downsample):
       #     self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
       #     dim *= 2
        # residual blocks
       # self.model += [ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]
       # self.model = nn.Sequential(*self.model)
        self.output_dim = dim

   # def forward(self, x):
   #     return self.model(x)
    def forward(self, x, alpha_in, alpha_out):
        # first layer:keep same resolution
        out=self.conv1(x, alpha_in=alpha_in, alpha_out=alpha_out)
        out=self.bn1(out)
        out=self.re1(out)
        # second layer:redcue resolution hafives time
        out=self.conv2(out, alpha_in=alpha_in, alpha_out=alpha_out)
        out=self.bn2(out)
        out=self.re2(out)
        # three layer:redcue resolution hafives time
        out=self.conv3(out, alpha_in=alpha_in, alpha_out=alpha_out)
        out=self.bn3(out)
        out=self.re3(out)

        # residual blocks
        out = self.Res1(out, alpha_in, alpha_out)
        out = self.Res2(out, alpha_in, alpha_out)
        out = self.Res3(out, alpha_in, alpha_out)
        out = self.Res4(out, alpha_in, alpha_out)
        out = self.Res5(out, alpha_in, alpha_out)
        out = self.Res6(out, alpha_in, alpha_out)

        return out


class ContentEncoder_funit(nn.Module):
    def __init__(self, downs, n_res, input_dim, dim, norm, activ, pad_type):
        super(ContentEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3,
                                   norm=norm,
                                   activation=activ,
                                   pad_type=pad_type)]
        for i in range(downs):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1,
                                       norm=norm,
                                       activation=activ,
                                       pad_type=pad_type)]
            dim *= 2
        self.model += [ResBlocks(n_res, dim,
                                 norm=norm,
                                 activation=activ,
                                 pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)



class Decoder(nn.Module):
    def __init__(self, ups, n_res, dim, out_dim, res_norm='adain', activ='relu', pad_type='zero'):
        super(Decoder, self).__init__()
        oct_conv_on = True
        norm='in'
        alpha_in, alpha_out = 0.5, 0.5
        conv7x7 = oct_conv7x7 if oct_conv_on else norm_conv7x7
        conv5x5 = oct_conv5x5 if oct_conv_on else norm_conv5x5
        conv4x4 = oct_conv4x4 if oct_conv_on else norm_conv4x4
        conv3x3 = oct_conv3x3 if oct_conv_on else norm_conv3x3
        norm_func = Oct_conv_norm if oct_conv_on else conv_norm
        act_func = Oct_conv_reLU if oct_conv_on else nn.ReLU
        up_func = Oct_conv_up if oct_conv_on else nn.Upsample 

        # residual blocks
        self.Res1 = ResidualOctBlock_basic(conv_dim=4*dim, alpha_in=alpha_in, alpha_out=alpha_out, oct_conv_on = oct_conv_on, norm=res_norm)
        self.Res2 = ResidualOctBlock_basic(conv_dim=4*dim, alpha_in=alpha_in, alpha_out=alpha_out, oct_conv_on = oct_conv_on, norm=res_norm)
        self.Res3 = ResidualOctBlock_basic(conv_dim=4*dim, alpha_in=alpha_in, alpha_out=alpha_out, oct_conv_on = oct_conv_on, norm=res_norm)
        self.Res4 = ResidualOctBlock_basic(conv_dim=4*dim, alpha_in=alpha_in, alpha_out=alpha_out, oct_conv_on = oct_conv_on, norm=res_norm)
        self.Res5 = ResidualOctBlock_basic(conv_dim=4*dim, alpha_in=alpha_in, alpha_out=alpha_out, oct_conv_on = oct_conv_on, norm=res_norm)
        self.Res6 = ResidualOctBlock_basic(conv_dim=4*dim, alpha_in=alpha_in, alpha_out=alpha_out, oct_conv_on = oct_conv_on, norm=res_norm)

        # first layer:double resolution
        dim = 4*dim
        self.up1 = up_func(scale_factor=2)
        self.conv1 = conv3x3(in_planes=dim, out_planes=dim//2, alpha_in=alpha_in, alpha_out=alpha_out, padding=1, type="normal")
        self.bn1 = norm_func(planes = dim//2, norm = norm)
        self.re1 = act_func(inplace=True)

        # second layer:double resolution 
        self.up2 = up_func(scale_factor=2)
        self.conv2 = conv3x3(in_planes=dim//2, out_planes=dim//4, alpha_in=alpha_in, alpha_out=alpha_out, padding=1,  type="normal")
        self.bn2 = norm_func(planes = dim//4, norm = norm)
        self.re2 = act_func(inplace=False)
        # three layer:keep same resolution
        self.conv3 = conv7x7(in_planes=dim//4, out_planes=out_dim, alpha_in=alpha_in, alpha_out=alpha_out, padding=3, type="last")
        self.tanh = nn.Tanh()



       # self.model = []
       # # AdaIN residual blocks
       # self.model += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
       # # upsampling blocks
       # for i in range(n_upsample):
       #     self.model += [nn.Upsample(scale_factor=2),
       #                    Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
       #     dim //= 2
       # # use reflection padding in the last conv layer
       # self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)]
       # self.model = nn.Sequential(*self.model)

   # def forward(self, x):
   #     return self.model(x)
    def forward(self, x, alpha_in, alpha_out):

        # residual blocks
        out = self.Res1(x, alpha_in, alpha_out)
        out = self.Res2(out, alpha_in, alpha_out)
        out = self.Res3(out, alpha_in, alpha_out)
        out = self.Res4(out, alpha_in, alpha_out)
        out = self.Res5(out, alpha_in, alpha_out)
        out = self.Res6(out, alpha_in, alpha_out)

        # first layer:double resolution
        out=self.up1(out)
        out=self.conv1(out, alpha_in, alpha_out)
        out=self.bn1(out)
        out=self.re1(out)
        # second layer:double resolution hafives time
        out=self.up2(out)
        out=self.conv2(out, alpha_in, alpha_out)
        out=self.bn2(out)
        out=self.re2(out)
        # three layer:keep same resolution 
        out=self.conv3(out, alpha_in, alpha_out)
        out=self.tanh(out)

        return out

class Decoder_funit(nn.Module):
    def __init__(self, ups, n_res, dim, out_dim, res_norm, activ, pad_type):
        super(Decoder, self).__init__()

        self.model = []
        self.model += [ResBlocks(n_res, dim, res_norm,
                                 activ, pad_type=pad_type)]
        for i in range(ups):
            self.model += [nn.Upsample(scale_factor=2),
                           Conv2dBlock(dim, dim // 2, 5, 1, 2,
                                       norm='in',
                                       activation=activ,
                                       pad_type=pad_type)]
            dim //= 2
        self.model += [Conv2dBlock(dim, out_dim, 7, 1, 3,
                                   norm='none',
                                   activation='tanh',
                                   pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, dim, n_blk, norm, activ):

        super(MLP, self).__init__()
        self.model = []
        self.model += [LinearBlock(in_dim, dim, norm=norm, activation=activ)]
        for i in range(n_blk - 2):
            self.model += [LinearBlock(dim, dim, norm=norm, activation=activ)]
        self.model += [LinearBlock(dim, out_dim,
                                   norm='none', activation='none')]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))
