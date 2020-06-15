"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import copy

import torch
import torch.nn as nn

from networks import FewShotGen, GPPatchMcResDis
import pdb

def entropy_loss(output, pooling, softmax, logsoftmax):
    pooling_hf, pooling_lf = pooling
    softmax_hf, softmax_lf = softmax
    logsoftmax_hf, logsoftmax_lf = logsoftmax
    output_hf,  output_lf = output
    pool_hf = pooling_hf(output_hf)
    le_hf = - torch.mean(torch.mul(softmax_hf(pool_hf), logsoftmax_hf(pool_hf)))

    pool_lf = pooling_lf(output_lf)
    le_lf = - torch.mean(torch.mul(softmax_lf(pool_lf), logsoftmax_lf(pool_lf)))
    return le_hf + le_lf
def entropy_loss_old(output, pooling, softmax, logsoftmax):
    pooling_hf, pooling_lf = pooling
    softmax_hf, softmax_lf = softmax
    logsoftmax_hf, logsoftmax_lf = logsoftmax
    output_hf,  output_lf = output
    pool_hf = pooling_hf(output_hf)
    B,C,H,W = pool_hf.size()
    pool_hf = pool_hf.view(B*C, H*W)
    le_hf = - torch.mean(torch.mul(softmax_hf(pool_hf), logsoftmax_hf(pool_hf)))

    pool_lf = pooling_lf(output_lf)
    B,C,H,W = pool_lf.size()
    pool_lf = pool_lf.view(B*C, H*W)
    le_lf = - torch.mean(torch.mul(softmax_lf(pool_lf), logsoftmax_lf(pool_lf)))
    return le_hf + le_lf
class Avgpool(nn.Module):
    def __init__(self, kernel_size=4, stride=4):
        super(Avgpool, self).__init__()
        self.pooling = nn.AvgPool2d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        return self.pooling(x)
def recon_criterion(predict, target):
    return torch.mean(torch.abs(predict - target))


class SMIT(nn.Module):
    def __init__(self, hp):
        super(SMIT, self).__init__()
        self.gen = FewShotGen(hp['gen'])
        self.dis = GPPatchMcResDis(hp['dis'])
        self.gen_test = copy.deepcopy(self.gen)
        self.pooling_hf = Avgpool()
        self.logsoftmax_hf = nn.LogSoftmax(dim=1).cuda()
        self.softmax_hf = nn.Softmax(dim=1).cuda()
        self.pooling_lf = Avgpool(kernel_size=2, stride=2)
        self.logsoftmax_lf = nn.LogSoftmax(dim=1).cuda()
        self.softmax_lf = nn.Softmax(dim=1).cuda()
    def forward(self, co_data, cl_data, octave_alpha, hp, mode, constant_octave=0.25):
        #pdb.set_trace()
        xa = co_data[0].cuda()
        la = co_data[1].cuda()
        xb = cl_data[0].cuda()
        lb = cl_data[1].cuda()
        if mode == 'gen_update':
            c_xa = self.gen.enc_content(xa, alpha_in=octave_alpha, alpha_out=octave_alpha)
            s_xa = self.gen.enc_class_model(xa, alpha_in=octave_alpha, alpha_out=octave_alpha)
            s_xb = self.gen.enc_class_model(xb, alpha_in=octave_alpha, alpha_out=octave_alpha)
            xt = self.gen.decode(c_xa, s_xb, octave_alpha)  # translation
            xr = self.gen.decode(c_xa, s_xa, octave_alpha)  # reconstruction
            l_adv_t, gacc_t, xt_gan_feat = self.dis.calc_gen_loss(xt, lb, constant_octave)
            l_adv_r, gacc_r, xr_gan_feat = self.dis.calc_gen_loss(xr, la, constant_octave)
            _, xb_gan_feat = self.dis(xb, lb, alpha_in=constant_octave, alpha_out=constant_octave)
            _, xa_gan_feat = self.dis(xa, la, alpha_in=constant_octave, alpha_out=constant_octave)
            # entropy loss
            l_e = entropy_loss(c_xa, (self.pooling_hf, self.pooling_lf), (self.softmax_hf, self.softmax_lf), (self.logsoftmax_hf, self.logsoftmax_lf))
            c_xt = self.gen.enc_content(xt, alpha_in=octave_alpha, alpha_out=octave_alpha)
            xr_cyc = self.gen.decode(c_xt, s_xa, octave_alpha)
            l_x_rec_cyc = recon_criterion(xr_cyc, xa)
            l_c_rec = recon_criterion(xr_gan_feat.mean(3).mean(2),
                                      xa_gan_feat.mean(3).mean(2))
            l_m_rec = recon_criterion(xt_gan_feat.mean(3).mean(2),
                                      xb_gan_feat.mean(3).mean(2))
            l_x_rec = recon_criterion(xr, xa)
            ## rec loss + cycle loss
            l_x_rec = l_x_rec + 1.*l_x_rec_cyc
            l_adv = 0.5 * (l_adv_t + l_adv_r)
            acc = 0.5 * (gacc_t + gacc_r)
            l_total = (hp['gan_w'] * l_adv + hp['r_w'] * l_x_rec + hp[
                'fm_w'] * (l_c_rec + l_m_rec)) + 0.01 * l_e
            l_total.backward()
            return l_total, l_adv, l_x_rec, l_c_rec, l_m_rec, acc
        elif mode == 'dis_update':
            xb.requires_grad_()
            #In Disc I use constant octave:  constant_octave = 0.25
            l_real_pre, acc_r, resp_r = self.dis.calc_dis_real_loss(xb, lb, constant_octave)
            l_real = hp['gan_w'] * l_real_pre
            l_real.backward(retain_graph=True)
            l_reg_pre = self.dis.calc_grad2(resp_r, xb)
            l_reg = 10 * l_reg_pre
            l_reg.backward()
            with torch.no_grad():
                c_xa = self.gen.enc_content(xa, alpha_in=octave_alpha, alpha_out=octave_alpha)
                s_xb = self.gen.enc_class_model(xb, alpha_in=octave_alpha, alpha_out=octave_alpha)
                xt = self.gen.decode(c_xa, s_xb, octave_alpha)
            l_fake_p, acc_f, resp_f = self.dis.calc_dis_fake_loss(xt.detach(),
                                                                  lb, constant_octave)
            l_fake = hp['gan_w'] * l_fake_p
            l_fake.backward()
            l_total = l_fake + l_real + l_reg
            acc = 0.5 * (acc_f + acc_r)
            return l_total, l_fake_p, l_real_pre, l_reg_pre, acc
        else:
            assert 0, 'Not support operation'


    def test(self, co_data, cl_data):
        self.eval()
        self.gen.eval()
        self.gen_test.eval()
        xa = co_data[0].cuda()
        xb = cl_data[0].cuda()
        for octave_alpha_value_index in range(11):
            octave_alpha_value = octave_alpha_value_index / 10.
            alpha_in, alpha_out = octave_alpha_value, octave_alpha_value 

            c_xa_current = self.gen.enc_content(xa, alpha_in=alpha_in, alpha_out=alpha_out)
            s_xa_current = self.gen.enc_class_model(xa, alpha_in=alpha_in, alpha_out=alpha_out)
            s_xb_current = self.gen.enc_class_model(xb, alpha_in=alpha_in, alpha_out=alpha_out)
            xt_current = self.gen.decode(c_xa_current, s_xb_current, octave_alpha_value)
            xr_current = self.gen.decode(c_xa_current, s_xa_current, octave_alpha_value)
            c_xa = self.gen_test.enc_content(xa, alpha_in=alpha_in, alpha_out=alpha_out)
            s_xa = self.gen_test.enc_class_model(xa, alpha_in=alpha_in, alpha_out=alpha_out)
            s_xb = self.gen_test.enc_class_model(xb, alpha_in=alpha_in, alpha_out=alpha_out)
            xt = self.gen_test.decode(c_xa, s_xb, octave_alpha_value)
            xr = self.gen_test.decode(c_xa, s_xa, octave_alpha_value)

            if octave_alpha_value_index==0:
               xt_current_set = [xt_current]  
               xr_current_set = [xr_current]
               xt_set = [xt]  
               xr_set = [xr]
            else:
               xt_current_set.append(xt_current)  
               xr_current_set.append(xr_current)
               xt_set.append(xt)  
               xr_set.append(xr)
        self.train()
        #return xa, xr_current, xt_current, xb, xr, xt
        return xa, xr_current_set[5], xt_current_set[5], xb, xr_set[5], xt_set[0], xt_set[1], xt_set[2], xt_set[3],xt_set[4], xt_set[5],xt_set[6], xt_set[7],xt_set[8], xt_set[9], xt_set[10]

    def translate_k_shot(self, co_data, cl_data, k):
        self.eval()
        xa = co_data[0].cuda()
        xb = cl_data[0].cuda()
        c_xa_current = self.gen_test.enc_content(xa)
        if k == 1:
            c_xa_current = self.gen_test.enc_content(xa)
            s_xb_current = self.gen_test.enc_class_model(xb)
            xt_current = self.gen_test.decode(c_xa_current, s_xb_current)
        else:
            s_xb_current_before = self.gen_test.enc_class_model(xb)
            s_xb_current_after = s_xb_current_before.squeeze(-1).permute(1,
                                                                         2,
                                                                         0)
            s_xb_current_pool = torch.nn.functional.avg_pool1d(
                s_xb_current_after, k)
            s_xb_current = s_xb_current_pool.permute(2, 0, 1).unsqueeze(-1)
            xt_current = self.gen_test.decode(c_xa_current, s_xb_current)
        return xt_current

    def compute_k_style(self, style_batch, k):
        self.eval()
        style_batch = style_batch.cuda()
        s_xb_before = self.gen_test.enc_class_model(style_batch)
        s_xb_after = s_xb_before.squeeze(-1).permute(1, 2, 0)
        s_xb_pool = torch.nn.functional.avg_pool1d(s_xb_after, k)
        s_xb = s_xb_pool.permute(2, 0, 1).unsqueeze(-1)
        return s_xb

    def translate_simple(self, content_image, class_code):
        self.eval()
        xa = content_image.cuda()
        s_xb_current = class_code.cuda()
        c_xa_current = self.gen_test.enc_content(xa)
        xt_current = self.gen_test.decode(c_xa_current, s_xb_current)
        return xt_current
