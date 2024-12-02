import torch
import torch.nn as nn 
import torch.nn.functional as F

class AdversarialLoss(nn.Module):
    """
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """

    def __init__(self, type='nsgan', target_real_label=1.0, target_fake_label=0.0):
        """
        type = nsgan | lsgan | hinge
        """
        super(AdversarialLoss, self).__init__()

        self.type = type
        self.register_buffer('real_label', torch.tensor(target_real_label).cuda())
        self.register_buffer('fake_label', torch.tensor(target_fake_label).cuda())

        # self.register_buffer('real_label', torch.tensor(target_real_label))   ### original code
        # self.register_buffer('fake_label', torch.tensor(target_fake_label))   ### original code

        if type == 'nsgan':
            self.criterion = nn.BCELoss()

        elif type == 'lsgan':
            self.criterion = nn.MSELoss()

        elif type == 'hinge':
            self.criterion = nn.ReLU()

    def __call__(self, outputs, is_real, is_disc=None):
        if self.type == 'hinge':
            if is_disc:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()

        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
            loss = self.criterion(outputs, labels)
            return loss

class TextRemovalLoss(nn.Module):
    def __init__(self):
        super(TextRemovalLoss, self).__init__()

    def forward(self, preds, mask_gt, gt, structure_lbl):

        D_fake = -torch.mean(preds['fake_prob_G'])
        msr_loss = self.MSR_loss(preds['output'], mask_gt, gt) 

        prc_loss = self.percetual_loss(preds['feat_output_comp'], preds['feat_output'], preds['feat_gt'], preds['feat_inpaint_out'], preds['feat_structure_out'], preds['feat_structure_lbl'])
        style_loss = self.style_loss(preds['feat_output_comp'], preds['feat_output'], preds['feat_gt'], preds['feat_inpaint_out'], preds['feat_structure_out'], preds['feat_structure_lbl'])

        FM_loss = 5 * F.l1_loss(preds['inpaint_out'] * (1 - mask_gt), gt * (1 - mask_gt)) + \
                        F.l1_loss(preds['inpaint_out'] * (mask_gt), gt * (mask_gt)) 
        

        structure_loss = 3 * F.l1_loss(preds['structure_output'] * (1 - mask_gt), structure_lbl * (1 - mask_gt)) + \
                        F.l1_loss(preds['structure_output'] * (mask_gt), structure_lbl * (mask_gt)) 


        losses = {'MSR_loss': msr_loss, 'prc_loss': prc_loss, 'style_loss': style_loss, 'FM_loss': FM_loss,  
                 'D_fake': D_fake, 'structure_loss': structure_loss}
        return losses

    def mask_loss(self, mask_pred, mask_label):
        return dice_loss(mask_pred, mask_label)
    
    @staticmethod
    def discriminator_loss(real_prob, fake_prob):
        return hinge_loss(real_prob, 1) + hinge_loss(fake_prob, -1)
    
    def percetual_loss(self, feat_output_comp, feat_output, feat_gt, feat_inpaint, feat_str, feat_str_lbl):
    # def percetual_loss(self, feat_output_comp, feat_output, feat_gt):
        pcr_losses = []
        for i in range(3):
            pcr_losses.append(F.l1_loss(feat_output[i], feat_gt[i]))
            pcr_losses.append(F.l1_loss(feat_output_comp[i], feat_gt[i]))
            pcr_losses.append(F.l1_loss(feat_inpaint[i], feat_gt[i]))
            pcr_losses.append(F.l1_loss(feat_str[i], feat_str_lbl[i]))
        return sum(pcr_losses)
    
    def style_loss(self, feat_output_comp, feat_output, feat_gt, feat_inpaint, feat_str, feat_str_lbl):
    # def style_loss(self, feat_output_comp, feat_output, feat_gt):
        style_losses = []
        for i in range(3):
            style_losses.append(F.l1_loss(gram_matrix(feat_output[i]), gram_matrix(feat_gt[i])))
            style_losses.append(F.l1_loss(gram_matrix(feat_output_comp[i]), gram_matrix(feat_gt[i])))
            style_losses.append(F.l1_loss(gram_matrix(feat_inpaint[i]), gram_matrix(feat_gt[i])))
            style_losses.append(F.l1_loss(gram_matrix(feat_str[i]), gram_matrix(feat_str_lbl[i])))
        return sum(style_losses)

    def MSR_loss(self, outputs, mask, gt, scale_factors=[0.25, 0.5, 1.0], weights= [[5,0.8], [6,1], [10,2]]):
        msr_losses = []
        for output, scale_factor, weight in zip(outputs, scale_factors, weights):
            if scale_factor != 1:
                mask_ = F.interpolate(mask, scale_factor=scale_factor, recompute_scale_factor=True)
                gt_ = F.interpolate(gt, scale_factor=scale_factor, recompute_scale_factor=True)
            else:
                mask_ = mask; gt_ = gt 
            msr_losses.append(weight[0] * F.l1_loss((1 - mask_) * output, (1 - mask_) * gt_))
            msr_losses.append(weight[1] * F.l1_loss(mask_ * output, mask_ * gt_))
        return sum(msr_losses)
    
    def refine_loss(self, output, mask, gt):
        refine_loss = 10 * F.l1_loss(output * (1 - mask), gt * (1 - mask)) + 2 * F.l1_loss(output * mask, gt * mask)
        return refine_loss  
            
def gram_matrix(feat):
    # https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    return gram

def hinge_loss(input, target):
    return torch.mean(F.relu(1 - target * input))

def dice_loss(input, target):
    input = torch.sigmoid(input)
    input = input.flatten(1)
    target = target.flatten(1)

    a = torch.sum(input * target, 1)
    b = torch.sum(input * input, 1) + 0.001
    c = torch.sum(target * target, 1) + 0.001
    dice_loss = (2 * a) / (b + c)
    dice_loss = torch.mean(dice_loss)
    return 1 - dice_loss