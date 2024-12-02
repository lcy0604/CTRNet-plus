# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import math
from typing import Text
import torch
import torch.nn.functional as F

from torch import nn
from util.misc import NestedTensor, nested_tensor_from_tensor_list
from .backbone import build_backbone
from .transformer import build_transformer
from .network import build_discriminator, build_feature_extractor,\
                                build_pixel_decoder, build_structure, build_ffc
from .loss_for_removal import TextRemovalLoss, AdversarialLoss
from PIL import Image
import numpy as np

def visual(image):
    im =(image).transpose(1,2).transpose(2,3).detach().cpu().numpy()
    Image.fromarray(im[0].astype(np.uint8)).save('1.jpg')

class CTRNetPP(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, structure_generator, pixel_decoder, feature_extractor, ffc_inapint, pixel_embed_dim, num_queries):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super(CTRNetPP, self).__init__()
        self.transformer = transformer
        self.vgg16 = feature_extractor

        self.backbone = backbone
        self.init_size = int(math.sqrt(num_queries))
        self.pixel_embed_dim = pixel_embed_dim

        self.pixel_decoder = pixel_decoder

        self.structure = structure_generator

        self.ffc_inpaint = ffc_inapint

        # self.conv_input = nn.Conv2d(4, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, samples: NestedTensor, mask_label, gt, structure_im, structure_lbl, soft_masks):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x 1 x H x W], containing 1 on padded pixels, 0 denotes fg, 1 denotes bg 

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        # features, pos = self.backbone(samples)
        # import pdb;pdb.set_trace()
        images = samples.tensors
        # samples.tensors = self.conv_input(torch.cat([samples.tensors, (1 - mask_label)], 1))
        structure_out = self.structure(torch.cat((structure_im, 1 - soft_masks),1))
        # import pdb;pdb.set_trace()
        # structure_out = self.structure(structure_im)

        features, pos = self.backbone(samples, structure_out, 1 - soft_masks)
        src, mask = features[-1].decompose()
        # im_f = self.input_proj(src)
        assert mask is not None
        # hs, inter_hs = self.transformer(im_f)
        # import pdb;pdb.set_trace()

        # import pdb;pdb.set_trace()
        skip_connect_features = [ele.decompose()[0] for ele in features]

        ### 2nd stage with hcg ###
        masked_image = samples.tensors * (mask_label)
        masked_image = torch.cat([masked_image, 1 - mask_label], dim=1)
        if self.training:
            inpaint_out, middle_feat, decode_feat = self.ffc_inpaint(masked_image)
        else:
            inpaint_out, middle_feat, decode_feat = self.ffc_inpaint(masked_image)
        # layout, feature_recon = self.hcg_encoder(samples.tensors, mask_label)
        # import pdb;pdb.set_trace()
        hs, inter_hs = self.transformer(src, middle_feat)
        # import pdb;pdb.set_trace()
        final_output = self.pixel_decoder(hs, inter_hs, middle_feat, decode_feat, skip_connect_features)

        output_comp = mask_label * images + (1 - mask_label) * final_output[-1]  
        # output_comp = mask_label * gt + (1 - mask_label) * final_output[-1] 
        # import pdb;pdb.set_trace()
        if not self.training:
            return final_output[-1]  
        
        # real_prob = self.discriminator(gt, mask_label)
        # with torch.no_grad():
        #     pixel_output_detach = torch.empty_like(pixel_output[-1]).copy_(pixel_output[-1])
        # fake_prob_D = self.discriminator(pixel_output[-1].clone().detach(), mask_label)
        # fake_prob_G = self.discriminator(pixel_output[-1], mask_label)


        feat_output_comp = self.vgg16(output_comp)
        feat_output = self.vgg16(final_output[-1])
        feat_gt = self.vgg16(gt)

        feat_inpaint = self.vgg16(inpaint_out)

        feat_structure = self.vgg16(structure_out)
        feat_structure_lbl = self.vgg16(structure_lbl)


        preds = {
            'output': final_output,
            'output_com': output_comp,
            'feat_output_comp': feat_output_comp,
            'feat_output': feat_output,
            'feat_gt': feat_gt,
            'structure_output': structure_out,
            'inpaint_out': inpaint_out,
            'feat_inpaint_out': feat_inpaint,
            'feat_structure_out': feat_structure,
            'feat_structure_lbl': feat_structure_lbl,
        }
        return preds


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    device = torch.device(args.device)

    structure_generator = build_structure(args)

    backbone = build_backbone(args)
    transformer = build_transformer(args)
    # deconv_decoder = build_deconv_decoder(args)
    discriminator = build_discriminator(args)
    feature_extractor = build_feature_extractor(args)

    pixel_decoder = build_pixel_decoder(args)

    ffc_inapint = build_ffc(args)

    model = CTRNetPP(backbone, transformer, structure_generator, pixel_decoder, feature_extractor, ffc_inapint, args.pixel_embed_dim, args.pix2pix_queries)
        
    criterion = TextRemovalLoss().to(device)
    # criterion_adversial = AdversarialLoss('nsgan').to(device)

    return model, discriminator, criterion#, criterion_adversial