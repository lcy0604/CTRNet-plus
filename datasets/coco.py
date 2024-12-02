# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""

import torch
import torchvision
import torch.utils.data
import datasets.transforms as T

from pathlib import Path
from pycocotools import mask as coco_mask
from torch.utils.data import ConcatDataset


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, dataset_name):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks, dataset_name)

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img1, target1 = self._transforms(img, target)
            img2, target2 = self._transforms(img, target)
        return img1, img2, target1, target2


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False, dataset_name=''):
        self.return_masks = return_masks
        self.dataset_name = dataset_name

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]
        # if len(anno) == 0:
        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # import pdb;pdb.set_trace()
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        target['dataset_name'] = self.dataset_name
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        recog = [obj['rec'] for obj in anno]
        recog = torch.tensor(recog, dtype=torch.long).reshape(-1, 25)
        target["rec"]  = recog

        bezier_pts = [obj['bezier_pts'] for obj in anno]
        bezier_pts = torch.tensor(bezier_pts, dtype=torch.float32).reshape(-1, 16)
        target['bezier_pts'] = bezier_pts

        # from util.visualize import draw_bezier_points, draw_bezier_curves
        # import cv2
        # import numpy as np
        # image_ = np.asarray(image)
        # image_ = cv2.cvtColor(image_, cv2.COLOR_RGB2BGR)
        # # import pdb;pdb.set_trace()    
        # for bezier_pts_ in bezier_pts:
        #     bezier_pts_ = bezier_pts_.numpy().reshape(-1, 2)
        #     image_ = draw_bezier_points(image_, bezier_pts_)
        #     image_ = draw_bezier_curves(image_, bezier_pts_[:4, :].transpose())
        #     image_ = draw_bezier_curves(image_, bezier_pts_[4:, :].transpose())
        # import pdb; pdb.set_trace()
        return image, target


def make_coco_transforms(image_set, max_size_train, min_size_train, max_size_test, min_size_test,
                         crop_min_ratio, crop_max_ratio, crop_prob, rotate_max_angle, rotate_prob,
                         brightness, contrast, saturation, hue, distortion_prob):

    transforms = []
    if image_set == 'train':
        transforms.append(T.RandomSizeCrop(crop_min_ratio, crop_max_ratio, True, crop_prob))
        transforms.append(T.RandomRotate(rotate_max_angle, rotate_prob))
        transforms.append(T.RandomResize(min_size_train, max_size_train))
        transforms.append(T.RandomDistortion(brightness, contrast, saturation, hue, distortion_prob))
    if image_set == 'val':
        transforms.append(T.RandomResize([min_size_test], max_size_test))
    transforms.append(T.ToTensor())
    transforms.append(T.Normalize(None, None))

    return T.Compose(transforms)


def build(image_set, args):
    root = Path(args.data_root)
    if image_set == 'train':
        dataset_names = args.train_dataset.split(':')
    elif image_set == 'val':
        dataset_names = args.val_dataset.split(':')
    
    datasets = []
    for dataset_name in dataset_names:
        if dataset_name == 'totaltext_train':
            img_folder = root / "totaltext" / "train_images"; ann_file = root / "totaltext" / "train.json"
        elif dataset_name == 'totaltext_val':
            img_folder = root / "totaltext" / "test_images"; ann_file = root / "totaltext" / "test.json"
        elif dataset_name == 'mlt_train':
            img_folder = root / "mlt2017" / "MLT_train_images"; ann_file = root / "mlt2017" / "train.json"
        elif dataset_name == 'ctw1500_train':
            img_folder = root / "CTW1500" / "ctwtrain_text_image"; ann_file = root / "CTW1500" / "annotations" / "train_ctw1500_maxlen25_v2.json"
        elif dataset_name == 'ctw1500_val':
            img_folder = root / "CTW1500" / "ctwtest_text_image"; ann_file = root / "CTW1500" / "annotations" / "test_ctw1500_maxlen25.json"
        elif dataset_name == 'syntext1_train':
            img_folder = root / "syntext1" / "syntext_word_eng"; ann_file = root / "syntext1" / "train.json"
        elif dataset_name == 'syntext2_train':
            img_folder = root / "syntext2" / "emcs_imgs"; ann_file = root / "syntext2" / "train.json"
        elif dataset_name == 'cocotextv2_train':
            img_folder = root / "cocotextv2" / "train2014"; ann_file = root / "cocotextv2" / "cocotext.v2.rewrite.json"
        elif dataset_name == 'ic13_train':
            img_folder = root / "IC13" / "ch2_training_images"; ann_file = root / "IC13" / "icdar_2013_ist_v2.json"
        elif dataset_name == 'ic15_train':
            img_folder = root / "IC15" / "ch4_training_images"; ann_file = root / "IC15" / "icdar_2015_ist_v2.json"
        elif dataset_name == 'ic13_val':
            img_folder = root / "IC13" / "ic13_Test_Images"; ann_file = root / "IC13" / "icdar_2013_ist_test.json"
        elif dataset_name == 'ic15_val':
            img_folder = root / "IC15" / "ic15_Test_Images"; ann_file = root / "IC15" / "icdar_2015_ist_test.json"
        else:
            raise NotImplementedError
        
        transforms = make_coco_transforms(image_set, args.max_size_train, args.min_size_train,
              args.max_size_test, args.min_size_test, args.crop_min_ratio, args.crop_max_ratio,
              args.crop_prob, args.rotate_max_angle, args.rotate_prob, args.brightness, args.contrast,
              args.saturation, args.hue, args.distortion_prob)
        dataset = CocoDetection(img_folder, ann_file, transforms=transforms, return_masks=args.masks, dataset_name=dataset_name)
        datasets.append(dataset)
    
    if len(datasets) > 1:
        dataset = ConcatDataset(datasets)

    return dataset
