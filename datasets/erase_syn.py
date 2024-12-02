import os

from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, ConcatDataset
from torchvision.transforms import Compose
import cv2
import numpy as np
import pyclipper
import Polygon as plg
from shapely.geometry import Polygon
# import .transforms as T
from . import transforms as T

class EraseData(Dataset):
    def __init__(self, data_root, transform, phase):
        image_folder = os.path.join(data_root, 'image')
        label_folder = os.path.join(data_root, 'label')
        mask_folder = os.path.join(data_root, 'mask')

        structure_folder = os.path.join(data_root, 'structure_im')
        structure_lbl_folder = structure_folder

        image_names = os.listdir(image_folder)
        image_names.sort()
        self.image_paths = [os.path.join(image_folder, ele) for ele in image_names]
        self.label_paths = [os.path.join(label_folder, ele) for ele in image_names]
        self.mask_paths = [os.path.join(mask_folder, ele) for ele in image_names]

        self.structure_im_paths = [os.path.join(structure_folder, ele) for ele in image_names]
        self.structure_lbl_paths = [os.path.join(structure_lbl_folder, ele) for ele in image_names]
    
        self.transform = transform
        self.phase = phase

    def __getitem__(self, index):
        image = cv2.imread(self.image_paths[index])
        im1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        label = cv2.imread(self.label_paths[index])
        lbl1 = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
        label = Image.fromarray(cv2.cvtColor(label, cv2.COLOR_BGR2RGB))

        mask_gt = Image.open(self.mask_paths[index]).convert('1')
        mask = Image.new('1', image.size, 1)

        mask1 = np.abs(im1.astype(np.float32) - lbl1.astype(np.float32))
        index1 = np.where(mask1 > 5)
        mask1[index1] = 255
        index1 = np.where(mask1 <= 5)
        mask1[index1] = 0

        mask1 = Image.fromarray(np.uint8(255 - mask1)).convert('1')

        structure_im = cv2.imread(self.structure_im_paths[index])
        try:
            structure_im = Image.fromarray(cv2.cvtColor(structure_im, cv2.COLOR_BGR2RGB))
        except:
            print(self.structure_im_paths[index])
        structure_lbl = cv2.imread(self.structure_lbl_paths[index])
        structure_lbl = Image.fromarray(cv2.cvtColor(structure_lbl, cv2.COLOR_BGR2RGB))

        data = {'image': image, 'label': label, 'mask': mask, 'mask_gt': mask1, 'structure_im': structure_im, 'structure_lbl': structure_lbl, 'soft_mask': mask1}

        if not self.transform is None:
            data = self.transform(data)

        data['image_path'] = self.image_paths[index]
        return data

    def __len__(self):
        return len(self.image_paths)


def get_stroke_mask(image, label):
    im = np.array(image)
    lbl = np.array(label)
    # import pdb;pdb.set_trace()
    im = cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
    lbl = cv2.cvtColor(lbl,cv2.COLOR_RGB2GRAY)
    
    ma = np.abs(im.astype(np.float32) - lbl.astype(np.float32))
    index = np.where(ma > 25)
    ma[index] = 255
    index = np.where(ma <= 25)
    ma[index] = 0

    ma = Image.fromarray(ma)
    ma = ma.convert('RGB')
    ma.save('1.jpg')
    return ma

def get_anno(img, gt_path):
    h, w = img.shape[0:2]
    bboxes = []
    f1 = open(gt_path, 'r')
    lines = f1.readlines()
    # import pdb;pdb.set_trace()
    for line in lines[:]:
        line = line.strip().split(',')
        # import pdb;pdb.set_trace()
        bbox = []
        for i in range(len(line)):
            bbox.append(float(line[i]))
        point_num = int(len(line)/2)
        # import pdb;pdb.set_trace()
        bbox = np.asarray(bbox)/ ([w * 1.0, h * 1.0] * point_num)
        bboxes.append(bbox)
    return bboxes

def make_erase_transform(image_set, args):
    transforms = []
    if image_set == 'train':
        transforms.append(T.RandomCrop(args.crop_min_ratio, args.crop_max_ratio, args.crop_prob))
        transforms.append(T.RandomHorizontalFlip(args.horizontal_flip_prob))
        transforms.append(T.RandomRotate(args.rotate_max_angle, args.rotate_prob))
    transforms.append(T.Resize((args.pix2pix_size, args.pix2pix_size)))
    transforms.append(T.ToTensor())
    return Compose(transforms)

def build(image_set, args):
    root = Path(args.data_root)
    if image_set == 'train':
        dataset_names = args.train_dataset.split(':')
    elif image_set == 'val':
        dataset_names = args.val_dataset.split(':')
    
    datasets = []
    for dataset_name in dataset_names:
        if dataset_name == 'scutsyn_train':
            data_root = root / 'SCUT-Syn' / 'train'
        elif dataset_name == 'scutsyn_test':
            data_root = root / 'SCUT-Syn' / 'test'
        elif dataset_name == 'scutens_train':
            data_root = root / 'SCUT-ENS' / 'train'
        elif dataset_name == 'scutens_test':
            data_root = root / 'SCUT-ENS' / 'test'
        else:
            raise NotImplementedError 
        
        transforms = make_erase_transform(image_set, args)
        dataset = EraseData(data_root, transforms, image_set)
        datasets.append(dataset)
    
    if len(datasets) > 1:
        dataset = ConcatDataset(datasets)

    return dataset
