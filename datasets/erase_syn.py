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

def draw_border_map(polygon, canvas, mask_ori, mask):
    polygon = np.array(polygon)
    assert polygon.ndim == 2
    assert polygon.shape[1] == 2

    ### shrink box ###
    polygon_shape = Polygon(polygon)
    distance = polygon_shape.area * \
        (1 - np.power(0.95, 2)) / polygon_shape.length
    subject = [tuple(l) for l in polygon]
    padding = pyclipper.PyclipperOffset()
    padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    padded_polygon = np.array(padding.Execute(-distance)[0])
    cv2.fillPoly(mask, [padded_polygon.astype(np.int32)], 1.0)
    ### shrink box ###

    cv2.fillPoly(mask_ori, [polygon.astype(np.int32)], 1.0)

    polygon = padded_polygon
    polygon_shape = Polygon(padded_polygon)
    distance = polygon_shape.area * \
        (1 - np.power(0.4, 2)) / polygon_shape.length
    subject = [tuple(l) for l in polygon]
    padding = pyclipper.PyclipperOffset()
    padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    padded_polygon = np.array(padding.Execute(distance)[0])

    xmin = padded_polygon[:, 0].min()
    xmax = padded_polygon[:, 0].max()
    ymin = padded_polygon[:, 1].min()
    ymax = padded_polygon[:, 1].max()
    width = xmax - xmin + 1
    height = ymax - ymin + 1

    polygon[:, 0] = polygon[:, 0] - xmin
    polygon[:, 1] = polygon[:, 1] - ymin

    xs = np.broadcast_to(
        np.linspace(0, width - 1, num=width).reshape(1, width), (height, width))
    ys = np.broadcast_to(
        np.linspace(0, height - 1, num=height).reshape(height, 1), (height, width))

    distance_map = np.zeros(
        (polygon.shape[0], height, width), dtype=np.float32)
    for i in range(polygon.shape[0]):
        j = (i + 1) % polygon.shape[0]
        # import pdb;pdb.set_trace()
        absolute_distance = coumpute_distance(xs, ys, polygon[i], polygon[j])
        distance_map[i] = np.clip(absolute_distance / distance, 0, 1)
    distance_map = distance_map.min(axis=0)

    xmin_valid = min(max(0, xmin), canvas.shape[1] - 1)
    xmax_valid = min(max(0, xmax), canvas.shape[1] - 1)
    ymin_valid = min(max(0, ymin), canvas.shape[0] - 1)
    ymax_valid = min(max(0, ymax), canvas.shape[0] - 1)
    canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1] = np.fmax(
        1 - distance_map[
            ymin_valid-ymin:ymax_valid-ymax+height,
            xmin_valid-xmin:xmax_valid-xmax+width],
        canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1])

def coumpute_distance(xs, ys, point_1, point_2):
    '''
    compute the distance from point to a line
    ys: coordinates in the first axis
    xs: coordinates in the second axis
    point_1, point_2: (x, y), the end of the line
    '''
    height, width = xs.shape[:2]
    square_distance_1 = np.square(
        xs - point_1[0]) + np.square(ys - point_1[1])
    square_distance_2 = np.square(
        xs - point_2[0]) + np.square(ys - point_2[1])
    square_distance = np.square(
        point_1[0] - point_2[0]) + np.square(point_1[1] - point_2[1])

    cosin = (square_distance - square_distance_1 - square_distance_2) / \
        (2 * np.sqrt(square_distance_1 * square_distance_2))
    square_sin = 1 - np.square(cosin)
    square_sin = np.nan_to_num(square_sin)
    result = np.sqrt(square_distance_1 * square_distance_2 *
                     square_sin / square_distance)

    result[cosin < 0] = np.sqrt(np.fmin(
        square_distance_1, square_distance_2))[cosin < 0]
    # extend_line(point_1, point_2, result)
    return result

def get_seg_map(img, label):
    canvas = np.zeros(img.shape[:2], dtype = np.float32)
    mask = np.zeros(img.shape[:2], dtype = np.float32)
    mask_ori = np.zeros(img.shape[:2], dtype = np.float32)
    polygons = label

    for i in range(len(polygons)):
        draw_border_map(polygons[i], canvas, mask_ori, mask=mask)
    return canvas, mask, mask_ori


class EraseData(Dataset):
    def __init__(self, data_root, transform, phase):
        image_folder = os.path.join(data_root, 'image')
        label_folder = os.path.join(data_root, 'label')
        mask_folder = os.path.join(data_root, 'mask')

        structure_folder = os.path.join(data_root, 'structure_im')
        structure_lbl_folder = structure_folder
        # structure_lbl_folder = os.path.join(data_root, 'structure_lbl')

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

        # image = Image.open(self.image_paths[index]).convert('RGB')
        # label = Image.open(self.label_paths[index]).convert('RGB')
        mask_gt = Image.open(self.mask_paths[index]).convert('1')
        mask = Image.new('1', image.size, 1)

        mask1 = np.abs(im1.astype(np.float32) - lbl1.astype(np.float32))
        index1 = np.where(mask1 > 5)
        mask1[index1] = 255
        index1 = np.where(mask1 <= 5)
        mask1[index1] = 0
        # index1 = np.where(mask1 != 0)
        # mask1[index1] = 255
        # import pdb;pdb.set_trace()
        # mask1 = np.greater(np.mean(np.abs(np.array(im1).astype(np.float32) - np.array(lbl1).astype(np.float32)), axis=-1),15).astype(np.uint8)
        mask1 = Image.fromarray(np.uint8(mask1)).convert('1')
        # import pdb;pdb.set_trace()

        structure_im = cv2.imread(self.structure_im_paths[index])
        try:
            structure_im = Image.fromarray(cv2.cvtColor(structure_im, cv2.COLOR_BGR2RGB))
        except:
            print(self.structure_im_paths[index])
        structure_lbl = cv2.imread(self.structure_lbl_paths[index])
        structure_lbl = Image.fromarray(cv2.cvtColor(structure_lbl, cv2.COLOR_BGR2RGB))

        data = {'image': image, 'label': label, 'mask': mask, 'mask_gt': mask1, 'structure_im': structure_im, 'structure_lbl': structure_lbl, 'soft_mask': mask1}
        # structure_im = Image.open(self.structure_im_paths[index]).convert('RGB')
        # structure_lbl = Image.open(self.structure_lbl_paths[index]).convert('RGB')

        # stroke_mask = get_stroke_mask(image, label)
        # import pdb;pdb.set_trace()
        # gt_path = self.image_paths[index].replace('image', 'gt').replace('jpg','txt')
        # gt_path = self.image_paths[index].replace('image', 'detect_re').replace('jpg','txt')
        # img = np.array(image)
        # bboxes = get_anno(img, gt_path)

        # gt_instance = np.zeros(img.shape[0:2], dtype='uint8')

        # if len(bboxes) > 0:
        #     for i in range(len(bboxes)):
        #         bboxes[i] = np.reshape(bboxes[i] * ([img.shape[1], img.shape[0]] * (bboxes[i].shape[0] //2)),
        #                                 (bboxes[i].shape[0] // 2, 2)).astype('int32')
        #     for i in range(len(bboxes)):
        #         cv2.drawContours(gt_instance, [bboxes[i]], -1, i + 1, -1)

        # gt_text= gt_instance.copy()
        # gt_text[gt_text > 0] = 1

        # gt_text = Image.fromarray((1 - gt_text)*255)
        # gt_text = gt_text.convert('L')
        # # import pdb;pdb.set_trace()

        # canvas, shrink_mask, mask_ori = get_seg_map(img, bboxes)
        # soft_mask = canvas + mask_ori
        # index_mask = np.where(soft_mask > 1)
        # soft_mask[index_mask] = 1

        # # import pdb;pdb.set_trace()
        # soft_mask = Image.fromarray((1 - soft_mask) * 255)
        # soft_mask = soft_mask.convert('L')
        # # import pdb;pdb.set_trace()

        # data = {'image': image, 'label': label, 'mask': mask, 'mask_gt': gt_text, 'structure_im': structure_im, 'structure_lbl': structure_lbl, 'soft_mask': soft_mask}
        # print(self.transform)
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
    # ma = np.greater(np.mean(np.abs(np.array(im).astype(np.float32) - np.array(lbl).astype(np.float32)), axis=-1),25).astype(np.uint8)
    # ma = np.greater(np.mean(np.abs(im.astype(np.float32) - lbl.astype(np.float32)), axis=-1), 30).astype(np.uint8) # Threshold is set to 25

    # ma = np.expand_dims(ma, axis=2).astype(np.float32)
    # import pdb;pdb.set_trace()
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