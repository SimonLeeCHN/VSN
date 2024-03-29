from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
import logging
from PIL import Image


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1, mask_suffix='', force_samesize=False, target_size=(512,512)):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.force_samesize = force_samesize                # 是否需要将图片都强制转为同样尺寸的图像
        self.target_size = target_size                      # 如果需要强制转换，目标尺寸是多少
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        # listdir 返回指定文件夹下包含的文件或文件夹名字列表
        # splitext 分割路径中的文件名与拓展名
        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale, force_samesize=False, target_size=(512,512)):
        w, h = pil_img.size

        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        """
        强制将图片转为目标尺寸的图像
        """
        if force_samesize:
            pil_img = pil_img.resize(target_size)

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + self.mask_suffix + '.*')        # 使用glob去查找符合正则的文件名
        img_file = glob(self.imgs_dir + idx + '.*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'
        img = self.preprocess(img, self.scale, self.force_samesize, self.target_size)
        mask = self.preprocess(mask, self.scale, self.force_samesize, self.target_size)

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor)
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        super().__init__(imgs_dir, masks_dir, scale, mask_suffix='_mask')
