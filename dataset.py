import os.path
from glob import glob

import imageio
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from nibabel.viewers import OrthoSlicer3D
from torch.utils.data import DataLoader, Dataset

import transforms


class CT3DDataset(Dataset):
    """
    :param images: 3D CT图像的路径
    """

    # noinspection PyShadowingNames
    def __init__(self, images, targets=None, slices_path=None, transforms=None, mode='train', dataset_selected='B'):
        self.dataset_selected = dataset_selected
        self.image_slices = []
        self.target_slices = []
        if slices_path is not None:
            root_image_slice_path = slices_path + '/image/*'
            root_target_slice_path = slices_path + '/mask/*'
            # self.image_slice_path = glob(root_image_slice_path)
            # self.target_slice_path = glob(root_target_slice_path)
            self.image_slices = glob(root_image_slice_path)
            self.target_slices = glob(root_target_slice_path)
        self.transforms = transforms
        if targets is not None:
            if len(self.image_slices) == 0 or len(self.target_slices) == 0:
                for image_path, target_path in zip(images, targets):

                    image_splits = image_path.rsplit('/', 1)
                    mask_splits = target_path.rsplit('/', 1)
                    if mode == 'train':
                        image_save_path = image_splits[0].rsplit('/', 1)[0] + '/slices/train/image/'
                        target_save_path = mask_splits[0].rsplit('/', 1)[0] + '/slices/train/mask/'
                    elif mode == 'val':
                        image_save_path = image_splits[0].rsplit('/', 1)[0] + '/slices/val/image/'
                        target_save_path = mask_splits[0].rsplit('/', 1)[0] + '/slices/val/mask/'
                    else:
                        image_save_path = image_splits[0].rsplit('/', 1)[0] + '/slices/test/image/'
                        target_save_path = mask_splits[0].rsplit('/', 1)[0] + '/slices/test/mask/'

                    if not os.path.exists(image_save_path):
                        os.makedirs(image_save_path)
                    image_save_path += image_splits[-1].split('.')[0]

                    if not os.path.exists(target_save_path):
                        os.makedirs(target_save_path)
                    target_save_path += mask_splits[-1].split('.')[0]

                    image_data = nib.load(image_path)
                    target_data = nib.load(target_path)
                    image = image_data.get_fdata()
                    target = target_data.get_fdata()

                    # 只对训练集进行去除不包含肺部切片的处理
                    if mode:
                        image, target = remove_no_lung_slice(image, target)

                    # 对训练集和验证集同时进行去除不包含肺部切片的处理
                    # image, target = remove_no_lung_slice(image, target)

                    # 对训练集和验证集合都不进行去除不包含肺部切片的处理

                    image = image.astype('float32')
                    target = target.astype('float32')

                    slices = image.shape[-1]
                    for i in range(slices):
                        image_name = image_save_path + '_' + str(i) + '.png'
                        mask_name = target_save_path + '_' + str(i) + '.png'
                        image_slice = np.expand_dims(image[:, :, i], -1)
                        mask_slice = np.expand_dims(target[:, :, i], -1)
                        # self.image_slices.append(image_slice)
                        # self.target_slices.append(mask_slice)
                        imageio.imwrite(image_name, image_slice)
                        imageio.imwrite(mask_name, mask_slice)
                        # plt.imsave(image_name, image_slice)
                        # plt.imsave(mask_name, mask_slice)
                        self.image_slices.append(image_name)
                        self.target_slices.append(mask_name)

        else:
            if len(self.image_slices) == 0:
                for image_path in images:

                    image_splits = image_path.rsplit('/', 1)
                    image_save_path = image_splits[0].rsplit('/', 1)[0] + '/slices/image/'
                    if os.path.exists(image_save_path):
                        os.makedirs(image_save_path)
                    image_save_path += image_splits[-1].split('.')[0]

                    image_data = nib.load(image_path)
                    image = image_data.get_fdata()
                    slices = image.shape[-1]
                    for i in range(slices):
                        image_name = image_save_path + '_' + str(i) + '.png'
                        image_slice = np.expand_dims(image[:, :, i], -1)
                        # self.image_slices.append(np.expand_dims(image[:, :, i], -1))
                        imageio.imwrite(image_name, image_slice)
                        # plt.imsave(image_name, image_slice)
                        self.image_slices.append(image_name)

    def __getitem__(self, item):
        if len(self.target_slices) != 0:
            assert len(self.image_slices) == len(self.target_slices)
        image_slice = plt.imread(self.image_slices[item])
        target_slice = None
        if self.target_slices is not None:
            target_name = get_related_mask_slice(self.image_slices[item], self.target_slices, self.dataset_selected)
            target_slice = plt.imread(target_name)
        if self.transforms is not None:
            image_slice, target_slice = self.transforms(image_slice, target_slice)
        return image_slice, target_slice

    def __len__(self):
        return len(self.image_slices)


def split_train_val(data_path, target_path=None, val_size=0.2):
    """从训练集中划分出验证集

    :param data_path: 整个训练集的CT图像路径集合
    :param target_path: 整个训练集的CT图像真实分割图的路径集合
    :param val_size: 验证集的比例，默认为0.2
    :return: 划分后的训练集和验证集的图像路径集合/图像路径及对应的图像真实分割图路径集合
    """
    data_length = len(data_path)
    val_length = int(data_length * val_size)
    train_length = data_length - val_length

    np.random.seed(42)
    np.random.shuffle(data_path)

    train_path = data_path[: train_length]
    val_path = data_path[train_length:]
    if target_path is not None:
        assert len(data_path) == len(target_path)
        np.random.seed(42)
        np.random.shuffle(target_path)
        train_mask_path = target_path[: train_length]
        val_mask_path = target_path[train_length:]
        return train_path, train_mask_path, val_path, val_mask_path
    return train_path, val_path


# noinspection PyShadowingNames
def get_relate_target(image_paths, target_paths, dataset='B'):
    """可能由于系统的缘故，文件夹下面的文件排列顺序是不一致的，因此需要将图像和其对应的标签图按顺序进行对齐

    """
    reordered_target_paths = []
    if dataset == 'B':
        for path in image_paths:
            split_str = path.split('_ct')
            name_prefix = split_str[0]
            name_suffix = split_str[-1]
            target_path = name_prefix + '_seg' + name_suffix
            if target_path not in target_paths:
                raise RuntimeError('target path is not exist!')
            reordered_target_paths.append(target_path)
    elif dataset == 'A':
        for path in image_paths:
            split_str = path.split('.nii')
            name_prefix = split_str[0]
            name_suffix = split_str[-1]
            target_path = name_prefix + '.nii' + name_suffix
            if target_path not in target_paths:
                raise RuntimeError('target path is not exist!')
            reordered_target_paths.append(target_path)
    return reordered_target_paths


# noinspection PyShadowingNames
def get_related_mask_slice(image_slice_name, target_slices, dataset_selected):
    target_path = None
    if dataset_selected == 'B':
        split_str = image_slice_name.split('_ct')
        split_str_1 = split_str[0].rsplit('/', 2)
        name_prefix = split_str_1[0] + '/mask/' + split_str_1[-1]
        name_suffix = split_str[-1]
        target_path = name_prefix + '_seg' + name_suffix
        if target_path not in target_slices:
            raise RuntimeError('target path is not exist!')
    elif dataset_selected == 'A':
        split_str = image_slice_name.split('.nii')
        name_prefix = split_str[0]
        name_suffix = split_str[-1]
        target_path = name_prefix + '.nii' + name_suffix
        if target_path not in target_slices:
            raise RuntimeError('target path is not exist!')
    return target_path


def volume_resample(image_volume, target_volume=None):
    """对三维体素进行重采样到相同大小，因为不同的数据可能它的体素值不一样，这样不利于模型训练

    :param image_volume: 图像
    :param target_volume: 图像对应的标注
    :return: 重采样后的图像 or 重采样后的图像以及对应的标注
    """
    # todo 体素重采样还是有必要做一下的，以防万一，先用没有重采样的数据训练，然后使用重采样后的数据训练，看一下是否有变化
    pass


def remove_no_lung_slice(image_volume, target_volume):
    """移除CT图像中沿着深度方向没有肺部的slice

    :param image_volume: CT图像, numpy ndarray
    :param target_volume: CT图像对应的标注, numpy ndarray
    :return:
    """
    assert image_volume.shape == target_volume.shape
    depth = image_volume.shape[-1]
    head_index = _search_index(target_volume, 0, depth // 2)
    foot_index = _search_index(target_volume, depth // 2, depth, reverse=True)
    target_volume = target_volume[:, :, head_index:foot_index]
    image_volume = image_volume[:, :, head_index:foot_index]
    return image_volume, target_volume


def _search_index(target_volume, left, right, reverse=False):
    """

    :param target_volume:
    :param left:
    :param right:
    :param reverse: 如果reverse为False，表示寻找从头部向脚部方向的索引，否则表示寻找从脚部向头部方向的索引
    :return:
    """
    mid = 0
    while left < right:
        mid = (left + right) // 2
        if mid == left or mid == right:
            break
        if np.max(target_volume[:, :, :mid]) == 0:
            left = mid
            if reverse:
                right = mid
        else:
            right = mid
            if reverse:
                left = mid
    return mid

    # if np.max(target_volume[:, :, mid]) == 0.:  # 全部像素相同，即不包含肺部
    #     left = mid
    #     if reverse:
    #         right = mid
    # else:
    #     right = mid
    #     if reverse:
    #         left = mid
    # if left >= right:
    #     return mid
    # else:
    #     _search_index(target_volume, left, right)


def load_dataset(dataset_select='B', batch_size=1, train=True, train_transforms=None, test_transforms=None):
    image_paths = []
    target_paths = []
    if train:
        if dataset_select.find('A') != -1:
            image_paths.extend(glob('/home/ivan/Xiong/COVID-19-CT-Seg_20cases/COVID-19-CT-Seg_20cases/*'))
            target_paths.extend(glob('/home/ivan/Xiong/COVID-19-CT-Seg_20cases/Infection_Mask/*'))
            target_paths = get_relate_target(image_paths, target_paths, dataset='A')
        if dataset_select.find('B') != -1:
            data_path = glob('/home/ivan/Xiong/COVID-19-20/COVID-19-20_v2/Train/*')
            image_paths.extend([path for path in data_path if path.find('ct') != -1])
            target_paths.extend([path for path in data_path if path.find('seg') != -1])
            target_paths = get_relate_target(image_paths, target_paths, dataset='B')
        # 从训练集中分割出验证集
        train_paths, train_mask_paths, val_paths, val_mask_paths = split_train_val(image_paths, target_paths)
        train_dataset = CT3DDataset(train_paths, train_mask_paths,
                                    slices_path='/home/ivan/Xiong/COVID-19-20/COVID-19-20_v2/slices/train',
                                    transforms=train_transforms, mode='train', dataset_selected=dataset_select)
        val_dataset = CT3DDataset(val_paths, val_mask_paths,
                                  slices_path='/home/ivan/Xiong/COVID-19-20/COVID-19-20_v2/slices/val',
                                  transforms=test_transforms, mode='val', dataset_selected=dataset_select)
        # 获取训练集和验证集的DataLoader
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, drop_last=True)
        val_loader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, drop_last=False)
        return train_loader, val_loader
    else:
        pass


# noinspection PyShadowingNames
def show_ct(path):
    image_path = '/home/ivan/Xiong/COVID-19-20/COVID-19-20_v2/Train/volume-covid19-A-0011_ct.nii.gz'
    target_path = '/home/ivan/Xiong/COVID-19-20/COVID-19-20_v2/Train/volume-covid19-A-0011_seg.nii.gz'
    image_data = nib.load(image_path)
    target_data = nib.load(target_path)
    image = image_data.get_fdata()
    target = target_data.get_fdata()
    print(target)
    # OrthoSlicer3D(image).show()
    OrthoSlicer3D(target).show()
    depth = image.shape[-1]
    figure, axes = plt.subplots(6, 6)
    for i in range(depth):
        if i >= 36:
            break
        axes[i // 6][i - (i // 6) * 6].imshow(target[:, :, i], cmap='gray')
    plt.show()


if __name__ == '__main__':
    image_path = 'D:/dataset/COVID-19-20/COVID-19-20_v2/Train/volume-covid19-A-0003_ct.nii.gz'
    target_path = 'D:/dataset/COVID-19-20/COVID-19-20_v2/Train/volume-covid19-A-0003_seg.nii.gz'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=(0, 180)),
        transforms.RandomCrop(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    val_trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    # net = CGSegNet()
    # optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.5, 0.99))
    # criterion = nn.BCELoss()
    train_loader, val_loader = load_dataset('B', batch_size=2, train_transforms=train_trans, test_transforms=val_trans)
