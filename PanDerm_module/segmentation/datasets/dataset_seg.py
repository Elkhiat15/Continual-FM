import os
import pandas as pd
from PIL import Image
import numpy as np
import torch
import random
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import glob
from tqdm import tqdm
from sklearn.model_selection import KFold


class SegImageDataset(Dataset):
    def __init__(self, args, split):
        if "ISIC2018" in args.dataset_path:
            self.image_paths, self.label_paths, self.names = self._get_paths_official(args, split)
        else:
            raise ValueError("Dataset not supported")
        print("=> Loading {} dataset with {} images".format(split, len(self.image_paths)))
        self.train = True if split == "train" else False
        self.im_transform, self.label_transform = build_transform(args, self.train)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label_path = self.label_paths[index] if self.label_paths is not None else None
        name = self.names[index]

        if not os.path.exists(image_path):
            print("Image not found: ", image_path)
        image = cv2.imread(image_path)[..., ::-1]
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
        image = Image.fromarray(np.uint8(image))

        # label
        if label_path is None:
            label = Image.new("1", (224, 224))
        else:
            label = cv2.imread(label_path)
            label = cv2.resize(label, (224, 224), interpolation=cv2.INTER_NEAREST)
            label = label[..., 0]
            label = Image.fromarray(np.uint8(label)).convert("1")

        seed = np.random.randint(2147483647)
        torch.manual_seed(seed)
        random.seed(seed)

        if self.im_transform is not None:
            im_t = self.im_transform(image)

        torch.manual_seed(seed)
        random.seed(seed)
        if self.label_transform is not None:
            label_t = self.label_transform(label)
            label_t = torch.squeeze(label_t).long()

        if self.train:
            return im_t, label_t
        return im_t, label_t, name

    def _get_paths_official(self, args, split):
        inputs, targets, names = [], [], []
        val_inputs, val_targets, val_names = [], [], []
        test_inputs, test_targets, test_names = [], [], []

        input_pattern = glob.glob("/data2/wangzh/datasets/ISIC2018/Training_Data/*.jpg")
        targetlist = ("/data2/wangzh/datasets/ISIC2018/Training_GroundTruth/{}_segmentation.png")

        for i in tqdm(range(len(input_pattern))):
            inputpath = input_pattern[i]
            name = analyze_name(inputpath)
            targetpath = targetlist.format(str(name))

            if os.path.exists(inputpath):
                inputs.append(inputpath)
                targets.append(targetpath)
                names.append(name)

        inputs = np.array(inputs)
        targets = np.array(targets)
        names = np.array(names)

        val_input_pattern = glob.glob("/data2/wangzh/datasets/ISIC2018/Validation_Data/*.jpg")
        val_targetlist = ("/data2/wangzh/datasets/ISIC2018/Validation_GroundTruth/{}_segmentation.png")

        val_input_pattern.sort()

        for i in tqdm(range(len(val_input_pattern))):
            inputpath = val_input_pattern[i]
            name = analyze_name(inputpath)
            targetpath = val_targetlist.format(str(name))

            if os.path.exists(inputpath):
                val_inputs.append(inputpath)
                val_targets.append(targetpath)
                val_names.append(name)

        val_inputs = np.array(val_inputs)
        val_targets = np.array(val_targets)
        val_names = np.array(val_names)

        test_input_pattern = glob.glob("/data2/wangzh/datasets/ISIC2018/Test_Data/*.jpg")
        test_targetlist = ("/data2/wangzh/datasets/ISIC2018/Test_GroundTruth/{}_segmentation.png")

        test_input_pattern.sort()

        for i in tqdm(range(len(test_input_pattern))):
            inputpath = test_input_pattern[i]
            name = analyze_name(inputpath)
            targetpath = test_targetlist.format(str(name))

            if os.path.exists(inputpath):
                test_inputs.append(inputpath)
                test_targets.append(targetpath)
                test_names.append(name)

        test_inputs = np.array(test_inputs)
        test_targets = np.array(test_targets)
        test_names = np.array(test_names)

        index = int(args.percent * len(inputs) / 100)
        inputs = inputs[:index]
        targets = targets[:index]
        names = names[:index]

        if split == "train":
            return inputs, targets, names
        elif split == "val":
            return val_inputs, val_targets, val_names
        else:
            return test_inputs, test_targets, test_names

    def _get_paths(self, args, split):
        inputs, targets, names = [], [], []

        input_pattern = glob.glob("/data2/wangzh/datasets/ISIC2018/Training_Data/*.jpg")
        targetlist = ("/data2/wangzh/datasets/ISIC2018/Training_GroundTruth/{}_segmentation.png")

        for i in tqdm(range(len(input_pattern))):
            inputpath = input_pattern[i]
            name = analyze_name(inputpath)
            targetpath = targetlist.format(str(name))

            if os.path.exists(inputpath):
                inputs.append(inputpath)
                targets.append(targetpath)
                names.append(name)

        inputs = np.array(inputs)
        targets = np.array(targets)
        names = np.array(names)

        kf = KFold(n_splits=5, shuffle=True, random_state=436)
        for ifold, (train_index, test_index) in enumerate(kf.split(inputs)):
            test_index = np.append(test_index, train_index[-1])
            if ifold != args.fold:
                continue
            X_trainset, X_test = inputs[train_index], inputs[test_index]
            y_trainset, y_test = targets[train_index], targets[test_index]
            names_trainset, names_test = names[train_index], names[test_index]

        index_s = int(len(X_trainset) * 0.875)
        X_train, X_val, y_train, y_val, names_train, names_val = (
            X_trainset[:index_s],
            X_trainset[index_s:],
            y_trainset[:index_s],
            y_trainset[index_s:],
            names_trainset[:index_s],
            names_trainset[index_s:],
        )

        if split == "train":
            return X_train, y_train, names_train
        elif split == "val":
            return X_val, y_val, names_val
        else:
            return X_test, y_test, names_test


def load_seg_dataset(args, train=False):
    if train:
        train_dataset = SegImageDataset(args, "train")
        val_dataset = SegImageDataset(args, "val")
        return train_dataset, val_dataset
    else:
        test_dataset = SegImageDataset(args, "test")
        return test_dataset


def analyze_name(path):
    name = os.path.split(path)[1]
    name = os.path.splitext(name)[0]
    return name


def build_transform(args, train=False):

    size = args.size
    if train:
        transform = transforms.Compose(
            [
                transforms.Resize(size),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomAutocontrast(p=0.2),
                transforms.RandomInvert(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        label_transform = transforms.Compose(
            [
                transforms.Resize(size, interpolation=Image.NEAREST),
                transforms.ToTensor(),
            ]
        )
        return transform, label_transform

    test_transform = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    test_label_transform = transforms.Compose(
        [
            transforms.Resize(size, interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ]
    )
    return test_transform, test_label_transform
