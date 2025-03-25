# coding=utf-8
# Copyright: (c) 2024, Amsterdam University Medical Centers
# Apache License, Version 2.0, (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0)

import warnings
from pathlib import Path
from typing import Union

import SimpleITK as sitk
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import zoom

from .utils import WeightedAverageImageResampler


class TrainSliceExtractor(Dataset):
    """Extract slices from 3D volumes and return them as 2D images with metadata."""

    def __init__(
        self, 
        fnames, 
        labels_dir, 
        scan_info_path, 
        resample_input_volumes, 
        resize_input_images, 
        balance_classes=False,
        standard_resolution=[1.5, 0.66, 0.66]
    ):
        super().__init__()
        self.labels_dir = labels_dir
        self.scan_info_path = scan_info_path
        self.resample_input_volumes = resample_input_volumes
        self.resize_input_images = resize_input_images

        examples = {}

        # rows are slice indices, cols are: image_idx, implane (ax, co, sa), sliceidx
        slice_indices = np.empty((0, 3), np.int32)

        for idx, fname in enumerate(fnames):
            img, labels_bboxes, spacing, voxel_size, filename = self.read_image_and_metadata(fname)
            examples[idx] = img, labels_bboxes, spacing, voxel_size, filename

            # Create slice indices
            arr = np.empty((0, 3), np.int32)
            for ax_idx, s in enumerate(img.shape):
                # ax_idx is the in-plane index (axial, coronal, sagittal)
                arr = np.concatenate(
                    (
                        arr,
                        np.hstack(
                            (
                                np.ones((s, 1), dtype=np.int32) * idx,
                                np.ones((s, 1), dtype=np.int32) * ax_idx,
                                np.arange(s, dtype=np.int32)[:, np.newaxis],
                            )
                        ),
                    )
                )

            slice_indices = np.concatenate((slice_indices, arr))

        self.examples = examples
        self.slice_indices = slice_indices
        self.balance_classes = balance_classes
        self.standard_resolution = standard_resolution
        self.rs = np.random.RandomState(42)

    def resampleThroughPlane(self, image, spacing, origin, original_slice_thickness):
        """
        Resample the image through-plane to 3.0 mm slice thickness and 1.5 mm spacing.

        :param image: The image to resample.
        :type image: np.ndarray
        :param spacing: The spacing of the image.
        :type spacing: tuple
        :param origin: The origin of the image.
        :type origin: tuple
        :param original_slice_thickness: The original slice thickness of the image.
        :type original_slice_thickness: float
        :return: The resampled image, spacing, and origin.
        :rtype: tuple
        """
        # All images should be in feet-to-head orientation, but originally the images were
        # resampled before correcting the orientation of images that were originally head-to-feet.
        # For compatibility reasons, we therefore flip those images back.
        resampler = WeightedAverageImageResampler(3.0, 1.5)
        imageRes, spacingRes, originRes = resampler.resample(image, spacing, origin, original_slice_thickness)
        # Normalize image values since some vendors use values < -1024 outside the FOV of the scan
        imageRes = np.clip(imageRes, -1000, 3096)
        spacingNew = np.array([1.5, spacing[1], spacing[2]])
        return imageRes, spacingNew, originRes

    def resampleInPlane(self, image, spacing, resolution=None, maskBool=False):
        """
        Resample the image in-plane to the specified resolution.

        :param image: The image to resample.
        :type image: np.ndarray
        :param spacing: The spacing of the image.
        :type spacing: tuple
        :param resolution: The resolution to resample to.
        :type resolution: tuple
        :param maskBool: Whether to resample a mask.
        :type maskBool: bool
        :return: The resampled image and resolution.
        :rtype: tuple
        """
        if resolution is None:
            resolution = self.standard_resolution
        resampling_factors = tuple(o / n for o, n in zip(spacing, resolution))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            imageRes = zoom(image, resampling_factors, order=3 if not maskBool else 0, mode='nearest')
        return imageRes, resolution

    def read_image_and_metadata(self, fname: Union[str, Path]):
        """
        Read image and metadata from file

        :param fname: path to the image file
        :type fname: Union[str, Path]
        :return: image, bounding boxes, spacing, voxel size, filename
        :rtype: tuple
        """
        img = sitk.ReadImage(fname)
        default_origin = tuple(reversed(list(img.GetOrigin())))
        default_spacing = tuple(reversed(list(img.GetSpacing())))

        # Open the scan info file and find the voxel size for the current scan
        with open(self.scan_info_path, 'r') as f:
            for line in f.readlines():
                scan = line.split(',')[0]
                if scan == Path(fname).stem:
                    voxel_size = float(line.split(',')[-1])

        img = sitk.GetArrayFromImage(img)

        # Get label bounding boxes
        labels_bboxes = sitk.GetArrayFromImage(sitk.ReadImage(Path(self.labels_dir) / Path(fname).name))

        if self.resample_input_volumes:
            # Thick-slice through-plane resampling to 3.0 mm slice thickness and 1.5 mm spacing
            ts_image, ts_spacing, ts_origin = self.resampleThroughPlane(
                img, default_spacing, default_origin, voxel_size
            )
            img, spacing = self.resampleInPlane(ts_image, ts_spacing)

            # Thick-slice through-plane resampling to 3.0 mm slice thickness and 1.5 mm spacing for labels
            ts_labels, ts_labels_spacing, _ = self.resampleThroughPlane(
                labels_bboxes, default_spacing, default_origin, voxel_size
            )
            labels_bboxes, _ = self.resampleInPlane(ts_labels, ts_labels_spacing)
        else:
            print("Resampling of input images has been set to False. Please make sure your scans are properly"
                  "preprocessed. You can also enable automatic resampling, by using the --resample_input_volumes"
                  "flag.")
            spacing = default_spacing
            assert spacing == self.standard_resolution

        # Normalize image values since some vendors use values < -1024 outside the FOV of the scan
        img = np.clip(img, -1000, 3096)

        return img, labels_bboxes, spacing, voxel_size, Path(fname).name

    @staticmethod
    def randomRotateCrop(im, width, height):
        """
        Randomly rotate and crop an image

        :param im: Image to rotate and crop
        :type im: np.ndarray
        :param width: image width
        :type width: int
        :param height: image height
        :type height: int
        :return: rotated and cropped image
        :rtype: np.ndarray
        """
        rs = np.random.rand()
        diffy = height - im.shape[0]
        diffx = width - im.shape[1]
        transly = int(diffy * rs)
        translx = int(diffx * rs)
        rows, cols = im.shape
        rot = (rs * 20) - 10
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), rot, 1)
        M[:, 2] = (translx, transly)
        dst = cv2.warpAffine(im, M, (width, height), borderValue=-1000)
        return dst

    @staticmethod
    def centerCrop(im, width, height):
        """
        Center crop an image

        :param im: Image to center crop
        :type im: np.ndarray
        :param width: image width
        :type width: int
        :param height: image height
        :type height: int
        :return: center cropped image
        :rtype: np.ndarray
        """
        d_w = int(width - im.shape[1])
        d_h = int(height - im.shape[0])
        pad_w = (0, 0)
        pad_h = (0, 0)
        if d_w > 0:
            pad_w = (int(d_w / 2), int(d_w / 2 + d_w % 2))
            d_w = 0
        if d_h > 0:
            pad_h = (int(d_h / 2), int(d_h / 2 + d_h % 2))
            d_h = 0
        padded = np.pad(im, (pad_h, pad_w), mode='constant', constant_values=-1000)
        row_start = int(abs(d_h) / 2)
        col_start = int(abs(d_w) / 2)
        new_img = padded[row_start : row_start + height, col_start : col_start + width]
        return new_img

    def __len__(self):
        """Return the number of slices in the dataset."""
        return len(self.slice_indices)

    def __getitem__(self, idx):
        """Get a slice from the dataset."""
        im_idx, ax_idx, sl_idx = self.slice_indices[idx]
        sliced = [None, None, None]
        sliced[ax_idx] = sl_idx
        img, labels_bboxes, spacing, voxel_size, fname = self.examples[im_idx]

        # find which plane to index
        if sliced[0] is not None:
            plane = 'axial'
            img = img[sliced[0]]
            bbox = labels_bboxes[sliced[0]]
            spacing = spacing[1:]
        elif sliced[1] is not None:
            plane = 'coronal'
            img = img[:, sliced[1]]
            bbox = labels_bboxes[:, sliced[1]]
            spacing = (spacing[0], spacing[2])
        elif sliced[2] is not None:
            plane = 'sagittal'
            img = img[:, :, sliced[2]]
            bbox = labels_bboxes[:, :, sliced[2]]
            spacing = spacing[:2]

        return img, bbox, spacing, voxel_size, fname, plane, sl_idx

    def batch(self, batch_size, shuffle=True, center_crop=False, imwidths=227, imheights=227):
        """
        Generate mini-batches of images

        :param batch_size: The size of the mini-batches.
        :type batch_size: int
        :param shuffle: Whether to shuffle the dataset.
        :type shuffle: bool
        :param center_crop: Whether to center crop the images.
        :type center_crop: bool
        :param imwidths: Image width(s) to use.
        :type imwidths: int or tuple
        :param imheights: Image height(s) to use.
        :type imheights: int or tuple
        :return: A mini-batch of images.
        :rtype: tuple
        """
        if not self.balance_classes:
            indices = np.arange(self.__len__(), dtype=np.int32)
        else:
            # Find the slice indices for each class
            class_indices = {}
            for idx in range(self.__len__()):
                _, bbox, _, _, _, _, _ = self.__getitem__(idx)
                if np.max(bbox) == 0:
                    class_indices.setdefault(0, []).append(idx)
                else:
                    class_indices.setdefault(1, []).append(idx)
            # Find the class with the fewest examples
            min_class = min(len(class_indices[0]), len(class_indices[1]))
            indices = np.concatenate(
                (
                    self.rs.choice(class_indices[0], min_class, replace=False),
                    self.rs.choice(class_indices[1], min_class, replace=False),
                )
            )

        if shuffle:
            self.rs.shuffle(indices)

        # Generate mini-batches
        for start_idx in range(0, indices.shape[0] - batch_size + 1, batch_size):
            # Get the indices for the current mini-batch
            excerpt = np.take(indices, np.arange(start_idx, start_idx + batch_size), mode='wrap')

            # Get the image width and height
            imwidth = self.rs.choice(imwidths, 1)[0] if isinstance(imwidths, tuple) else imwidths
            imheight = self.rs.choice(imheights, 1)[0] if isinstance(imheights, tuple) else imheights

            images = torch.empty((batch_size, imheight, imwidth))

            original_imgs = []
            targets = []
            slices = []

            # Get the images, targets, and slices for the current mini-batch
            for batch_idx, batch_jdx in enumerate(excerpt):
                img, bbox, spacing, voxel_size, fname, plane, slice_idx = self.__getitem__(batch_jdx)

                original_imgs.append(img.copy())

                if self.resize_input_images:
                    # Resize in-plane the input images to (spacing[0] / voxel_size, spacing[1] / voxel_size) pixels
                    fy, fx = spacing[0] / voxel_size, spacing[1] / voxel_size
                    img = cv2.resize(img, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
                    bbox = cv2.resize(bbox, None, fx=fx, fy=fy, interpolation=cv2.INTER_NEAREST)

                # Randomly rotate and crop the image
                img = (
                    self.centerCrop(img, imwidth, imheight)
                    if center_crop
                    else self.randomRotateCrop(img, imwidth, imheight)
                )

                img = torch.from_numpy(img)
                images[batch_idx] = img

                targets.append(torch.tensor([np.max(bbox)], dtype=torch.float32))
                slices.append(slice_idx)

            targets = torch.stack(targets, dim=0)

            yield images, original_imgs, targets, fname, plane, slices
