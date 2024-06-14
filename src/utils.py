# coding=utf-8
# Copyright: (c) 2024, Amsterdam University Medical Centers
# Apache License, Version 2.0, (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0)

import json
import random
from pathlib import Path

import cv2
import numpy as np


def load_train_files(train_images_json, val_images_json, scan_info, sample_rate):
    """
    Load the training and validation files.

    :param train_images_json: Path to json file containing paths to training images.
    :type train_images_json: Path
    :param val_images_json: Path to json file containing paths to validation images.
    :type val_images_json: Path
    :param scan_info: Path to txt file containing scan information.
    :type scan_info: Path
    :param sample_rate: Sample rate.
    :type sample_rate: float
    :return: Training and validation files.
    :rtype: tuple
    """
    with open(train_images_json, 'r') as f:
        examples = json.load(f)
    train_fnames = [Path(example) for example in examples]

    with open(val_images_json, 'r') as f:
        examples = json.load(f)
    val_fnames = [Path(example) for example in examples]

    if sample_rate < 1.0:  # sample by slice
        random.shuffle(train_fnames)
        random.shuffle(val_fnames)
        train_fnames = train_fnames[: int(len(train_fnames) * sample_rate)]
        val_fnames = val_fnames[: int(len(val_fnames) * sample_rate)]

    # Get image heights and widths for resizing all mini-batches to the same size
    shapes = []
    for fname in train_fnames:
        with open(fname, 'r', encoding='ISO-8859-1') as f:
            for l in f:
                if l.startswith('ElementSpacing'):
                    sz = np.array(l[17:].split()[::-1], dtype=np.float32)
                if l.startswith('DimSize'):
                    sh = np.array(l[10:].split()[::-1], dtype=np.uint)
        with open(scan_info, 'r') as f:
            for line in f.readlines():
                scan = line.split(',')[0]
                if scan == Path(fname).stem:
                    voxel_size = float(line.split(',')[-1])
        shapes.append(sh * sz / voxel_size)
    p75 = np.clip(np.percentile(shapes, 75, 0), 224, 6600)
    p25 = np.clip(np.percentile(shapes, 25, 0), 224, 6600)
    imheights = (int(round(p25[0])), int(round(p75[0])))
    imwidths = (int(round(p25[1])), int(round(p75[1])))

    return train_fnames, val_fnames, imheights, imwidths


def resize(imslices, spacing, voxel_size):
    """
    Resize the image slices to the desired voxel size.

    :param imslices: Image slices to resize.
    :type imslices: np.ndarray
    :param spacing: Spacing between the slices.
    :type spacing: tuple
    :param voxel_size: Desired voxel size.
    :type voxel_size: float
    :return: Resized image slices.
    :rtype: np.ndarray
    """
    fy, fx = spacing[0] / voxel_size, spacing[1] / voxel_size
    im = cv2.resize(imslices.transpose(1, 2, 0), None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
    if im.ndim < 3:
        im = np.expand_dims(im, axis=2)
    return im.transpose(2, 0, 1)


class WeightedAverageImageResampler:
    """
    Resample the image slices to the desired slice thickness and spacing using a weighted average.

    :param target_slice_thickness: Desired slice thickness.
    :type target_slice_thickness: float
    :param target_slice_spacing: Desired slice spacing.
    :type target_slice_spacing: float
    """

    def __init__(self, target_slice_thickness, target_slice_spacing):
        self.target_slice_thickness = target_slice_thickness
        self.target_slice_spacing = target_slice_spacing

    def resample(self, image, spacing, origin, slice_thickness):
        """
        Resample the image slices.

        :param image: Image slices to resample.
        :type image: np.ndarray
        :param spacing: Spacing between the slices.
        :type spacing: tuple
        :param origin: Origin of the image.
        :type origin: tuple
        :param slice_thickness: Slice thickness of the image.
        :type slice_thickness: float
        :return: Resampled image, spacing, and origin.
        :rtype: tuple
        """
        slice_spacing = abs(float(spacing[0]))
        slice_thickness = abs(float(slice_thickness))
        # Compute number of slices in resampled image
        scan_thickness = slice_thickness + (slice_spacing * (image.shape[0] - 1))
        target_num_slices = int(
            np.floor((scan_thickness - self.target_slice_thickness) / self.target_slice_spacing + 1)
        )

        # Compute offset of the origin in the new image
        origin_offset_z = -(0.5 * slice_thickness) + (0.5 * self.target_slice_thickness)

        # Create a new (empty) image volume
        target_shape = (target_num_slices, image.shape[1], image.shape[2])
        resampled_image = np.empty(shape=target_shape, dtype=image.dtype)

        resampled_spacing = (self.target_slice_spacing, spacing[1], spacing[2])
        resampled_origin = (origin[0] + origin_offset_z, origin[1], origin[2])

        # Fill new image with values
        for z in range(resampled_image.shape[0]):
            sum_weights = 0
            sum_values = np.zeros((resampled_image.shape[1], resampled_image.shape[2]), dtype=float)

            slice_begin = z * self.target_slice_spacing
            slice_end = slice_begin + self.target_slice_thickness

            # Find first slice in the old image that overlaps with the new slice
            old_slice = 0
            old_slice_begin = 0
            old_slice_end = slice_thickness

            while old_slice_end < slice_begin:
                old_slice += 1
                old_slice_begin += slice_spacing
                old_slice_end += slice_spacing

            # Find all slices in the old image that overlap with the new slice
            while old_slice < image.shape[0] and old_slice_begin < slice_end:
                if old_slice_end <= slice_end:
                    weight = (old_slice_end - max(slice_begin, old_slice_begin)) / slice_thickness
                    sum_weights += weight
                    sum_values += weight * image[old_slice, :, :]
                elif old_slice_begin >= slice_begin:
                    weight = (slice_end - old_slice_begin) / slice_thickness

                    sum_weights += weight
                    sum_values += weight * image[old_slice, :, :]
                elif old_slice_begin <= slice_begin and old_slice_end >= slice_end:
                    weight = (self.target_slice_thickness) / slice_thickness

                    sum_weights += weight
                    sum_values += weight * image[old_slice, :, :]

                old_slice += 1
                old_slice_begin += slice_spacing
                old_slice_end += slice_spacing
            resampled_image[z, :, :] = np.round(sum_values / sum_weights)

        return resampled_image, resampled_spacing, resampled_origin
