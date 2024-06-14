# coding=utf-8
# Copyright: (c) 2024, Amsterdam University Medical Centers
# Apache License, Version 2.0, (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import glob
import os
import warnings
from pathlib import Path

import SimpleITK as sitk
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import zoom
from torch import nn
from tqdm import tqdm

warnings.filterwarnings('ignore', category=FutureWarning)


class PyramidPooling2d(nn.Module):
    """
    Pyramid pooling module for 2D inputs.

    Parameters
    ----------
    pool_dims : list
        The dimensions of the pooling layers.
    """

    def __init__(self, pool_dims):
        super().__init__()
        self.pooling_pyramid = [nn.AdaptiveMaxPool2d((pdim, pdim)) for pdim in pool_dims]

    def forward(self, x):
        """Forward pass through the network."""
        return torch.cat(tuple(pool(x).view(x.shape[0], -1) for pool in self.pooling_pyramid), 1)


def bnblock(inchans, outchans, **kwargs):
    """
    Create a batch normalization block.

    Parameters
    ----------
    inchans : int
        The number of input channels.
    outchans : int
        The number of output channels.

    Returns
    -------
    nn.Sequential
        The batch normalization block.
    """
    return nn.Sequential(nn.Conv2d(inchans, outchans, bias=False, **kwargs), nn.BatchNorm2d(outchans))


class BoBNet(nn.Module):
    """The BoBNet model for bounding box prediction as described in the original paper."""

    def __init__(self):
        super(BoBNet, self).__init__()

        self.conv = bnblock

        # build ENCODER
        self.encodeLayers = nn.ModuleList()
        self.encodeLayers.append(self.conv(1, 16, kernel_size=3, padding=1))
        self.encodeLayers.append(nn.ReLU())
        self.encodeLayers.append(nn.MaxPool2d(2, 2))
        self.encodeLayers.append(self.conv(16, 32, kernel_size=3, padding=1))
        self.encodeLayers.append(nn.ReLU())
        self.encodeLayers.append(nn.MaxPool2d(2, 2))
        self.encodeLayers.append(self.conv(32, 64, kernel_size=3, padding=1))
        self.encodeLayers.append(nn.ReLU())
        self.encodeLayers.append(self.conv(64, 64, kernel_size=3, padding=1))
        self.encodeLayers.append(nn.ReLU())
        self.encodeLayers.append(nn.MaxPool2d(2, 2))
        self.encodeLayers.append(self.conv(64, 128, kernel_size=3, padding=1))
        self.encodeLayers.append(nn.ReLU())
        self.encodeLayers.append(self.conv(128, 128, kernel_size=3, padding=1))
        self.encodeLayers.append(nn.ReLU())
        self.encodeLayers.append(nn.MaxPool2d(2, 2))
        self.encodeLayers.append(self.conv(128, 128, kernel_size=3, padding=1))
        self.encodeLayers.append(nn.ReLU())
        self.encodeLayers.append(self.conv(128, 128, kernel_size=3, padding=1))
        self.encodeLayers.append(nn.ReLU())

        # build POOLING
        pool_dims = [4, 2, 1]
        self.pool = PyramidPooling2d(pool_dims)

        # build CLASSIFIER
        self.classLayers = nn.ModuleList()
        nodes = 128 * np.sum(np.square([4, 2, 1]))
        self.classLayers.append(nn.Linear(nodes, 128, bias=False))
        self.classLayers.append(nn.BatchNorm1d(128))
        self.classLayers.append(nn.ReLU())
        self.classLayers.append(nn.Dropout(0.5))
        self.classLayers.append(nn.Linear(128, 128, bias=False))
        self.classLayers.append(nn.BatchNorm1d(128))
        self.classLayers.append(nn.ReLU())
        self.classLayers.append(nn.Dropout(0.5))
        self.classLayers.append(nn.Linear(128, 1))

    def forward(self, x):
        """Forward pass through the network."""
        encoding = x
        for layer in self.encodeLayers:
            encoding = layer(encoding)
        pooling = self.pool(encoding)
        classification = pooling
        for layer in self.classLayers:
            classification = layer(classification)
        return classification


def resize(imslices, spacing, voxel_size):
    """
    Resize the image to the target voxel size.

    Parameters
    ----------
    imslices : np.ndarray
        The input image.
    spacing : np.ndarray
        The spacing of the input image.
    voxel_size : float
        The target voxel size.

    Returns
    -------
    np.ndarray
        The resized image.
    """
    fy, fx = np.array(spacing, float) / voxel_size
    im = cv2.resize(imslices.transpose(1, 2, 0), None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
    if im.ndim < 3:
        im = np.expand_dims(im, axis=2)
    return im.transpose(2, 0, 1)


def resample_image(image, spacing, new_spacing, order=3):
    """
    Resample an image to a new spacing.

    Parameters
    ----------
    image : np.ndarray
        The input image.
    spacing : tuple
        The spacing of the input image.
    new_spacing : tuple
        The target spacing.
    order : int, optional
        The order of the interpolation. Default is 3.

    Returns
    -------
    np.ndarray
        The resampled image.
    """
    resampling_factors = tuple(o / n for o, n in zip(spacing, new_spacing))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return zoom(image, resampling_factors, order=order, mode='nearest')


def pad_or_crop_image(image, target_shape, fill=-1000):
    """
    Pad or crop an image to the target shape.

    Parameters
    ----------
    image : np.ndarray
        The input image.
    target_shape : tuple
        The target shape.
    fill : int, optional
        The value to fill the padding with. Default is -1000.

    Returns
    -------
    np.ndarray
        The padded or cropped image.
    """
    # Calculate the padding or cropping needed
    pads = [(max(0, t - s), max(0, t - s)) for s, t in zip(image.shape, target_shape)]
    pads = [(p[0] // 2, p[1] - p[0] // 2) for p in pads]  # Ensure symmetric padding
    crops = [(max(0, s - t), max(0, s - t)) for s, t in zip(image.shape, target_shape)]
    crops = [(c[0] // 2, c[1] - c[0] // 2) for c in crops]  # Ensure symmetric cropping
    # Apply padding if needed
    if any(p[0] > 0 for p in pads):
        padding = (pads[2][0], pads[2][1], pads[1][0], pads[1][1], pads[0][0], pads[0][1])
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
        image = F.pad(image, padding, "constant", value=fill)
    # Apply cropping if needed
    if any(c[0] > 0 for c in crops):
        slices = tuple(slice(c[0], s - c[1]) for c, s in zip(crops, image.shape))
        image = image[slices]
    return image


class WeightedAverageImageResampler:
    """
    Resampler for resampling images through-plane using weighted average.

    Parameters
    ----------
    target_slice_thickness : float
        The target slice thickness.
    target_slice_spacing : float
        The target slice spacing.
    """

    def __init__(self, target_slice_thickness, target_slice_spacing):
        self.target_slice_thickness = target_slice_thickness
        self.target_slice_spacing = target_slice_spacing

    def resample(self, image, spacing, origin, slice_thickness):
        """
        Resample the image through-plane to the standard slice thickness.

        Parameters
        ----------
        image : np.ndarray
            The input image.
        spacing : np.ndarray
            The spacing of the input image.
        origin : np.ndarray
            The origin of the input image.
        slice_thickness : float
            The original slice thickness of the input image.

        Returns
        -------
        np.ndarray
            The resampled image through-plane.
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


class BoundingBoxer:
    """
    Bounding box predictor for the heart.

    Parameters
    ----------
    checkpoint : str
        The path to the model checkpoint.
    postprocessing : bool, optional
        Whether to apply postprocessing. Default is False.
    voxel_size : float, optional
        The voxel size of the input image. By default is set to trained model's voxel size. Default is 1.5mm.
    batch_size : int, optional
        The batch size for prediction. Default is 50.
    device : str, optional
        The device to run the model on. Default is 'cuda'.
    """

    def __init__(self, checkpoint, postprocessing=False, voxel_size=1.0, batch_size=50, device='cuda'):
        # Load the model
        model = BoBNet()
        model.load_state_dict(torch.load(checkpoint, map_location=torch.device('cpu'))['model'])
        if device == 'cuda':
            model.cuda()
        self.model = model.eval()
        # Set the parameters
        self.postprocessing = postprocessing
        self.voxel_size = voxel_size
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        # Set the standard resolution
        self.standard_resolution = [1.5, 0.66, 0.66]

    @staticmethod
    def resampleAttrThroughplane(image, spacing, origin, original_slice_thickness):
        """
        Resample the image through-plane to the standard slice thickness.

        Parameters
        ----------
        image : np.ndarray
            The input image.
        spacing : np.ndarray
            The spacing of the input image.
        origin : np.ndarray
            The origin of the input image.
        original_slice_thickness : float
            The original slice thickness of the input image.

        Returns
        -------
        np.ndarray
            The resampled image through-plane.
        """
        # All images should be in feet-to-head orientation, but originally the images were resampled before correcting
        # the orientation of images that were originally head-to-feet. For compatibility reasons, we therefore flip
        # those images back.
        resampler = WeightedAverageImageResampler(3.0, 1.5)
        imageRes, spacingRes, originRes = resampler.resample(image, spacing, origin, original_slice_thickness)
        # Normalize image values since some vendors use values < -1024 outside the FOV of the scan
        imageRes = np.clip(imageRes, -1000, 3096)  # TODO: kept from legacy but it should be -1000, 3095 (not 3096)
        spacingNew = spacing
        spacingNew = np.array(spacingNew)
        spacingNew[0] = 1.5
        return imageRes, spacingNew, originRes

    def resampleAttrInplane(self, image, spacing, resolution=None, maskBool=False):
        """
        Resample the image in-plane to the specified resolution.

        Parameters
        ----------
        image : np.ndarray
            The input image.
        spacing : np.ndarray
            The spacing of the input image.
        resolution : list, optional
            The resolution to resample the image to. Default is the standard resolution of the model.
        maskBool : bool, optional
            Whether the image is a mask. Default is False.

        Returns
        -------
        np.ndarray
            The resampled image.
        """
        if resolution is None:
            resolution = self.standard_resolution
        imageRes = resample_image(image, spacing, resolution, order=3 if not maskBool else 0)
        return imageRes, resolution

    def post_process(self, final_mask, padding, ts_image, ts_spacing):
        """
        Apply postprocessing to the predicted bounding box mask.

        Parameters
        ----------
        final_mask : np.ndarray
            The predicted bounding box mask.
        padding : list
            The padding applied to the mask.
        ts_image : np.ndarray
            The resampled image.
        ts_spacing : np.ndarray
            The spacing of the resampled image.

        Returns
        -------
        np.ndarray
            The postprocessed bounding box mask.
        """
        # Remove padding
        final_mask = np.pad(final_mask, padding)
        # Resample mask to the original resolution
        final_mask = resample_image(final_mask, spacing=self.standard_resolution, new_spacing=ts_spacing, order=0)
        # Pad or crop mask to the original shape if needed
        if final_mask.shape != np.shape(ts_image):
            final_mask = pad_or_crop_image(final_mask, target_shape=np.shape(ts_image), fill=0)
        return final_mask

    def predict(self, input_image, slice_thickness):
        """
        Predict the bounding box mask of the heart from the input image.

        Parameters
        ----------
        input_image : str
            The path to the input image.
        slice_thickness : float
            The slice thickness of the input image.

        Returns
        -------
        SimpleITK.Image
            The bounding box mask of the heart.
        """
        # Convert SimpleITK Image to numpy array
        input_image = sitk.ReadImage(input_image)
        spacing = tuple(reversed(input_image.GetSpacing()))
        origin = tuple(reversed(input_image.GetOrigin()))
        direction = tuple(reversed(input_image.GetDirection()))
        image = sitk.GetArrayFromImage(input_image)
        # Resample through-plane (adjust slice thickness)
        thickslice_image, ts_spacing, ts_origin = self.resampleAttrThroughplane(
            image, spacing, origin, slice_thickness
        )
        # Resample in-plane
        image_resampled, new_spacing = self.resampleAttrInplane(thickslice_image, ts_spacing)
        # Clip intensity values (this might not be necessary depending on your specific application)
        image_resampled = np.clip(image_resampled, -1000, 3096)
        # Get bounding box in the resampled space
        final_mask, cropping, padding = self.get_vol_bounding_box(image_resampled, new_spacing)

        # Apply postprocessing if set
        if self.postprocessing:
            # Compute the final bounding box mask
            final_mask = self.post_process(final_mask, padding, image, spacing)
            bounding_box = np.where(final_mask != 0, 1, 0)
            axial_start = np.where(np.sum(bounding_box, axis=(1, 2)) != 0)[0][0]
            axial_end = np.where(np.sum(bounding_box, axis=(1, 2)) != 0)[0][-1]
            coronal_start = np.where(np.sum(bounding_box, axis=(0, 2)) != 0)[0][0]
            coronal_end = np.where(np.sum(bounding_box, axis=(0, 2)) != 0)[0][-1]
            sagittal_start = np.where(np.sum(bounding_box, axis=(0, 1)) != 0)[0][0]
            sagittal_end = np.where(np.sum(bounding_box, axis=(0, 1)) != 0)[0][-1]
            cropping = [axial_start, axial_end, coronal_start, coronal_end, sagittal_start, sagittal_end]
            bounding_box = np.zeros_like(image)
            bounding_box[
                axial_start : axial_end + 1, coronal_start : coronal_end + 1, sagittal_start : sagittal_end + 1
            ] = 1
            # Convert numpy array back to SimpleITK Image
            final_mask = sitk.GetImageFromArray(bounding_box.astype(np.uint8))
            final_mask.SetSpacing(tuple(reversed(spacing)))
            final_mask.SetOrigin(tuple(reversed(origin)))
            final_mask.SetDirection(tuple(reversed(direction)))
        else:
            # Compute the final bounding box mask
            bounding_box = np.zeros_like(image_resampled)
            bounding_box[cropping[0] : cropping[1], cropping[2] : cropping[3], cropping[4] : cropping[5]] = 1
            # Convert numpy array back to SimpleITK Image
            final_mask = sitk.GetImageFromArray(bounding_box.astype(np.uint8))
            final_mask.SetSpacing(tuple(reversed(ts_spacing)))
            final_mask.SetOrigin(tuple(reversed(ts_origin)))
            final_mask.SetDirection(tuple(reversed(direction)))

        return final_mask

    def get_vol_bounding_box(self, image, spacing):
        """
        Get the bounding box of the heart in the volume.

        Parameters
        ----------
        image : np.ndarray
            The input image.
        spacing : np.ndarray
            The spacing of the input image.

        Returns
        -------
        tuple
            The cropped to the bounding box image, the start and end indices of the bounding box, and the padding.
        """
        cropping = list()
        padding = list()
        axial, pad = self.get_bounding_box(image, np.array(spacing)[[1, 2]])
        cropping.append(axial)
        padding.append(pad)
        coronal, pad = self.get_bounding_box(image.transpose(1, 0, 2), np.array(spacing)[[0, 2]])
        cropping.append(coronal)
        padding.append(pad)
        sagittal, pad = self.get_bounding_box(image.transpose(2, 0, 1), np.array(spacing)[[0, 1]])
        cropping.append(sagittal)
        padding.append(pad)

        cropping = [cropping[0][0], cropping[0][1], cropping[1][0], cropping[1][1], cropping[2][0], cropping[2][1]]

        return image[axial[0] : axial[1], coronal[0] : coronal[1], sagittal[0] : sagittal[1]], cropping, padding

    @torch.no_grad()
    def get_bounding_box(self, oriented_image, spacing):
        """
        Get the bounding box of the heart in the oriented image.

        Parameters
        ----------
        oriented_image : np.ndarray
            The oriented image in the axial plane.
        spacing : np.ndarray
            The spacing of the oriented image.

        Returns
        -------
        tuple
            The start and end indices and the padding on the left and right sides of the predicted bounding box.
        """
        predictions = list()
        oriented_image = ((oriented_image.astype(np.float32) + 1000.0) / 4095.0).clip(0.0, 1.0)
        for idx in range(0, oriented_image.shape[0], self.batch_size):
            resized = resize(oriented_image[idx : idx + self.batch_size, :, :], spacing, self.voxel_size)
            slices = torch.from_numpy(resized[: self.batch_size, None]).to(self.device)
            prediction = torch.sigmoid_(self.model(slices))
            predictions.append(prediction)
        binarized = (torch.cat(predictions) >= 0.5).byte()
        pad_l = torch.argmax(binarized)
        pad_r = torch.argmax(binarized.flip(0))
        start_idx = pad_l
        end_idx = len(binarized) - pad_r
        return (start_idx, end_idx), (pad_l.detach().cpu(), pad_r.detach().cpu())


def main(args):
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize the bounding box predictor
    predictor = BoundingBoxer(
        checkpoint=args.checkpoint,
        postprocessing=args.postprocessing,
        voxel_size=args.voxel_size,
        batch_size=args.batch_size,
        device=args.device,
    )

    # Get all input images
    files = glob.glob(os.path.join(args.input_dir, '*.mh*'))
    files.sort()

    # Get slice thicknesses for each scan
    slice_thicknesses = dict()
    if args.scan_info is not None:
        with open(args.scan_info, 'r') as f:
            for line in f.readlines():
                scanid = line.split(',')[0]
                if os.path.join(args.input_dir, scanid + '.mha') in files:
                    slice_thicknesses[scanid] = float(line.split(',')[-1])
    else:
        for img in files:
            slice_thicknesses[os.path.basename(img[:-4])] = args.scan_slice_thickness

    # Predict bounding box masks
    for img in tqdm(files):
        mask = predictor.predict(img, slice_thicknesses[os.path.basename(img[:-4])])
        sitk.WriteImage(mask, os.path.join(output_dir, os.path.basename(img)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict bounding box masks of the heart from abdominal CT scans.')
    parser.add_argument('input_dir', help='Path to input directory with images in mha format.')
    parser.add_argument('output_dir', help="Path to output directory to save bounding box masks.")
    parser.add_argument(
        '--checkpoint',
        default='checkpoints/best.pth',
        help="Path to the model checkpoint. Default: checkpoint/best.pth.",
    )
    parser.add_argument(
        '--scan_info',
        type=str,
        default=None,
        help="Path to file containing slice thickness information for each scan.",
    )
    parser.add_argument(
        '--scan_slice_thickness',
        type=float,
        default=1.0,
        help="This can be set instead of providing a scan_info file. It is the slice thickness of the input scans, "
        "but it is applied to all scans. If you have different slice thicknesses, you should provide a scan_info "
        "file. Default: 1mm.",
    )
    parser.add_argument(
        '--postprocessing',
        action='store_true',
        help="Apply postprocessing to the predicted bounding box masks. Postprocessing will remove the padding and "
        "resample the mask to the original resolution. Default: False.",
    )
    parser.add_argument(
        '--voxel_size',
        type=float,
        default=1.5,
        help="This is set to the trained model's voxel size. It can be adjusted to match the input image's voxel size."
        "It is recommended to use the default (trained model's) voxel size for optimal performance. "
        "Default: 1.5mm",
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help="Batch size for prediction. It should be adjusted based on the available GPU memory. Default: 50.",
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help="Device to run the model on. It can be 'cuda' or 'cpu'. Default: cuda.",
    )
    args = parser.parse_args()
    main(args)
