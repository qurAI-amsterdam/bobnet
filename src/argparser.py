# coding=utf-8
# Copyright: (c) 2024, Amsterdam University Medical Centers
# Apache License, Version 2.0, (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train_images_json', help='Path to json file containing paths to training images.')
parser.add_argument('--val_images_json', help='Path to json file containing paths to validation images.')
parser.add_argument('--labels_dir', help='Path to folder containing labels.')
parser.add_argument('--scan_info', help='Path to txt file containing scan information.')
parser.add_argument('--output_dir', help='Path to output directory.')
parser.add_argument('--checkpoint', default=None, help='Path to trained model checkpoint.')
parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train.')
parser.add_argument('--sample_rate', type=float, default=1.0, help='Sample rate as a fraction of the dataset size.')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='Learning rate decay rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--nesterov', default=True, choices=[True, False], help='Whether to use nesterov momentum.')
parser.add_argument('--weight_decay', type=float, default=0.0005, help='Weight decay.')
parser.add_argument(
    '--center_crop',
    action='store_true',
    help='Whether to apply center crop during validation. During training center crop is applied by default.',
)
parser.add_argument(
    '--resample_input_images',
    action='store_true',
    help='Whether to resample through-plane the input images to 3.0 mm slice thickness and 1.5 mm spacing.',
)
parser.add_argument(
    '--resize_input_images',
    action='store_true',
    help='Whether to resize in-plane the input images to (spacing[0] / voxel_size, spacing[1] / voxel_size) pixels.',
)
parser.add_argument(
    '--out_classes', type=int, default=1, help='Number of output classes. Default is 1 for binary classification.'
)
parser.add_argument('--balance_classes', action='store_true', help='Whether to balance classes.')
parser.add_argument('--device', type=str, default='cuda', help='device')
args = parser.parse_args()
