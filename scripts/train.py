# coding=utf-8
# Copyright: (c) 2024, Amsterdam University Medical Centers
# Apache License, Version 2.0, (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0)

import logging
import os
import warnings
from pathlib import Path

import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import F1Score, Precision, Recall
from torchmetrics.classification import BinaryAccuracy

from src.argparser import args
from src.dataloader import TrainSliceExtractor
from src.nn import BoBNet
from src.utils import load_train_files

warnings.filterwarnings('ignore', category=FutureWarning)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def run_inference(inputs, model):
    """
    Run inference on a given slice.

    :param inputs: input slice
    :type inputs: torch.Tensor
    :param model: model
    :type model: nn.Module
    :return: predictions
    :rtype: torch.Tensor
    """
    inputs = ((torch.clip(inputs, -1000, 3096) + 1000.0) / 4095.0).clip(0.0, 1.0)
    return model(inputs.unsqueeze(1))


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')

    # create output directory
    log_dir = Path(args.output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # create tensorboard summary writer
    writer = SummaryWriter(log_dir=str(log_dir))
    logfile = log_dir / 'log.txt'

    # create logger
    logger = logging.getLogger('example_logger')
    logger.addHandler(logging.FileHandler(logfile))

    # create model
    model = BoBNet(out_classes=args.out_classes).to(device)
    logger.info(model)

    # Load the model from a checkpoint
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model'])
        optimizer = checkpoint['optimizer']
        logger.info(f"Loaded checkpoint from {args.checkpoint}")
    else:
        # Define the optimizer Nesterov
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            nesterov=args.nesterov,
            weight_decay=args.weight_decay,
        )
    logger.info(optimizer)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.1)
    logger.info(scheduler)

    # Load the images
    train_fnames, val_fnames, imheights, imwidths = load_train_files(
        args.train_images_json, args.val_images_json, args.scan_info, args.sample_rate
    )

    # Create a dataloader
    train_dataset = TrainSliceExtractor(
        train_fnames,
        args.labels_dir,
        args.scan_info,
        args.resample_input_images,
        args.resize_input_images,
        args.balance_classes,
    )
    val_dataset = TrainSliceExtractor(
        val_fnames, args.labels_dir, args.scan_info, args.resample_input_images, args.resize_input_images, False
    )

    logger.info(f"Train dataset: {len(train_dataset)} images")
    logger.info(f"Validation dataset: {len(val_dataset)} images")

    # Define the loss function as binary cross entropy with logits
    criterion = nn.BCEWithLogitsLoss(reduction='mean').to(device)
    logger.info(criterion)

    # Define the metrics
    acc_metric = BinaryAccuracy().to(device)
    logger.info(acc_metric)
    f1_metric = F1Score("binary").to(device)
    logger.info(f1_metric)
    precision_metric = Precision("binary").to(device)
    logger.info(precision_metric)
    recall_metric = Recall("binary").to(device)
    logger.info(recall_metric)

    best_loss = 1e10

    global_train_loss = []
    global_val_loss = []
    for epoch in range(args.epochs):
        # reduce learning rate by 0.1 every 10 epochs if epoch > 0 and epoch % 10 == 0:
        if epoch > 0 and epoch % 20 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= args.lr_decay_rate

        train_loss = 0.0
        train_batches = 0

        # Training loop
        for inputs, original_imgs, targets, fnames, planes, slices in train_dataset.batch(
            args.batch_size, True, args.center_crop, imwidths=imwidths, imheights=imheights
        ):
            optimizer.zero_grad()
            predictions = run_inference(inputs.to(device), model)

            if args.out_classes > 1:
                # If there are more than 1 output classes, stack the background class as well
                targets = torch.hstack([torch.logical_not(targets), targets])

            # Calculate the loss
            loss = criterion(predictions, targets.to(device))

            # Add L2 regularization to the loss
            l2_reg = 0
            for param in model.parameters():
                l2_reg += torch.norm(param) ** 2
            loss += 0.0005 * l2_reg

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_batches += 1

            global_train_loss.append(loss.detach().cpu())
            if not train_batches % 5:
                # plot the training loss
                plt.plot(global_train_loss, label='loss', color='blue')
                plt.ylim(0, min(1, max(global_train_loss) * 1.1))
                plt.savefig(log_dir / 'train_loss.png')
                plt.close()

        # log the training loss
        average_train_loss = train_loss / train_batches
        writer.add_scalar('Loss/Train', average_train_loss, epoch)

        logger.info(f"Epoch: {epoch + 1}/{args.epochs}, Train Loss: {average_train_loss:.4f}")

        model.eval()
        val_loss = 0.0
        val_acc = []
        val_f1 = []
        val_precision = []
        val_recall = []
        val_batches = 0

        # Validation loop
        with torch.no_grad():
            for inputs, original_imgs, targets, fnames, planes, slices in val_dataset.batch(
                args.batch_size, True, True, imwidths=imwidths, imheights=imheights
            ):
                predictions = run_inference(inputs.to(device), model)
                targets = targets.to(device)

                if args.out_classes > 1:
                    # If there are more than 1 output classes, stack the background class as well
                    targets = torch.hstack([torch.logical_not(targets), targets])

                # Calculate the loss
                loss = criterion(predictions, targets)
                val_loss += loss.item()

                global_val_loss.append(loss.detach().cpu())
                if not val_batches % 5:
                    # plot the validation loss
                    plt.plot(global_val_loss, label='loss', color='red')
                    plt.ylim(0, min(1, max(global_val_loss) * 1.1))
                    plt.savefig(log_dir / 'val_loss.png')
                    plt.close()

                # Calculate the metrics
                val_acc += [acc_metric(predictions, targets).item()]
                val_f1 += [f1_metric(predictions, targets).item()]
                val_precision += [precision_metric(predictions, targets).item()]
                val_recall += [recall_metric(predictions, targets).item()]
                val_batches += 1

        # log the validation metrics
        average_val_loss = val_loss / val_batches
        average_val_acc = torch.mean(torch.tensor(val_acc)) * 100
        average_val_f1 = torch.mean(torch.tensor(val_f1)) * 100
        average_val_precision = torch.mean(torch.tensor(val_precision)) * 100
        average_val_recall = torch.mean(torch.tensor(val_recall)) * 100

        save_path = os.path.join(log_dir, f"last.pth")
        state = {'model': model.float().state_dict(), 'optimizer': optimizer}
        torch.save(state, save_path)

        scheduler.step(average_val_loss)

        if average_val_loss < best_loss:
            best_loss = average_val_loss
            save_path = os.path.join(log_dir, f"best.pth")
            state = {'model': model.float().state_dict(), 'optimizer': optimizer}
            torch.save(state, save_path)

        lr = optimizer.param_groups[0]['lr']

        writer.add_scalar('LR', lr, epoch)
        writer.add_scalar('Loss/Val', average_val_loss, epoch)
        writer.add_scalar(f'Accuracy/Val/', average_val_acc, epoch)
        writer.add_scalar(f'F1/Val/', average_val_f1, epoch)
        writer.add_scalar(f'Precision/Val/', average_val_precision, epoch)
        writer.add_scalar(f'Recall/Val/', average_val_recall, epoch)

        logger.info(
            f"Epoch: {epoch + 1}/{args.epochs}, "
            f"LR: {lr} "
            f"Val Loss: {average_val_loss:.4f}, "
            f"Best Loss: {best_loss:.4f}, "
        )


if __name__ == "__main__":
    import sys

    sys.path.append('../src')
    main(args)
