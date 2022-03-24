import time
import logging
import numpy as np
from tqdm import tqdm
import os

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn import functional as F

from utils.utils import *

import datasets


def get_world_size():
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size()


def get_rank():
    if not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_rank()


def adjust_learning_rate(optimizer, base_lr, max_iters,
                         cur_iters, power=0.9):
    lr = base_lr*((1-float(cur_iters)/max_iters)**(power))
    optimizer.param_groups[0]['lr'] = lr
    return lr


def train(epoch, num_epoch, print_freq, epoch_iters, base_lr,
          trainloader, optimizer, criterion, model, writer_dict, device):
    # Training
    model.train()
    tic = time.time()
    batch_time = 0.0
    count = 0
    losses = 0.0
    time_count = 0
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']
    rank = get_rank()
    world_size = get_world_size()

    for i_iter, batch in enumerate(trainloader):
        images, labels, _, _ = batch
        inputcubes, labelcubes = covertBatch2TrainCubes(
            images, labels, windowSize=25, cubeSize=3000)

        hsiCubedatas = datasets.hsicube(inputcubes, labelcubes)
        hsiDataloader = torch.utils.data.DataLoader(
            hsiCubedatas,
            batch_size=512,
            shuffle=True,
        )
        for index, (batchImage, batchLabel) in enumerate(hsiDataloader):
            batchImage = batchImage.to(device)
            batchLabel = batchLabel.long().to(device)
            # spectra, neighbor = TwoCNN_dataprocess(batchImage)
            outputs = model(batchImage)
            # outputs = model(spectra, neighbor)
            loss = criterion(outputs, batchLabel)

            model.zero_grad()
            loss.backward()
            optimizer.step()

            # update average loss
            count += 1
            losses += loss.item()

            print('\r' + '[{}]{:.5f}'.format(i_iter,
                                             loss.item()), end='', flush=True)

        # measure elapsed time
        per_batch_time = time.time() - tic
        tic = time.time()
        batch_time += per_batch_time
        time_count += 1

        # lr = base_lr
        lr = adjust_learning_rate(optimizer,
                                  base_lr,
                                  num_epoch,
                                  epoch)        

        if i_iter % print_freq == 0 and rank == 0:
            print_loss = losses / count
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'lr: {:.6f}, Loss: {:.6f}'.format(
                      epoch, num_epoch, i_iter, epoch_iters,
                      batch_time / count, lr, print_loss)
            logging.info(msg)

            writer.add_scalar('train_loss', print_loss, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

        # recount
        batch_time = 0.0
        count = 0
        losses = 0.0
        time_count = 0


def validate(num_class, ignore_label, testloader, criterion, model, writer_dict, device):
    model.eval()
    count = 0
    losses = 0.0
    confusion_matrix = np.zeros(
        (num_class, num_class))

    with torch.no_grad():
        for _, batch in enumerate(testloader):
            image, label, _, _ = batch
            inputcubes, labelcubes = covertBatch2TrainCubes(
                image, label, windowSize=25, cubeSize=10000)

            hsiCubedatas = datasets.hsicube(inputcubes, labelcubes)
            hsiDataloader = torch.utils.data.DataLoader(hsiCubedatas,
                                                        batch_size=512,
                                                        shuffle=True
                                                        )
            for index, (batchImage, batchLabel) in enumerate(hsiDataloader):
                batchImage = batchImage.to(device)
                batchLabel = batchLabel.long().to(device)
                size = batchLabel.size()
                spectra, neighbor = TwoCNN_dataprocess(batchImage)
                pred = model(spectra, neighbor)
                # pred = model(batchImage)
                loss = criterion(pred, batchLabel)

                losses += loss.item()
                count += 1

                confusion_matrix += get_confusion_matrix_1d(
                    batchLabel,
                    pred,
                    size,
                    num_class,
                    ignore_label)
        pos = confusion_matrix.sum(1)
        res = confusion_matrix.sum(0)
        tp = np.diag(confusion_matrix)
        IoU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IoU = IoU_array.mean()

    print_loss = losses / count

    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']
    writer.add_scalar('valid_loss', print_loss, global_steps)
    writer_dict['valid_global_steps'] = global_steps + 1
    return print_loss, mean_IoU, IoU_array


def testval(num_class, ignore_label, rowSize, test_dataset, testloader, model, sv_dir='', sv_pred=False):
    model.eval()
    confusion_matrix = np.zeros(
        (num_class + 1, num_class + 1))
    with torch.no_grad():
        for index, batch in enumerate(tqdm(testloader)):
            image, label, _, name = batch
            size = label.size()
            pred = torch.rand(size[1], size[2], num_class)

            for j in range(0, size[1], rowSize):  # 多�?�测�??
                row_size = rowSize
                if size[1] - j < rowSize:
                    row_size = size[1] - j
                imageCubes, labelCubes = createTestCube(
                    image[0], label[0], windowSize=25, r=j, size=row_size)
                hsiDataset = datasets.hsicube(imageCubes, labelCubes)
                hsiDataloader = torch.utils.data.DataLoader(hsiDataset,
                                                            batch_size=size[2] *
                                                            row_size
                                                            )

                for _, (inputCube, _) in enumerate(hsiDataloader):
                    inputCube = inputCube.cuda()
                    # spectra, neighbor = TwoCNN_dataprocess(inputCube)
                    with torch.no_grad():
                        output = model.forward(inputCube)
                        # output = model(spectra, neighbor)
                    pred[j:j + row_size, :,
                         :] = output.reshape((row_size, size[2], num_class))

                print('\r' + '{}'.format(j), end='', flush=True)

            confusion_matrix += get_confusion_matrix(
                label,
                pred,
                size,
                num_class + 1,
                ignore_label)

            if sv_pred:
                sv_path = os.path.join(sv_dir, 'cnn_hsi')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)

                test_dataset.save_pred(pred, sv_path, name)
                # test_dataset.save_pred_gray(pred, sv_path, name)

            # if index % 100 == 0:
            logging.info('processing: %d images' % index)
            pos = confusion_matrix.sum(1)
            res = confusion_matrix.sum(0)
            tp = np.diag(confusion_matrix)
            IoU_array = (tp / np.maximum(1.0, pos + res - tp))
            mean_IoU = IoU_array.mean()
            logging.info('mIoU: %.4f' % (mean_IoU))

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    pixel_acc = tp.sum() / pos.sum()
    mean_acc = (tp / np.maximum(1.0, pos)).mean()
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()

    return mean_IoU, IoU_array, pixel_acc, mean_acc

def testvalOnlyBackground(num_class, ignore_label, test_dataset, testloader, model, batch_size=0, sv_dir='', sv_pred=False):
    model.eval()
    confusion_matrix = np.zeros(
        (num_class + 1, num_class + 1))
    with torch.no_grad():
        for index, batch in enumerate(tqdm(testloader)):
            image, label, _, name = batch
            size = label.size()
            pred = label[0].cuda()

            h, w = 0, 0
            while h != size[2] - 1 or w != size[1] - 1:
                imageCubes, labelCubes, ret = createOnlyBackgroundTestCube(
                    image[0], label[0], windowSize=17, h=h, w=w, size=batch_size)
                imageCubes = imageCubes.permute(0, 3, 1, 2)
                with torch.no_grad():
                        output = model.forward(imageCubes)
                
                k = 0
                while k != labelCubes.shape[0]:
                    if label[0, h, w] == 255:
                        pred[h, w] = torch.argmax(output[k], dim=0)
                        k += 1
                    h = h + (w + 1) // size[2]
                    w = (w + 1) % size[2]
            
                if ret:
                    break
                # print('\r' + f'{h} {w}', end='', flush=True)

            if sv_pred:
                sv_path = os.path.join(sv_dir, 'refined_label')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path, name)
                # test_dataset.save_pred_gray(pred, sv_path, name)

            # if index % 100 == 0:
            logging.info('processing: %d images' % index)

    return 0

