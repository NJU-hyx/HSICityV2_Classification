import argparse
import os
import timeit
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter

import _init_paths
import models
import datasets
from core.criterion import CrossEntropy
from core.function import testval, testvalOnlyBackground
from utils.utils import create_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')

    parser.add_argument('--path',
                        # default='/home/huangyx/code/HSIseg/data/')
                        default='data/')

    parser.add_argument('--output_dir',
                        default='output', type=str)
    parser.add_argument('--log_dir',
                        default='log', type=str)
    parser.add_argument('--model_file',
                        default='./output/hsicity/hsicity2/best17.pth', 
                        # default='', 
                        type=str)

    parser.add_argument('--model',
                        default='resnet')
    parser.add_argument('--model_name',
                        default='resnet50')
    parser.add_argument('--resume',
                        default=False)
    parser.add_argument('--num_classes',
                        default=19, type=int)
    parser.add_argument('--ignore_label',
                        default=255, type=int)
    parser.add_argument('--test_row_size',
                        default=200, type=int)

    parser.add_argument('--exp_name',
                        default='hsicity2')

    parser.add_argument("--local_rank", type=int)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    logger, final_output_dir, _ = create_logger(
        args.output_dir, 'hsicity', args.model, args.log_dir, args.exp_name, 'test'
    )

    # cudnn related setting
    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.enabled = True

    gpus = [0, 1, 2, 3]
    distributed = len(gpus) > 1
    device = torch.device(f'cuda:{args.local_rank}')

    # build model
    model = eval('models.' + args.model + '.' +
                 args.model_name)(num_classes=args.num_classes)
    # model = eval('models.' + args.model + '.' +
    #                args.model_name)(128, 19)
    
    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://",
        )

    if args.model_file:
        model_state_file = args.model_file
    else:
        model_state_file = os.path.join(final_output_dir,
                                        'final_state.pth')
    logger.info('=> loading model from {}'.format(model_state_file))

    pretrained_dict = torch.load(model_state_file)
    model_dict = model.state_dict()
    pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items()
                       if k[7:] in model_dict.keys()}
    # for k, _ in pretrained_dict.items():
        # logger.info(
        #     '=> loading {} from pretrained model'.format(k))
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    # model = nn.DataParallel(model, device_ids=gpus).cuda()    
    model = model.to(device)
    model = nn.parallel.DistributedDataParallel(
        model, device_ids=[args.local_rank], output_device=args.local_rank)

    # prepare data
    test_size = (1889, 1422)
    test_dataset = eval('datasets.hsicity2')(
        root='/data/huangyx/HSICityV2/',
        # root=r'F:\database\HSIcityscapes',
        list_path='data/list/hsicity2/train.lst',
        num_samples=None,
        num_classes=args.num_classes,
        multi_scale=False,
        flip=False,
        ignore_label=args.ignore_label,
        base_size=1889,
        crop_size=test_size,
        center_crop_test=False,
        downsample_rate=1)

    if distributed:
        test_sampler = DistributedSampler(test_dataset)
    else:
        test_sampler = None
    
    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        sampler=test_sampler)

    start = timeit.default_timer()
    res = testvalOnlyBackground(args.num_classes,
                                args.ignore_label,
                                test_dataset,
                                testloader,
                                model,
                                sv_pred=True,
                                sv_dir='result',
                                batch_size=2048
                                )
    # mean_IoU, IoU_array, pixel_acc, mean_acc = testval(args.num_classes,
    #                                                    args.ignore_label,
    #                                                    args.test_row_size,
    #                                                    test_dataset,
    #                                                    testloader,
    #                                                    model,
    #                                                    sv_pred=True,
    #                                                    sv_dir='result'
    #                                                    )

    msg = 'MeanIU: {: 4.4f}, Pixel_Acc: {: 4.4f}, \
        Mean_Acc: {: 4.4f}, Class IoU: '.format(mean_IoU,
                                                pixel_acc, mean_acc)
    logging.info(msg)
    logging.info(IoU_array)

    end = timeit.default_timer()
    logger.info('Mins: %d' % np.int((end - start) / 60))
    logger.info('Done')


if __name__ == '__main__':
    main()
