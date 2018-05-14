import argparse
import json
from pathlib import Path
#
#
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.backends.cudnn


from unet_models import TernausNet34, UNet11, UNet16
from validation import validation_binary, validation_multi
from loss import LossBinary, LossMulti
from dataset import MapDataset
import utils

import os
import pandas as pd
from sklearn.model_selection import train_test_split

from transforms import (DualCompose,
                        OneOf,
                        OneOrOther,
                        Transpose,
                        Distort1,
                        Distort2,
                        RandomFilter,
                        RandomFlip,
                        ImageOnly,
                        Normalize,
                        HorizontalFlip,
                        VerticalFlip,
                        RandomCrop,
                        RandomRotate90,
                        ShiftScaleRotate,
                        RandomHueSaturationValue,
                        RandomBrightness,
                        RandomContrast,
                        Rescale,
                        augment)

TRAIN_ANNOTATIONS_PATH = "../mapping-challenge-starter-kit/data/train/annotation.json"
TRAIN_ANNOTATIONS_SMALL_PATH = "../mapping-challenge-starter-kit/data/train/annotation-small.json"


VAL_ANNOTATIONS_PATH = "../mapping-challenge-starter-kit/data/val/annotation.json"
VAL_ANNOTATIONS_SMALL_PATH = "../mapping-challenge-starter-kit/data/val/annotation-small.json"

def main():
    parser = argparse.ArgumentParser()

    arg = parser.add_argument
    arg('--jaccard-weight', default=1, type=float)
    arg('--device-ids', type=str, default='0', help='For example 0,1 to run on two GPUs')
    arg('--fold', type=int, help='fold', default=0)
    arg('--root', default='runs/debug', help='checkpoint root')
    arg('--batch-size', type=int, default=8)
    arg('--n-epochs', type=int, default=14)
    arg('--lr', type=float, default=0.000001)
    arg('--workers', type=int, default=8)
    arg('--type', type=str, default='binary', choices=['binary', 'parts', 'instruments'])
    arg('--model', type=str, default='TernausNet', choices=['UNet', 'UNet11', 'LinkNet34', 'TernausNet'])

    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(exist_ok=True, parents=True)

    if args.type == 'parts':
        num_classes = 3
    elif args.type == 'instruments':
        num_classes = 8
    else:
        num_classes = 1

    if args.model == 'TernausNet':
        model = TernausNet34(num_classes=num_classes)
    else:
        model = TernausNet34(num_classes=num_classes)

    if torch.cuda.is_available():
        if args.device_ids:
            device_ids = list(map(int, args.device_ids.split(',')))
        else:
            device_ids = None
        model = nn.DataParallel(model, device_ids=device_ids).cuda()

    if args.type == 'binary':
        loss = LossBinary(jaccard_weight=args.jaccard_weight)
    else:
        loss = LossMulti(num_classes=num_classes, jaccard_weight=args.jaccard_weight)

    cudnn.benchmark = True

    def make_loader(file_names, shuffle=False, transform=None, mode='train',problem_type='binary'):
        return DataLoader(
            dataset=MapDataset(file_names, transform=transform, problem_type=problem_type, mode=mode),
            shuffle=shuffle,
            num_workers=args.workers,
            batch_size=args.batch_size,
            pin_memory=torch.cuda.is_available()
        )

    # labels = pd.read_csv('data/stage1_train_labels.csv')
    # labels = os.listdir('data/stage1_train_')
    # train_file_names, val_file_names = train_test_split(labels, test_size=0.2, random_state=42)

    # print('num train = {}, num_val = {}'.format(len(train_file_names), len(val_file_names)))

    # train_transform = DualCompose([
    #     HorizontalFlip(),
    #     VerticalFlip(),
    #     RandomCrop([256, 256]),
    #     RandomRotate90(),
    #     ShiftScaleRotate(),
    #     ImageOnly(RandomHueSaturationValue()),
    #     ImageOnly(RandomBrightness()),
    #     ImageOnly(RandomContrast()),
    #     ImageOnly(Normalize())
    # ])
    train_transform = DualCompose([
        OneOrOther(
            *(OneOf([
                Distort1(distort_limit=0.05, shift_limit=0.05),
                Distort2(num_steps=2, distort_limit=0.05)]),
              ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.10, rotate_limit=45)), prob=0.5),
        RandomRotate90(),
        RandomCrop([256, 256]),
        RandomFlip(prob=0.5),
        Transpose(prob=0.5),
        ImageOnly(RandomContrast(limit=0.2, prob=0.5)),
        ImageOnly(RandomFilter(limit=0.5, prob=0.2)),
        ImageOnly(RandomHueSaturationValue(prob=0.2)),
        ImageOnly(RandomBrightness()),
        ImageOnly(Normalize())
    ])

    val_transform = DualCompose([
        # RandomCrop([256, 256]),
        Rescale([256, 256]),
        ImageOnly(Normalize())
    ])

    train_loader = make_loader(TRAIN_ANNOTATIONS_PATH, shuffle=True, transform=train_transform, problem_type=args.type)
    valid_loader = make_loader(VAL_ANNOTATIONS_PATH, transform=val_transform, mode='valid', problem_type=args.type)

    root.joinpath('params.json').write_text(
        json.dumps(vars(args), indent=True, sort_keys=True))

    if args.type == 'binary':
        valid = validation_binary
    else:
        valid = validation_multi

    utils.train(
        init_optimizer=lambda lr: Adam(model.parameters(), lr=lr),
        args=args,
        model=model,
        criterion=loss,
        train_loader=train_loader,
        valid_loader=valid_loader,
        validation=valid,
        fold=args.fold,
        num_classes=num_classes
    )

if __name__ == '__main__':
    main()
