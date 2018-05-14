import json
from datetime import datetime
from pathlib import Path
import copy
from tensorboardX import SummaryWriter

import random
import numpy as np

import torch
from torch.autograd import Variable
import torchvision.utils as vutils
import tqdm



def variable(x, volatile=False):
    if isinstance(x, (list, tuple)):
        return [variable(y, volatile=volatile) for y in x]
    return cuda(Variable(x, volatile=volatile))


def cuda(x):
    return x.cuda(async=True) if torch.cuda.is_available() else x


def write_event(log, step: int, **data):

    data['step'] = step
    data['dt'] = datetime.now().isoformat()
    # data['loss'] = data['loss'].astype(np.float64)
    wt = json.dumps(data, sort_keys=True)
    log.write(wt)
    log.write('\n')
    log.flush()


def train(args, model, criterion, train_loader, valid_loader, validation, init_optimizer, n_epochs=None, fold=None, num_classes=None):
    lr = args.lr
    n_epochs = n_epochs or args.n_epochs
    optimizer = init_optimizer(lr)
    # writer = SummaryWriter()
    root = Path(args.root)
    model_path = root / 'model_{fold}.pt'.format(fold=fold)
    best_model_path = root / 'best_model_{fold}.pt'.format(fold=fold)
    if model_path.exists():
        state = torch.load(str(model_path))
        epoch = state['epoch']
        step = state['step']
        model.load_state_dict(state['model'])
        print('Restored model, epoch {}, step {:,}'.format(epoch, step))
    else:
        epoch = 1
        step = 0

    save = lambda ep: torch.save({
        'model': model.state_dict(),
        'epoch': ep,
        'step': step,
    }, str(model_path))

    save_best = lambda ep: torch.save({
        'model': best_model.state_dict(),
        'epoch': ep,
        'step': step,
    }, str(best_model_path))

    report_each = 50
    log = root.joinpath('train_{fold}.log'.format(fold=fold)).open('at', encoding='utf8')
    valid_losses = []
    for epoch in range(epoch, n_epochs + 1):
        min_val_loss = 10
        model.train()
        random.seed()
        tq = tqdm.tqdm(total=(len(train_loader) * args.batch_size))
        tq.set_description('Epoch {}, lr {}'.format(epoch, lr))
        losses = []
        tl = train_loader
        try:
            mean_loss = 0
            for i, (inputs, targets) in enumerate(tl):
                inputs, targets = variable(inputs), variable(targets)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                optimizer.zero_grad()
                batch_size = inputs.size(0)
                loss.backward()
                optimizer.step()
                step += 1
                tq.update(batch_size)
                losses.append(loss.data[0])
                mean_loss = np.mean(losses[-report_each:])
                tq.set_postfix(loss='{:.5f}'.format(mean_loss))
                if i and i % report_each == 0:
                    write_event(log, step, loss=mean_loss.astype(np.float64))
                    # writer.add_scalar('runs/tensorboard', mean_loss, step)
            # writer.add_scalar('runs/mean_loss', mean_loss, step)
            write_event(log, step, loss=mean_loss.astype(np.float64))
            tq.close()
            save(epoch + 1)
            valid_metrics = validation(model, criterion, valid_loader, epoch, num_classes)
            # writer.add_scalar('runs/valid_loss', valid_metrics['valid_loss'], step)
            # writer.add_scalar('runs/jaccard_loss', valid_metrics['jaccard_loss'], step)
            write_event(log, step, **valid_metrics)
            valid_loss = valid_metrics['valid_loss']
            if valid_loss < min_val_loss:
                min_val_loss = valid_loss
                best_model = copy.deepcopy(model)
                save_best(epoch + 1)
                # print(min_val_loss)
            valid_losses.append(valid_loss)
        except KeyboardInterrupt:
            tq.close()
            print('Ctrl+C, saving snapshot')
            save(epoch)
            print('done.')
            return

# def calc_watershed(mask):
#     full_mask = mask[0]
#     seed = mask[1]
#     border = mask[2]
#
#     distance = ndi.distance_transform_edt(mask)
#     # local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)),
#     #                             labels=image)
#     ret, markers = cv2.connectedComponents((seed > 0.7).astype(np.int8))
#     # markers = ndi.label(local_maxi)[0]
#     labels = watershed(-distance, markers, mask=full_mask)
#     return labels
#
# def read_masks(id):
#     masks = "data/stage1_train_/{}/masks/*.png".format(id)
#     masks = skimage.io.imread_collection(masks).concatenate()
#     height, width = masks[0].shape
#     num_masks = masks.shape[0]
#
#     # Make a ground truth label image (pixel value is index of object label)
#     labels = np.zeros((height, width), np.uint16)
#     for index in range(0, num_masks):
#         labels[masks[index] > 0] = index + 1
#     return labels
