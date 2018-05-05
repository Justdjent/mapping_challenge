import numpy as np
import utils
import json
from pycocotools import mask
from skimage import measure
import random
import json
import numpy as np
import argparse
import base64
import glob
import os
import matplotlib.pyplot as plt

import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from pycocotools.coco import COCO
from cocoeval import COCOeval

from torch import nn
from torch.nn import functional as F

# writer = SummaryWriter()

def validation_binary(model: nn.Module, criterion, valid_loader, epoch, num_classes=None):
    model.eval()
    losses = []
    accs = []
    recs = []
    jaccard = []

    for i, (inputs, targets, idx) in enumerate(valid_loader):
        inputs = utils.variable(inputs, volatile=True)
        targets = utils.variable(targets)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        # prec, rec = calc_coco_metric(targets, outputs)
        # accs.append(prec)
        # recs.append(rec)
        # if i % 10 != 0:
        #     x = vutils.make_grid(outputs.data, normalize=True, scale_each=True)
            # writer.add_image('Image', x, i)
        save_valid_results(inputs, targets, outputs, idx, epoch)
        losses.append(loss.data[0])
        jaccard += [get_jaccard( targets, (outputs > 0).float()).data[0]]

    valid_loss = np.mean(losses)  # type: float
    # valid_rec = np.mean(recs)
    # valid_prec = np.mean(prec)
    valid_jaccard = np.mean(jaccard)
    print('Valid loss: {:.5f}, jaccard: {:.5f}'.format(valid_loss, valid_jaccard))
    # print("Average Precision : {:.5f} || Average Recall : {:.5f}".format(valid_prec, valid_rec))

    metrics = {'valid_loss': valid_loss, 'jaccard_loss': valid_jaccard}
    # metrics = {'valid_loss': valid_loss, 'jaccard_loss': valid_jaccard,
    #            'average_precision': valid_prec, 'average_recall': valid_rec}

    return metrics


def get_jaccard(y_true, y_pred):
    epsilon = 1e-15
    intersection = (y_pred * y_true).sum(dim=-2).sum(dim=-1)
    union = y_true.sum(dim=-2).sum(dim=-1) + y_pred.sum(dim=-2).sum(dim=-1)

    return (intersection / (union - intersection + epsilon)).mean()


def validation_multi(model: nn.Module, criterion, valid_loader, num_classes):
    model.eval()
    losses = []
    jaccard = []
    # confusion_matrix = np.zeros(
    #     (num_classes, num_classes), dtype=np.uint32)
    for inputs, targets in valid_loader:
        inputs = utils.variable(inputs, volatile=True)
        targets = utils.variable(targets)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        losses.append(loss.data[0])
        for cls in range(num_classes):
            if cls == 0:
                jaccard_target = (targets[:, 0] == cls).float()
            else:
                jaccard_target = (targets[:, cls - 1] == 1).float()
            # jaccard_output = outputs[:, cls].exp()

            jaccard_output = F.sigmoid(outputs[:, cls])
            jaccard += [get_jaccard(jaccard_target, jaccard_output)]
        #     intersection = (jaccard_output * jaccard_target).sum()
        #
        #     union = jaccard_output.sum() + jaccard_target.sum() + eps
        # output_classes = outputs[:, 0].data.cpu().numpy().argmax(axis=1)
        # target_classes = targets[:, 0].data.cpu().numpy()
        # confusion_matrix += calculate_confusion_matrix_from_arrays(
        #     output_classes, target_classes, num_classes)

    # confusion_matrix = confusion_matrix[1:, 1:]  # exclude background
    valid_loss = np.mean(losses)  # type: float
    valid_jaccard = np.mean(jaccard)
    # ious = {'iou_{}'.format(cls + 1): iou
    #         for cls, iou in enumerate(calculate_iou(confusion_matrix))}
    #
    # dices = {'dice_{}'.format(cls + 1): dice
    #          for cls, dice in enumerate(calculate_dice(confusion_matrix))}
    #
    # average_iou = np.mean(list(ious.values()))
    # average_dices = np.mean(list(dices.values()))

    # print(
    #     'Valid loss: {:.4f}, average IoU: {:.4f}, average Dice: {:.4f}'.format(valid_loss, average_iou, average_dices))
    # print(
    #     'Valid loss: {:.4f}'.format(valid_loss))
    print('Valid loss: {:.5f}, jaccard: {:.5f}'.format(valid_loss, valid_jaccard.data[0]))
    # metrics = {'valid_loss': valid_loss, 'iou': average_iou}
    # metrics = {'valid_loss': valid_loss}
    metrics = {'valid_loss': valid_loss, 'jaccard_loss': valid_jaccard.data[0]}
    # metrics.update(ious)
    # metrics.update(dices)
    return metrics


def calculate_confusion_matrix_from_arrays(prediction, ground_truth, nr_labels):
    replace_indices = np.vstack((
        ground_truth.flatten(),
        prediction.flatten())
    ).T
    confusion_matrix, _ = np.histogramdd(
        replace_indices,
        bins=(nr_labels, nr_labels),
        range=[(0, nr_labels), (0, nr_labels)]
    )
    confusion_matrix = confusion_matrix.astype(np.uint32)
    return confusion_matrix


def calculate_iou(confusion_matrix):
    ious = []
    for index in range(confusion_matrix.shape[0]):
        true_positives = confusion_matrix[index, index]
        false_positives = confusion_matrix[:, index].sum() - true_positives
        false_negatives = confusion_matrix[index, :].sum() - true_positives
        denom = true_positives + false_positives + false_negatives
        if denom == 0:
            iou = 0
        else:
            iou = float(true_positives) / denom
        ious.append(iou)
    return ious


def calculate_dice(confusion_matrix):
    dices = []
    for index in range(confusion_matrix.shape[0]):
        true_positives = confusion_matrix[index, index]
        false_positives = confusion_matrix[:, index].sum() - true_positives
        false_negatives = confusion_matrix[index, :].sum() - true_positives
        denom = 2 * true_positives + false_positives + false_negatives
        if denom == 0:
            dice = 0
        else:
            dice = 2 * float(true_positives) / denom
        dices.append(dice)
    return dices

def calc_metric(labels, y_pred):
    # Compute number of objects
    true_objects = len(np.unique(labels))
    pred_objects = len(np.unique(y_pred))
    print("Number of true objects:", true_objects)
    print("Number of predicted objects:", pred_objects)
    # Compute intersection between all objects
    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins = true_objects)[0]
    area_pred = np.histogram(y_pred, bins = pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    union = union[1:,1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1   # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        p = tp / (tp + fp + fn)
        print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)
    print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)

def convert_bin_coco(in_mask, image_id):

    # ground_truth_binary_mask = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                                      [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
    #                                      [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
    #                                      [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
    #                                      [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
    #                                      [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)

    fortran_ground_truth_binary_mask = np.asfortranarray(in_mask.astype(np.uint8))
    encoded_ground_truth = mask.encode(fortran_ground_truth_binary_mask)
    ground_truth_area = mask.area(encoded_ground_truth)
    ground_truth_bounding_box = mask.toBbox(encoded_ground_truth)
    contours = measure.find_contours(in_mask, 0.5)
    annotation = {
        # "id": id,
        "segmentation": [],
        "area": ground_truth_area.tolist(),
        "iscrowd": False,
        "image_id": image_id,
        "bbox": ground_truth_bounding_box.tolist(),
        "category_id": 100,
    }

    for contour in contours:
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        annotation["segmentation"].append(segmentation)

    # print(json.dumps(annotation, indent=4))
    return annotation

def calc_coco_metric(targets, outputs):
    annotations_gt = []
    annotations_r = []
    for id, m in enumerate(range(targets.shape[0])):
        annotations_gt.append(convert_bin_coco(targets.data.numpy()[m][0], id))
        annotations_r.append(convert_bin_coco(outputs.data.numpy()[m][0], id))

    ground_truth_annotations = COCO(annotations_gt)
    results = ground_truth_annotations.loadRes(annotations_r)
    cocoEval = COCOeval(ground_truth_annotations, results, 'segm')
    cocoEval.evaluate()
    cocoEval.accumulate()
    average_precision = cocoEval._summarize(ap=1, iouThr=0.5, areaRng="all", maxDets=100)
    average_recall = cocoEval._summarize(ap=0, iouThr=0.5, areaRng="all", maxDets=100)
    # print("Average Precision : {} || Average Recall : {}".format(average_precision, average_recall))
    return average_precision, average_recall
# def single_annotation(image_id, number_of_points=10):
#     _result = {}
#     _result["image_id"] = image_id
#     _result["category_id"] = 100  # as 100 is the category_id of Building
#     _result["score"] = np.random.rand()  # a random score between 0 and 1
#
#     _result["segmentation"] = single_segmentation(IMAGE_WIDTH, IMAGE_HEIGHT, number_of_points=number_of_points)
#     _result["bbox"] = bounding_box_from_points(_result["segmentation"])
#     return _result

# predictions = []
# for image_id in IMAGE_IDS:
#     number_of_annotations = random.randint(0, MAX_NUMBER_OF_ANNOTATIONS)
#     for _idx in range(number_of_annotations):
#         _annotation = single_annotation(image_id)
#         predictions.append(_annotation)
def save_valid_results(inputs, targets, outputs, idx, epoch):
    fig = plt.figure(figsize=(10, 10))
    fig.add_subplot(1, 3, 1)
    plt.imshow(inputs[0][0].data.numpy())
    fig.add_subplot(1, 3, 2)
    plt.imshow(targets[0][0].data.numpy())
    fig.add_subplot(1, 3, 3)
    plt.imshow(outputs[0][0].data.numpy())
    plt.savefig('runs/valid_res/{}_{}.jpg'.format(idx, epoch))