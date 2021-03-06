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
# import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from pycocotools.coco import COCO
from cocoeval import COCOeval

from pycocotools import mask as cocomask
from skimage import measure
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
        if not idx[0] % 10:
            save_valid_results(inputs, targets, outputs, idx[0], epoch)
        losses.append(loss.data[0])
        jaccard += [get_jaccard( targets, (outputs > 0).float()).data[0]]

    valid_loss = np.mean(losses)  # type: float
    # valid_rec = np.mean(recs)
    # valid_prec = np.mean(prec)
    valid_jaccard = np.mean(jaccard)
    print('Valid loss: {:.5f}, jaccard: {:.5f}'.format(valid_loss, valid_jaccard))
    # print("Average Precision : {:.5f} || Average Recall : {:.5f}".format(valid_prec, valid_rec))

    metrics = {'valid_loss': valid_loss.astype(np.float64), 'jaccard_loss': valid_jaccard.astype(np.float64)}
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
    fortran_binary_mask = np.asfortranarray(in_mask.astype(np.uint8))
    # encoded_ground_truth = maskUtils(fortran_ground_truth_binary_mask)
    encoded_ = cocomask.encode(fortran_binary_mask)
    ground_truth_bounding_box = cocomask.toBbox(encoded_)
    # contours = measure.find_contours(in_mask, 0.5)
    encoded_["counts"] = encoded_["counts"].decode("UTF-8")
    # _result["segmentation"] = _mask

    annotation = {
        "segmentation": encoded_,
        "image_id": int(image_id),
        "bbox": np.around(ground_truth_bounding_box.tolist()).tolist(),
        "category_id": 100,
        "score": 1,
    }

    # for contour in contours:
    #     contour = np.flip(contour, axis=1)
    #     segmentation = contour.ravel().tolist()
    #     annotation["segmentation"].append(segmentation)
    #
    # annotation['segmentation'] = annotation['segmentation']
    return annotation


def pred_to_coco(preds):
    """

    :param preds: list og binary masks
    :return:
    """

    anns = []
    kernel = np.ones((5, 5), np.uint8)
    for pred in tqdm(preds):
        mask = pred
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        im2, contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cnt_areas = [cv2.contourArea(cnt) for cnt in contours]
        contours = np.array(contours)[np.array(cnt_areas) > 150]
        for cnt in range(len(contours)):
            dr_2 = np.zeros(mask.shape)
            draw_img = cv2.drawContours(dr_2, contours, cnt, 1, -1)
            fortran_binary_mask = np.asfortranarray(draw_img.astype(np.uint8))
            encoded_ = cocomask.encode(fortran_binary_mask)
            ground_truth_bounding_box = cocomask.toBbox(encoded_)
            encoded_["counts"] = encoded_["counts"].decode("UTF-8")

            annotation = {
                "segmentation": encoded_,
                "image_id": pred['image_id'],  # int(image_id),
                "bbox": np.around(ground_truth_bounding_box.tolist()).tolist(),
                "category_id": 100,
                "score": 1,
            }
            anns.append(annotation)
    return anns


def create_annotations(meta, predictions, logger, category_ids, save=True, experiment_dir='./'):
    '''
    :param meta: pd.DataFrame with metadata
    :param predictions: list of labeled masks or numpy array of size [n_images, im_height, im_width]
    :param logger:
    :param save: True, if one want to save submission, False if one want to return it
    :param experiment_dir: path to save submission
    :return: submission if save==False else True
    '''
    annotations = []
    logger.info('Creating submission')
    for image_id, prediction in zip(meta["ImageId"].values, predictions):
        score = 1.0
        for category_nr, category_instances in enumerate(prediction):
            if category_ids[category_nr] != None:
                masks = decompose(category_instances)
                for mask_nr, mask in enumerate(masks):
                    annotation = {}
                    annotation["image_id"] = int(image_id)
                    annotation["category_id"] = category_ids[category_nr]
                    annotation["score"] = score
                    annotation["segmentation"] = rle_from_binary(mask.astype('uint8'))
                    annotation['segmentation']['counts'] = annotation['segmentation']['counts'].decode("UTF-8")
                    annotation["bbox"] = bounding_box_from_rle(rle_from_binary(mask.astype('uint8')))
                    annotations.append(annotation)
    if save:
        submission_filepath = os.path.join(experiment_dir, 'submission.json')
        with open(submission_filepath, "w") as fp:
            fp.write(str(json.dumps(annotations)))
            logger.info("Submission saved to {}".format(submission_filepath))
            logger.info('submission head \n\n{}'.format(annotations[0]))
        return True
    else:
        return annotations


def rle_from_binary(prediction):
    prediction = np.asfortranarray(prediction)
    return cocomask.encode(prediction)


def bounding_box_from_rle(rle):
    return list(cocomask.toBbox(rle))

def decompose(labeled):
    nr_true = labeled.max()
    masks = []
    for i in range(1, min(nr_true + 1, 20)):
        msk = labeled.copy()
        msk[msk != i] = 0.
        msk[msk == i] = 255.
        masks.append(msk)

    if not masks:
        return [labeled]
    else:
        return masks

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
    plt.imshow(inputs[0][0].data.cpu().numpy())
    fig.add_subplot(1, 3, 2)
    plt.imshow(targets[0][0].data.cpu().numpy(), 'gray')
    fig.add_subplot(1, 3, 3)
    plt.imshow(outputs[0][0].data.cpu().numpy(), 'gray')
    plt.savefig('runs/valid_res/{}_{}.jpg'.format(idx, epoch))
    plt.close()

def bounding_box_from_points(points):
    """
        This function only supports the `poly` format.
    """
    points = np.array(points).flatten()
    even_locations = np.arange(points.shape[0]/2) * 2
    odd_locations = even_locations + 1
    X = np.take(points, even_locations.tolist())
    Y = np.take(points, odd_locations.tolist())
    bbox = [X.min(), Y.min(), X.max()-X.min(), Y.max()-Y.min()]
    bbox = [int(b) for b in bbox]
    return bbox