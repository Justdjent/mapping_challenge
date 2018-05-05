import numpy as np
import skimage.morphology
import cv2
import pandas as pd

def rle_of_binary(x):
    """ Run length encoding of a binary 2D array. """
    dots = np.where(x.T.flatten() == 1)[0] # indices from top to down
    run_lengths = []
    prev = -2
    for b in dots:
        if ( b > prev +1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def mask_to_rle(mask, cutoff=.5, min_object_size=1.):
    """ Return run length encoding of mask. """
    # segment image and label different objects
    lab_mask = skimage.morphology.label(mask > cutoff)

    # Keep only objects that are large enough.
    (mask_labels, mask_sizes) = np.unique(lab_mask, return_counts=True)
    if (mask_sizes < min_object_size).any():
        mask_labels = mask_labels[mask_sizes < min_object_size]
        for n in mask_labels:
            lab_mask[lab_mask == n] = 0
        lab_mask = skimage.morphology.label(lab_mask > cutoff)

        # Loop over each object excluding the background labeled by 0.
    for i in range(1, lab_mask.max() + 1):
        yield rle_of_binary(lab_mask == i)


if __name__ == "__name__":
    test_df = pd.read_csv('data/stage1_sample_submission.csv')
    # Run length encoding of predicted test masks.
    test_pred_rle = []
    test_pred_ids = []
    for n, id_ in enumerate(test_df['img_id']):
        # min_object_size = 20*test_df.loc[n,'img_height']*test_df.loc[n,'img_width']/(256*256)
        mask = cv2.imread('output/submission_mask/{}'.format(id_), 0)
        rle = list(mask_to_rle(mask))
        test_pred_rle.extend(rle)
        test_pred_ids.extend([id_]*len(rle))

    print('test_pred_ids.shape = {}'.format(np.array(test_pred_ids).shape))
    print('test_pred_rle.shape = {}'.format(np.array(test_pred_rle).shape))

    #  Create submission file
    sub = pd.DataFrame()
    sub['ImageId'] = test_pred_ids
    sub['EncodedPixels'] = pd.Series(test_pred_rle).apply(lambda x: ' '.join(str(y) for y in x))
    sub.to_csv('sub-dsbowl2018-1.csv', index=False)
    sub.head()