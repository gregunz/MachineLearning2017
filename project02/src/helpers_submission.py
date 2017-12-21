import numpy as np


def patch_to_label(patch, foreground_threshold=0.25):
    """
    From a patch return a label (0 = background, 1 = foreground)

    :param patch: a patch
    :param foreground_threshold: percentage of pixels > 1 required to assign a foreground label to a patch
    :return: the label
    """
    df = np.mean(patch)
    if df > foreground_threshold:
        return 1
    else:
        return 0


def mask_to_submission_strings(img, img_number):
    """
    Given a single image, outputs the strings that should go into the submission file

    :param img: the image
    :param img_number: the image number
    :return: a generator of the string for the submission
    """
    patch_size = 16
    for j in range(0, img.shape[1], patch_size):
        for i in range(0, img.shape[0], patch_size):
            patch = img[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch)
            yield ("{:03d}_{}_{},{}".format(img_number, j, i, label))


def masks_to_submission(submission_filename, masks):
    """
    Converts images into a submission file

    :param submission_filename: file name of the submission
    :param masks: masks (prediction) of each images
    :return: None
    """
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for idx, mask in enumerate(masks):
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(mask, idx + 1))
