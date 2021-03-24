# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import os

import numpy as np
import matplotlib.pyplot as plt

from utils.aicity_reader import AICityChallengeAnnotationReader
from utils.average_precision import mean_average_precision


def compute(data_reader, gts, vals, prob=None, std=None):
    result = []
    for i, val in enumerate(vals):
        if i != len(vals) - 1:
            noises = {'drop': val if prob is None else prob, 'mean': 0, 'std': val if std is None else std}
            noised_gt = data_reader.get_annotations(classes=['car'], noise_params=noises)

            y_gt = []
            y_pred = []
            for frame in gts.keys():
                y_gt.append(gts.get(frame))
                y_pred.append(noised_gt.get(frame, []))

            mAp, _, _ = mean_average_precision(y_gt, y_pred)
            result.append(mAp)

    return result


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    reader = AICityChallengeAnnotationReader(path='./data/ai_challenge_s03_c010-full_annotation.xml')
    gt = reader.get_annotations(classes=['car'])

    # drop out bounding boxes at different rates
    probs = np.linspace(0, 1, 11)
    prob_maps = compute(reader, gt, probs, std=0)

    plt.plot(probs[:-1], prob_maps)
    plt.ylabel('mAP')
    plt.xlabel('Probability to remove bounding boxes')
    plt.title('Introduce probability to delete bounding boxes')
    plt.show()
    plt.savefig(os.path.join('./', 'map_drop_bbs.png'))

    # add noise to the bounding boxes
    stds = np.linspace(0, 100, 11)
    std_maps = compute(reader, gt, stds, prob=0)

    plt.plot(stds[:-1], std_maps)
    plt.xticks(stds)
    plt.xlabel('Standard deviation box changes')
    plt.ylabel('mAP')
    plt.title('Introduce probability to delete bounding boxes')
    plt.show()
    plt.savefig(os.path.join('./', 'map_std_bbs.png'))

    # try different detectors
    provided_detections = ['mask_rcnn', 'ssd512', 'yolo3']
    for det in provided_detections:
        reader_with_det = AICityChallengeAnnotationReader(path=f'./data/AICity_data/train/S03/c010/det/det_{det}.txt')
        gt_with_det = reader.get_annotations(classes=['car'])

        y_gt = []
        y_pred = []
        for frame in gt.keys():
            y_gt.append(gt.get(frame))
            y_pred.append(gt_with_det.get(frame, []))

        mAP, _, _ = mean_average_precision(y_gt, y_pred)
        print(f'mAP for {det} = {mAP:.4f}')



