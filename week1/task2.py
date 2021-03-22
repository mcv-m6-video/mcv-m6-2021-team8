import utils
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import pandas as pd
import shutil
import time
import sklearn
import seaborn as sns; sns.set()


## Defining path and the arguments ##
detectionFileFolder = '/Users/siddhantbhambri/Desktop/m6/Datasets/AICity/train/S03/c010/det/'
detectionFileNames = ['det_mask_rcnn.txt', 'det_ssd512.txt', 'det_yolo3.txt']
detectorNames = ['Mask RCNN', 'SSD512', 'YOLO3']

#####################################

groundTruth = utils.read_annotations('/Users/siddhantbhambri/Desktop/m6/Datasets/AICity/aicity_annotations.xml', 2141)
groundTruthPerFrame = utils.getDetectionsPerFrame(groundTruth)
print("IoU")

#####################################

for detectorName, detectionFile in zip(detectorNames, detectionFileNames):
    detections = utils.getDetections(detectionFileFolder + detectionFile)
    sortedDetections = utils.sortDetectionsByKey(detections, 'confidence', decreasing=True)
    detectionsPerFrame = utils.getDetectionsPerFrame(detections)
    iousPerFrame = np.zeros(len(detectionsPerFrame.keys()))
    utils.calculate_mAP(groundTruth, detections, IoU_threshold=0.5)
    for frame in tqdm(detectionsPerFrame.keys()):
        detectionBboxes = [utils.get_bounding_box_drom_detection(detection) for detection in detectionsPerFrame[frame]] if frame in detectionsPerFrame.keys() else []
        gtBboxes = [utils.get_bounding_box_drom_detection(gtItem) for gtItem in groundTruthPerFrame[frame]] if frame in groundTruthPerFrame.keys() else []
        results = utils.single_frame_results(detectionBboxes, gtBboxes, 0.5)
        if results['true_pos'] > 0:
            tmpIous = []
            for det_bbox in detectionBboxes:
                tmpIous.append(np.max([utils.bb_iou(det_bbox, gt_bbox) for gt_bbox in gtBboxes]))
                finalIous = []
                for i, iou in enumerate(tmpIous):
                    if iou > 0.5:
                        finalIous.append(iou)
            iousPerFrame[frame-1] = np.mean(finalIous)
    plt.plot(iousPerFrame)
    plt.ylim((0, 1.0))
    plt.title('Results for ' + detectorName)
    plt.show()
    utils.add_bounding_boxes_to_frames('/Users/siddhantbhambri/Desktop/m6/Datasets/AICity/frames', detections, groundTruth, detectorName)

print("IoU GT")
detections = utils.add_noise_to_detections('/Users/siddhantbhambri/Desktop/m6/Datasets/AICity/aicity_annotations.xml', 2141)
detectionsPerFrame = utils.getDetectionsPerFrame(detections)
iousPerFrame = np.zeros(len(detectionsPerFrame.keys()))
for frame in tqdm(detectionsPerFrame.keys()):
    detectionBboxes = [utils.get_bounding_box_drom_detection(detection) for detection in detectionsPerFrame[frame]] if frame in detectionsPerFrame.keys() else []
    gtBboxes = [utils.get_bounding_box_drom_detection(gtItem) for gtItem in groundTruthPerFrame[frame]] if frame in groundTruthPerFrame.keys() else []
    results = utils.single_frame_results(detectionBboxes, gtBboxes, 0.5)
    if results['true_pos'] > 0:
        tmpIous = []
        for det_bbox in detectionBboxes:
            tmpIous.append(np.max([utils.bb_iou(det_bbox, gt_bbox) for gt_bbox in gtBboxes]))
            finalIous = []
            for i, iou in enumerate(tmpIous):
                if iou > 0.5:
                    finalIous.append(iou)
        iousPerFrame[frame - 1] = np.mean(tmpIous)
plt.plot(iousPerFrame)
plt.ylim((0, 1.0))
plt.title('GT with noise')
plt.show()

utils.add_bounding_boxes_to_frames('/Users/siddhantbhambri/Desktop/m6/Datasets/AICity/frames', detections, groundTruth, "ground_truth")
exit()
