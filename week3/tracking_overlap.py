import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

from evaluate.bbox_iou import iou_list
from evaluate.evaluation_funcs import compute_mAP, compute_mAP_track
from utils.plotting import draw_video_bboxes
from utils.reading import read_annotations_file, read_homography_matrix
from tracking import track_objects




# Groundtruth
video_path = "datasets/AICity_data/train/S03/c010/vdo.avi"
groundtruth_xml_path = "annotations/m6-full_annotation.xml"
groundtruth_path = "datasets/AICity_data/train/S03/c010/gt/gt.txt"

# Given detections
detections_path = "datasets/AICity_data/train/S03/c010/det/"
detectors = ["det_ssd512.txt", "det_mask_rcnn.txt", "det_yolo3.txt"]
roi_path = 'datasets/AICity_data/train/S03/c010/roi.jpg'

# three offshelf detections
mask_detections_path = "datasets/AICity_data/train/S03/c010/det/det_mask_rcnn.txt"
yolo_detections_path = "datasets/AICity_data/train/S03/c010/det/det_yolo3.txt"
ssd_detections_path  = "datasets/AICity_data/train/S03/c010/det/det_ssd512.txt"

yolo_fine_tuned_detections_path = "annotations/yolo_fine_tunning_detections.txt"

if __name__ == '__main__':
    # Flags
    use_pkl = True
    display_frames = False
    export_frames = False

    # Load/Read groundtruth
    print("Getting groundtruth")
    if use_pkl and os.path.exists('groundtruth.pkl'):
        with open('groundtruth.pkl', 'rb') as p:
            print("Reading detections from groundtruth.pkl")
            groundtruth_list, tracks_gt_list = pickle.load(p)
    else:
        groundtruth_list, tracks_gt_list = read_annotations_file(groundtruth_xml_path, video_path)
        with open('groundtruth.pkl', 'wb') as f:
            pickle.dump([groundtruth_list, tracks_gt_list], f)



        # Task 2.1: Tracking by Overlap
    print("\nComputing tracking by overlap")
    detected_tracks = track_objects(video_path, mask_detections_list, groundtruth_list, display=display_frames, export_frames=export_frames)

    # Compute mAP
    compute_mAP(groundtruth_list, detected_tracks)


