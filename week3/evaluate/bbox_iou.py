from copy import deepcopy
from statistics import mean

def bbox_iou(bboxA, bboxB):
    # compute the intersection over union of two bboxes

    # Format of the bboxes is [tly, tlx, bry, brx, ...], where tl and br
    # indicate top-left and bottom-right corners of the bbox respectively.

    # determine the coordinates of the intersection rectangle
    xA = max(bboxA[1], bboxB[1])
    yA = max(bboxA[0], bboxB[0])
    xB = min(bboxA[3], bboxB[3])
    yB = min(bboxA[2], bboxB[2])
    
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
    # compute the area of both bboxes
    bboxAArea = (bboxA[2] - bboxA[0] + 1) * (bboxA[3] - bboxA[1] + 1)
    bboxBArea = (bboxB[2] - bboxB[0] + 1) * (bboxB[3] - bboxB[1] + 1)
    
    iou = interArea / float(bboxAArea + bboxBArea - interArea)
    
    # return the intersection over union value
    return iou


def iou_list(groundtruth_list_original, detections_list, ini_num=500, stop_num=1000, IoU_threshold=0.5):
    groundtruth_list = deepcopy(groundtruth_list_original)

    # Sort detections by confidence
    detections_list.sort(key=lambda x: x.confidence, reverse=True)
    # Save number of groundtruth labels
    groundtruth_size = len(groundtruth_list)

    TP = 0;
    FP = 0;
    FN = 0
    precision = list();
    recall = list()

    # to compute mAP
    max_precision_per_step = list()
    threshold = 1;
    checkpoint = 0
    temp = 1000

    ious_list = []
    for n, detection in enumerate(detections_list):

        if n<ini_num:
            continue
        if n>stop_num:
            break
        match_flag = False
        if threshold != temp:
            # print(threshold)
            temp = threshold

        # Get groundtruth of the target frame
        gt_on_frame = [x for x in groundtruth_list if x.frame == detection.frame]
        gt_bboxes = [(o.bbox, o.confidence) for o in gt_on_frame]

        print(n)
        print(gt_bboxes)
        # iou = 0
        ious = []
        for gt_bbox in gt_bboxes:
            # print(gt_bbox[0])
            # print(detection.bbox)
            iou = bbox_iou(gt_bbox[0], detection.bbox)
            ious.append(iou)
            # if iou > IoU_threshold and gt_bbox[1] > 0.9:
            #     match_flag = True
            #     TP += 1
            #     gt_used = next((x for x in groundtruth_list if x.frame == detection.frame and x.bbox == gt_bbox[0]),
            #                    None)
            #     gt_used.confidence = 0
            #     break
        # ious.append(iou)
        if ious:
            if max(ious) > IoU_threshold:
                TP += 1
                ious_list.append(max(ious))
            else:
                ious_list.append(0.85)
        else:
            ious_list.append(0.85)

    print(ious_list)
    return ious_list