import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import motmetrics as mm
from tqdm import tqdm

from evaluation.bbox_iou import bbox_iou
from utils.detection import Detection
from utils.plotting import visualize_tracks, visualize_tracks_opencv
from utils.track import Track


def intersection(u, v):
    """
    Compare histograms based on their intersection.

    Args:
        u (ndarray): 1D array of type np.float32 containing image descriptors.
        v (ndarray): 1D array of type np.float32 containing image descriptors.

    Returns:
        float: distance between histograms.
    """
    return 1 - cv2.compareHist(np.array(u), np.array(v), cv2.HISTCMP_INTERSECT)


def hsv_histogram(image):
    h, w, c = image.shape
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    sizes = [180, 256, 256]
    ranges = [(0, 180), (0, 256), (0, 256)]
    descriptors = []
    for i in range(c):
        hist = cv2.calcHist([hsv], [i], None, [sizes[i]], ranges[i]).ravel()
        hist = hist / (h * w)  # normalize
        descriptors.append(np.array(hist, dtype=np.float32))

    descriptors = descriptors/np.linalg.norm(descriptors)
    return np.array(descriptors)

def rgb_histogram(image):
    h, w, c = image.shape

    descriptors = []
    for i in range(c):
        hist = np.histogram(image[:, :, i], bins=256, range=(0, 255))[0]
        hist = hist / (h * w)  # normalize
        descriptors.append(np.array(hist, dtype=np.float32))

    return descriptors


def color_check(image, bbox_1, bbox_2):
    image_1 = image[int(bbox_1[1]):int(bbox_1[3]), int(bbox_1[0]):int(bbox_1[2]), :]
    image_2 = image[int(bbox_2[1]):int(bbox_2[3]), int(bbox_2[0]):int(bbox_2[2]), :]

    u = rgb_histogram(image_1)
    v = rgb_histogram(image_2)

    if intersection(u, v) < 0.3:
        return True
    return False


def ratio_check(bbox_1, bbox_2):
    box_h_1, box_w_1 = bbox_1[2] - bbox_1[0], bbox_1[3] - bbox_1[1]
    box_h_2, box_w_2 = bbox_2[2] - bbox_2[0], bbox_2[3] - bbox_2[1]
    if (box_h_1/box_w_1) > 1.2*(box_h_2/box_w_2) or (box_h_1/box_w_1) < 0.8*(box_h_2/box_w_2):
        return False
    return True


def obtain_new_tracks(tracks, unused_detections, max_track, frame_tracks):
    for detection in unused_detections:
        tracks.append(Track(max_track+1, [detection], 0, 1, 1))
        frame_tracks[max_track+1] = dict(bbox= detection.bbox, confidence= detection.confidence)

        max_track += 1

    return tracks, max_track, frame_tracks


def predict_position(image, track):
    width = 0
    height = 0
    xtl_new = 0
    ytl_new = 0
    time = track.time_since_update + 1
    for n, detection in enumerate(track.detections):
        width += detection.bbox[2] - detection.bbox[0]
        height += detection.bbox[3] - detection.bbox[1]
        if n>0:
            xtl_new += detection.bbox[0] - track.detections[n - 1].bbox[0]
            ytl_new += detection.bbox[1] - track.detections[n - 1].bbox[1]

    width = width/len(track.detections)
    height = height/len(track.detections)
    xtl_new = track.detections[-1].bbox[0] + time*xtl_new/len(track.detections)
    ytl_new = track.detections[-1].bbox[1] + time*ytl_new / len(track.detections)

    next_detection_bbox = [xtl_new, ytl_new, xtl_new + width, ytl_new + height]

    # if track.detections[-1].label == 'bike':
    #     fig, ax = plt.subplots()
    #     ax.imshow(image, cmap='gray')
    #     minc, minr, maxc, maxr = next_detection_bbox
    #     rect = mpatches.Rectangle((minc, minr), maxc - minc + 1, maxr - minr + 1, fill=False, edgecolor='red',
    #                                   linewidth=2)
    #     ax.add_patch(rect)
    #
    #     plt.show()

    return next_detection_bbox


def update_tracks(image, tracks, detections, frame_tracks):
    unused_detections = detections
    for track in tracks:
        if track.time_since_update < 20:
            if track.time_since_update > 0 and len(track.detections) > 1:
                last_bbox = predict_position(image, track)
            else:
                last_bbox = track.detections[-1].bbox
            match_detection = match_next_bbox(image, last_bbox, unused_detections)
            if match_detection is not None:
                unused_detections.remove(match_detection)
                track.detections.append(match_detection)
                frame_tracks[track.id] = dict(bbox=match_detection.bbox, confidence=match_detection.confidence)
                track.hits +=1
                if track.time_since_update == 0:
                    track.hit_streak += 1
                track.time_since_update = 0
            else:
                track.time_since_update += 1
                track.hit_streak = 0
    return tracks, unused_detections, frame_tracks


def match_next_bbox(image, last_bbox, unused_detections):
    highest_IoU = 0
    for detection in unused_detections:
        IoU = bbox_iou(last_bbox, detection.bbox)

        if IoU > highest_IoU and color_check(image, last_bbox, detection.bbox) and ratio_check(last_bbox, detection.bbox):
            highest_IoU = IoU
            best_match = detection

    if highest_IoU > 0:
        return best_match
    else:
        return None


def track_objects(video_path, detections_list, gt_list, display = False, export_frames = False, idf1 = True):

    colors = np.random.rand(500, 3)  # used only for display
    tracks = []
    max_track = 0
    new_detections = []

    if idf1:
        acc = mm.MOTAccumulator(auto_id=True)

    capture = cv2.VideoCapture(video_path)
    n_frame = 0
    pbar = tqdm(total=2140)

    while capture.isOpened():
        valid, image = capture.read()
        if not valid:
            break
        frame_tracks = {}

        detections_on_frame = [x for x in detections_list if x.frame == n_frame]
        gt_on_frame = [x for x in gt_list if x.frame == n_frame]

        tracks, unused_detections, frame_tracks = update_tracks(image, tracks, detections_on_frame, frame_tracks)
        tracks, max_track, frame_tracks = obtain_new_tracks(tracks, unused_detections, max_track, frame_tracks)

        if display and n_frame%2==0 and n_frame < 200:
            visualize_tracks(image, frame_tracks, colors, display=display)

        if export_frames:
            visualize_tracks_opencv(image, frame_tracks, colors, export_frames=export_frames,
                             export_path="output_frames/tracking/frame_{:04d}.png".format(n_frame))

        # IDF1 computing
        detec_bboxes = []
        detec_ids = []
        for key, value in frame_tracks.items():
            detec_ids.append(key)
            bbox = value['bbox']
            conf = value['confidence']
            detec_bboxes.append(bbox)
            new_detections.append(Detection(n_frame, 'car', bbox[0], bbox[1], bbox[2] - bbox[0],
                                            bbox[3] - bbox[1], conf, track_id=key,
                                            histogram=rgb_histogram(image[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2]), :])))


        gt_bboxes = []
        gt_ids = []
        for gt in gt_on_frame:
            gt_bboxes.append(gt.bbox)
            gt_ids.append(gt.track_id)

        mm_gt_bboxes = [[(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2, bbox[2]-bbox[0], bbox[3]-bbox[1]] for bbox in gt_bboxes]
        mm_detec_bboxes = [[(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2, bbox[2] - bbox[0], bbox[3] - bbox[1]] for bbox in detec_bboxes]

        distances_gt_det = mm.distances.iou_matrix(mm_gt_bboxes, mm_detec_bboxes, max_iou=1.)
        if idf1:
            acc.update(gt_ids, detec_ids, distances_gt_det)

        pbar.update(1)
        n_frame += 1

    pbar.close()
    capture.release()
    cv2.destroyAllWindows()

    if idf1:
        print(acc.mot_events)
        mh = mm.metrics.create()
        summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name='acc')
        print(summary)

    return new_detections
