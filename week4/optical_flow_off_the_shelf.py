import os

import imageio
from tqdm import tqdm

import numpy as np
import cv2

from src.utils.aicity_reader import AICityChallengeAnnotationReader
from src.tracking.tracking import update_tracks_by_overlap
from src.evaluation.average_precision import mean_average_precision
from src.evaluation.idf1 import MOTAcumulator


if __name__ == '__main__':
    # Tracking with optical flow

    cap = cv2.VideoCapture('data/AICity_data/train/S03/c010/vdo.avi')
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    save_path = 'results/week4/task_31'
    os.makedirs(save_path, exist_ok=True)

    reader = AICityChallengeAnnotationReader(path='data/ai_challenge_s03_c010-full_annotation.xml')
    gt = reader.get_annotations(classes=['car'])
    reader = AICityChallengeAnnotationReader(path='data/AICity_data/train/S03/c010/det/det_mask_rcnn.txt')
    dets = reader.get_annotations(classes=['car'])

    if save_path:
        writer = imageio.get_writer(os.path.join(save_path, f'task31.gif'), fps=fps)

    accumulator = MOTAcumulator()
    y_true = []
    y_pred = []
    y_pred_refined = []
    tracks = []
    max_track = 0
    previous_frame = None
    end = int(n_frames * video_percentage)
    for i, frame in tqdm(enumerate(dets.keys())):
        if i == end:
            break

        if save_path:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ret, img = cap.read()

        if i == 0:
            optical_flow = None
        else:
            height, width = previous_frame.shape[:2]

            # get points on which to detect the flow
            points = []
            for det in detections_on_frame:
                points.append([det.xtl, det.ytl])
                points.append([det.xbr, det.ybr])
            p0 = np.array(points, dtype=np.float32)

            # params for lucas-kanade optical flow
            lk_params = dict(winSize=(15, 15), maxLevel=2,
                             criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

            p1, st, err = cv2.calcOpticalFlowPyrLK(previous_frame, img, p0, None, **lk_params)

            p0 = p0.reshape((len(detections_on_frame) * 2, 2))
            p1 = p1.reshape((len(detections_on_frame) * 2, 2))
            st = st.reshape(len(detections_on_frame) * 2)

            # flow field computed by subtracting prev points from next points
            flow = p1 - p0
            flow[st == 0] = 0

            optical_flow = np.zeros((height, width, 2), dtype=np.float32)
            for jj, det in enumerate(detections_on_frame):
                optical_flow[int(det.ytl), int(det.xtl)] = flow[2 * jj]
                optical_flow[int(det.ybr), int(det.xbr)] = flow[2 * jj + 1]

        previous_frame = img.copy()

        detections_on_frame = dets.get(frame, [])
        tracks, frame_tracks, max_track = update_tracks_by_overlap(tracks,
                                                                   detections_on_frame,
                                                                   max_track,
                                                                   refinement=False,
                                                                   optical_flow=optical_flow)

        frame_detections = []
        for track in frame_tracks:
            det = track.last_detection()
            frame_detections.append(det)
            if save_path:
                cv2.rectangle(img, (int(det.xtl), int(det.ytl)), (int(det.xbr), int(det.ybr)), track.color, 2)
                cv2.rectangle(img, (int(det.xtl), int(det.ytl)), (int(det.xbr), int(det.ytl) - 15), track.color, -2)
                cv2.putText(img, str(det.id), (int(det.xtl), int(det.ytl)), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
                for dd in track.detections:
                    cv2.circle(img, dd.center, 5, track.color, -1)

        y_pred_refined.append(frame_detections)
        y_pred.append(detections_on_frame)
        y_true.append(gt.get(frame, []))

        accumulator.update(y_true[-1], y_pred_refined[-1])

        if save_path:
            writer.append_data(cv2.resize(img, (600, 350)))

    cv2.destroyAllWindows()
    if save_path:
        writer.close()

    ap, prec, rec = mean_average_precision(y_true, y_pred, classes=['car'], sort_method='score')
    print(f'Original AP: {ap:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}')
    ap, prec, rec = mean_average_precision(y_true, y_pred_refined, classes=['car'], sort_method='score')
    print(f'After refinement AP: {ap:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}')
    print('\nAdditional metrics:')
    print(accumulator.get_idf1())


