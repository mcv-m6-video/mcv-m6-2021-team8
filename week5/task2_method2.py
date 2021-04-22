import pickle
from utils.evaluation import compute_idf1

import cv2
from utils.multiple_camera import merge_tracks, sort_track, addTracksToFrames_multi_cam_gif


def crop_from_detection(det):
    image = cv2.imread(det['img_path'])
    box = [int(det['left']), int(det['top']), int(det['width']), int(det['height'])]
    cropped = image[box[1]:box[1]+box[3], box[0]:box[0]+box[2], :]
    return cropped


if __name__ == "__main__":
    frame_path_S3 = './data/AIC20_track3/train/S03/'
    camera_list = ['c010', 'c011', 'c012', 'c013', 'c014', 'c015']
    timestamp = {'c010': 8.715, 'c011': 8.457, 'c012': 5.879, 'c013': 0, 'c014': 5.042,'c015': 8.492}
    for key in timestamp.keys():
        timestamp[key] = int(timestamp[key])

    fps_ratio = {'c010': 1.0, 'c011': 1.0, 'c012': 1.0, 'c013': 1.0, 'c014': 1.0, 'c015': 10.0 / 8.0}

    video_length_list = {'c010': 2141, 'c011': 2279, 'c012': 2422, 'c013': 2415, 'c014': 2332, 'c015': 1928}

    offset = {'c010': [6000, 2000], 'c011': [6000, 0], 'c012': [3000, 0], 'c013': [0, 2000], 'c014': [3000, 2000], 'c015': [0, 0]}

    with open("detections_tracks_all_camera.pkl", 'rb') as f:
        detections_tracks_all_camera = pickle.load(f)
        f.close()
    with open("gt_tracks_all_camera.pkl", 'rb') as f:
        gt_tracks_all_camera = pickle.load(f)
        f.close()

    detections_tracks_all_camera_list = []
    idx = 0
    for cam in camera_list:
        sort_track(detections_tracks_all_camera[cam])
        for track_one in detections_tracks_all_camera[cam]:
            track_one.id = idx
            idx = idx + 1
            for detection in track_one.detections:
                detection['img_path'] = "{}{}/frames/{}.jpg".format(frame_path_S3, cam,
                                                                    str(detection['frame']).zfill(5))
                detection['cam'] = cam
                detection['frame'] = int(detection['frame'] * fps_ratio[cam] + timestamp[cam])

            detections_tracks_all_camera_list.append(track_one)

    gt_tracks_all_camera_list = []
    for cam in camera_list:
        sort_track(gt_tracks_all_camera[cam])
        for track_one in gt_tracks_all_camera[cam]:
            for detection in track_one.detections:
                detection['img_path'] = "{}{}/frames/{}.jpg".format(frame_path_S3, cam,
                                                                    str(detection['frame']).zfill(5))
                detection['cam'] = cam
                detection['frame'] = int(detection['frame'] * fps_ratio[cam] + timestamp[cam])
            gt_tracks_all_camera_list.append(track_one)

    compute_idf1(gt_tracks_all_camera_list, detections_tracks_all_camera_list, 2422)

    with open("nca.pkl", 'rb') as f:
        nca = pickle.load(f)
        f.close()

    with open("detections_tracks_all_camera_list.pkl", 'rb') as f:
        detections_tracks_all_camera_list = pickle.load(f)
        f.close()
    load_result_tracks = True

    new_detections_tracks = merge_tracks(detections_tracks_all_camera_list)

    gt_tracks_all_camera_list = []
    for cam in gt_tracks_all_camera.keys():
        for track_one in gt_tracks_all_camera[cam]:
            gt_tracks_all_camera_list.append(track_one)
    new_gt_tracks = merge_tracks(gt_tracks_all_camera_list)

    sort_track(new_detections_tracks)
    sort_track(new_gt_tracks)

    compute_idf1(gt_tracks_all_camera_list, detections_tracks_all_camera_list, 2422)

    addTracksToFrames_multi_cam_gif(frame_path_S3, new_detections_tracks, new_gt_tracks, offset, camera_list, timestamp, fps_ratio, video_length_list,
                                    start_frame=1000, end_frame=1180)
