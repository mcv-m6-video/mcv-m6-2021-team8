import os
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
from torchvision.models import detection
from torchvision.transforms import transforms

from utils.average_precision import mean_average_precision
from utils.aicity_reader import AICityChallengeAnnotationReader
from utils.detection import Detection
from utils.non_maximum_supression import get_nms

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


MODELS = {
    'fasterrcnn': detection.fasterrcnn_resnet50_fpn,
    'maskrcnn': detection.fasterrcnn_resnet50_fpn,
    'retina': detection.retinanet_resnet50_fpn,
}


def gif_save(ground_truths, dets, video_path, title='', save_path=None):
    frames = list(gt.keys())
    overlaps = []

    for frame in frames:
        boxes1 = [d.bbox for d in gt.get(frame)]
        boxes2 = [d.bbox for d in det.get(frame, [])]
        iou = mean_intersection_over_union(boxes1, boxes2)
        overlaps.append(iou)

    cap = cv2.VideoCapture(video_path)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    image = ax1.imshow(np.zeros((height, width)))
    artists = [image]

    def update(i):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frames[i])
        ret, img = cap.read()
        for d in gt[frames[i]]:
            cv2.rectangle(img, (int(d.xtl), int(d.ytl)), (int(d.xbr), int(d.ybr)), (0, 255, 0), 2)
        for d in det[frames[i]]:
            cv2.rectangle(img, (int(d.xtl), int(d.ytl)), (int(d.xbr), int(d.ybr)), (0, 0, 255), 2)
        artists[0].set_data(img[:, :, ::-1])
        return artists

    ani = animation.FuncAnimation(fig1, update, len(frames), interval=2, blit=True)
    ax1.axis('off')
    if save_path is not None:
        ani.save(os.path.join(save_path, 'video.gif'), writer=animation.PillowWriter())

    fig2, ax2 = plt.subplots()
    line, = plt.plot(frames, overlaps)
    artists = [line]

    def update1(i):
        artists[0].set_data(frames[:i + 1], overlaps[:i + 1])
        return artists

    ani = animation.FuncAnimation(fig2, update1, len(frames), interval=2, blit=True)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel('#frame')
    ax2.set_ylabel('mean IoU')
    fig2.suptitle(title)
    if save_path is not None:
        ani.save(os.path.join(save_path, 'iou.gif'), writer=animation.PillowWriter())

    with open(os.path.join(save_path, 'maskrcnn.pkl'), 'wb') as f:
        pickle.dump(overlaps, f)

    plt.plot(frames, overlaps)
    plt.xlabel('#frame')
    plt.ylabel('mIoU')
    plt.xticks(np.arange(0, len(frames), step=500))
    plt.yticks(np.arange(0, 1, step=0.2))
    plt.savefig('iou.png')


if __name__ == '__main__':
    model = 'fasterrcnn'
    start = 500
    length = 500
    save_path = './results/week3'
    model = MODELS[model](pretrained=True)

    # Load sequence and ground truth bbs
    cap = cv2.VideoCapture('./data/AICity_data/train/S03/c010/vdo.avi')
    if not length:
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    reader = AICityChallengeAnnotationReader(path='./data/ai_challenge_s03_c010-full_annotation.xml')
    gt = reader.get_annotations(classes=['car'])
    gt = {frame: gt[frame] for frame in range(start, start + length)}

    # Inference
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
    model.to(device)
    model.eval()

    detections = {}
    y_gt = []
    y_pred = []
    with torch.no_grad():
        for frame in range(start, length):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ret, img = cap.read()

            x = [transforms.ToTensor()(img).to(device)]
            preds = model(x)[0]

            # filter car predictions and confidences
            joint_preds = list(zip(preds['labels'], preds['boxes'], preds['scores']))
            car_det = list(filter(lambda x: x[0] == 3, joint_preds))
            # car_det = list(filter(lambda x: x[2] > 0.70, car_det))
            car_det = get_nms(car_det, 0.7)

            # add detections
            detections[frame] = []
            for det in car_det:
                det_obj = Detection(frame=frame, id=None, label='car',
                                    xtl=float(det[1][0]), ytl=float(det[1][1]), xbr=float(det[1][2]), ybr=float(det[1][3]), score=det[2])
                detections[frame].append(det_obj)
            y_gt.append(gt.get(frame, []))
            y_pred.append(detections[frame])

    ap, prec, rec = mean_average_precision(y_gt, y_pred, classes=['car'])
    print(f'Model: {model}, AP: {ap:.4f}')
    print(f'Saving result to {save_path}')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    gif_save(gt, detections, video_path='./data/AICity_data/train/S03/c010/vdo.avi', title=f'{model} detections', save_path=save_path)

    cv2.destroyAllWindows()
    print('Finished.')
