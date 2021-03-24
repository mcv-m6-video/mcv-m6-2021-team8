# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import os

import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import trange
import imageio

from utils.aicity_reader import AICityChallengeAnnotationReader
from utils.processing import postprocess, bounding_boxes
from utils.average_precision import mean_average_precision


class SingleGaussianBackgroundModel:
    def __init__(self, video_path, color_space='gray', channels=None, resize=None):
        self.cap = cv2.VideoCapture(video_path)
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.color_space = color_space
        self.channels = channels
        self.resize = resize

        if self.resize is not None:
            self.height = int(self.height * self.resize)
            self.width = int(self.width * self.resize)

    def fit(self, start=0, length=None):
        if length is None:
            length = self.length
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        # Welford's online variance algorithm
        # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
        count = 0
        num_channels = len(self.channels)
        mean = np.zeros((self.height, self.width, num_channels))
        M2 = np.zeros((self.height, self.width, num_channels))
        for _ in trange(length, desc='modelling background'):
            img = self._read_and_preprocess()
            count += 1
            delta = img - mean
            mean += delta / count
            delta2 = img - mean
            M2 += delta * delta2
        self.mean = mean
        self.std = np.sqrt(M2 / count)

    def evaluate(self, frame, alpha=2.5, rho=0.01, only_update_bg=True):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        img = self._read_and_preprocess()

        # segment foreground
        fg = np.bitwise_and.reduce(np.abs(img - self.mean) >= alpha * (self.std + 2), axis=2)
        bg = ~fg

        # update background model
        if rho > 0:
            if only_update_bg:
                self.mean[bg, :] = rho * img[bg, :] + (1-rho) * self.mean[bg, :]
                self.std[bg, :] = np.sqrt(rho * np.power(img[bg, :] - self.mean[bg, :], 2) + (1-rho) * np.power(self.std[bg, :], 2))
            else:
                self.mean = rho * img + (1-rho) * self.mean
                self.std = np.sqrt(rho * np.power(img - self.mean, 2) + (1-rho) * np.power(self.std, 2))

        return img, (fg * 255).astype(np.uint8), self.mean.astype(np.uint8)

    def _read_and_preprocess(self):
        ret, img = self.cap.read()
        if self.resize is not None:
            img = cv2.resize(img, None, fx=self.resize, fy=self.resize, interpolation=cv2.INTER_CUBIC)
        return img


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    adaptive = False
    random_search = False
    model_frac = 0.25

    # bounding box restriction
    min_width = 120
    max_width = 800
    min_height = 100
    max_height = 600

    save_path = './results/week2'

    reader = AICityChallengeAnnotationReader(path='./data/AICity_data/train/S03/c010/gt/gt.txt')
    gt = reader.get_annotations(classes=['car'], only_not_parked=True)

    roi = cv2.imread('./data/AICity_data/train/S03/c010/roi.jpg', cv2.IMREAD_GRAYSCALE)

    bg_model = SingleGaussianBackgroundModel(video_path='./data/AICity_data/train/S03/c010/vdo.avi')
    bg_model.fit(start=0, length=int(bg_model.length * model_frac))

    start = int(bg_model.length * model_frac)
    end = bg_model.length

    # hyper-parameter search
    if random_search:
        alphas = np.random.choice(np.linspace(3, 8, 50), 25)
        rhos = np.random.choice(np.linspace(0.001, 0.1, 50), 25) if adaptive else [0]
        combinations = [(alpha, rho) for alpha, rho in zip(alphas, rhos)]
    else:
        alphas = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10]
        rhos = [0.005, 0.01, 0.025, 0.05, 0.1] if adaptive else [0]
        combinations = [(alpha, rho) for alpha in alphas for rho in rhos]
    mAPs = []

    for alpha, rho in combinations:
        if save_path:
            writer = imageio.get_writer(os.path.join(save_path, f'task1_2_alpha{alpha:.1f}_rho{rho:.3f}.gif'), fps=10)

        y_true = []
        y_pred = []
        for frame in trange(start, end, desc='evaluating frames'):
            _, mask, _ = bg_model.evaluate(frame=frame, alpha=alpha, rho=rho)
            mask = mask & roi
            mask = postprocess(mask)
            detections = bounding_boxes(mask, min_height, max_height, min_width, max_width, frame)

            annotations = gt.get(frame, [])

            if save_path:
                img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                for det in detections:
                    cv2.rectangle(img, (det.xtl, det.ytl), (det.xbr, det.ybr), (0, 255, 0), 2)
                for det in annotations:
                    cv2.rectangle(img, (int(det.xtl), int(det.ytl)), (int(det.xbr), int(det.ybr)), (255, 0, 0), 2)

                if save_path:
                    writer.append_data(img)

                cv2.imshow('result', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            y_pred.append(detections)
            y_true.append(annotations)

        cv2.destroyAllWindows()

        if save_path and (alpha == 1 or alpha == 2 or alpha == 4):
            writer.close()

        ap, prec, rec = mean_average_precision(y_true, y_pred, classes=['car'])
        mAPs.append(ap)

        print(f'alpha: {alpha:.1f}, rho: {rho:.3f}, AP: {ap:.4f}')
    print(f'mAPs: {combinations}')




