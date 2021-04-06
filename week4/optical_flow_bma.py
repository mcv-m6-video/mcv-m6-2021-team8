import time
from itertools import product

import cv2
import pandas as pd

from src.optical_flow.block_matching_flow import block_matching_flow
from src.optical_flow.utils import read_flow, evaluate_flow


if __name__ == '__main__':
    img_prev = cv2.imread('./data/data_stereo_flow/training/image_0/000045_10.png', cv2.IMREAD_GRAYSCALE)
    img_next = cv2.imread('./data/data_stereo_flow/training/image_0/000045_11.png', cv2.IMREAD_GRAYSCALE)
    flow_noc = read_flow('./data/data_stereo_flow/training/flow_noc/000045_10.png')

    motion_type = ['forward', 'backward']
    search_area = [16, 32, 64, 128]
    block_size = [4, 8, 16, 32]

    data = []
    for m, p, n in product(motion_type, search_area, block_size):
        tic = time.time()
        flow = block_matching_flow(img_prev, img_next, motion_type=m, search_area=p, block_size=n, algorithm='corr')
        toc = time.time()
        msen, pepn = evaluate_flow(flow_noc, flow)
        data.append([m, p, n, msen, pepn, toc - tic])
    df = pd.DataFrame(data, columns=['motion_type', 'search_area', 'block_size', 'msen', 'pepn', 'runtime'])
    print(df)
