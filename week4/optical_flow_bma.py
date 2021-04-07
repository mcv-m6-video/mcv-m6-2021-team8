import time
from itertools import product

import cv2
import pandas as pd

from block_matching_flow import block_matching_flow
from utils import read_flow, evaluate_flow


if __name__ == '__main__':
    img_prev = cv2.imread('./data/data_stereo_flow/training/image_0/000045_10.png', cv2.IMREAD_GRAYSCALE)
    img_next = cv2.imread('./data/data_stereo_flow/training/image_0/000045_11.png', cv2.IMREAD_GRAYSCALE)
    flow_noc = read_flow('./data/data_stereo_flow/training/flow_noc/000045_10.png')

    compensation_type = ['forward', 'backward']
    search_area = [32]
    block_size = [28]
    searching_algorithm = ['corr']  # ['exhaustive search', 'three step search']
    err_func = ['ssd']  # ['sum of squared differences', 'sum of absolute differences']

    results = []
    for idx, (p, n) in enumerate(product(search_area, block_size)):
        print(f'try combination {idx}: p = {p}, n = {n}')
        start = time.time()
        estimated_of = block_matching_flow(img_prev, img_next, motion_type='forward', search_area=p, block_size=n, metric='ssd', algorithm='corr')
        msen, pepn = evaluate_flow(flow_noc, estimated_of)
        runtime = time.time() - start
        results.append([p, n, msen, pepn, runtime])
        print(f'results for combination {idx}: msen = {msen}, pepn = {pepn}, runtime = {runtime}')
        optical_flow_arrow_plot(gray_frame, estimated_of, path='./')
        optical_flow_magnitude_plot(gray_frame, estimated_of, path='./')

    df = pd.DataFrame(results, columns=['search_area', 'block_size', 'msen', 'pepn', 'runtime'])
    print(results)

    with open('of_bma.pkl', 'wb') as f:
        pickle.dump(results, f)
