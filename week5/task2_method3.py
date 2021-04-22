import os
from utils.reid import reid, write_results

from utils.idf1 import MOTAcumulator
from utils.aicity_reader import group_by_frame, parse_annotations_from_txt


if __name__ == '__main__':
    root = './data/aic19-track1-mtmc-train'
    seq = 'S03'
    model_path = './data/metric_learning/runs/epoch_19__ckpt.pth'
    reid_method = 'exhaustive'  # ['exhaustive']

    # obtain reid results
    path_results = os.path.join('results', 'week5', seq)
    results = reid(root, seq, model_path, reid_method)
    write_results(results, path=path_results)

    # compute metrics
    accumulator = MOTAcumulator()
    for cam in os.listdir(os.path.join(root, 'train', seq)):
        dets_true = group_by_frame(parse_annotations_from_txt(os.path.join(root, 'train', seq, cam, 'gt', 'gt.txt')))
        dets_pred = group_by_frame(parse_annotations_from_txt(os.path.join(path_results, cam, 'results.txt')))
        for frame in dets_true.keys():
            y_true = dets_true.get(frame, [])
            y_pred = dets_pred.get(frame, [])
            accumulator.update(y_true, y_pred)
    print(f'Metrics: {accumulator.get_metrics()}')