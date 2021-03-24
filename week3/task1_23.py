import os
from collections import defaultdict, OrderedDict
import xml.etree.ElementTree as ET
import numpy as np
import cv2

import torch
import torch.utils.data
import torchvision
from utils.engine import train_one_epoch, evaluate

from sklearn.model_selection import KFold

MODELS = {
    'fasterrcnn': torchvision.models.detection.fasterrcnn_resnet50_fpn,
    'maskrcnn': torchvision.models.detection.fasterrcnn_resnet50_fpn,
    'retina': torchvision.models.detection.retinanet_resnet50_fpn,
}


def parse_annotations(ann_file):
    tree = ET.parse(ann_file)
    root = tree.getroot()

    annotations = defaultdict(list)
    for track in root.findall('track'):
        if track.attrib['label'] == 'car':
            for box in track.findall('box'):
                frame = int(box.attrib['frame'])
                xtl = float(box.attrib['xtl'])
                ytl = float(box.attrib['ytl'])
                xbr = float(box.attrib['xbr'])
                ybr = float(box.attrib['ybr'])
                annotations[frame].append([xtl, ytl, xbr, ybr])

    return OrderedDict(annotations)


class AICityDataset(torch.utils.data.Dataset):
    def __init__(self, video_file, ann_file, transform=None):
        self.video_file = video_file
        self.ann_file = ann_file
        self.transform = transform

        self.boxes = parse_annotations(self.ann_file)
        self.video_cap = cv2.VideoCapture(self.video_file)

        self.classes = ['background', 'person', 'bicycle', 'car']
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}

    def __getitem__(self, idx):
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, img = self.video_cap.read()

        if self.transform is not None:
            img = self.transform(img)

        boxes = torch.as_tensor(self.boxes.get(idx, []), dtype=torch.float32)
        labels = torch.full((len(boxes),), self.class_to_idx['car'], dtype=torch.int64)
        image_id = torch.tensor([idx])

        target = {'boxes': boxes, 'labels': labels, 'image_id': image_id}

        return img, target

    def __len__(self):
        return int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))


def get_model(model_name, fine_tune=True, num_classes=4):
    model = MODELS[model_name](pretrained=True)
    if fine_tune:
        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features,
                                                                                                   num_classes)

    model.roi_heads.mask_roi_pool = None
    model.roi_heads.mask_head = None
    model.roi_heads.mask_predictor = None

    return model


if __name__ == '__main__':
    np.random.seed(42)
    model_name = 'fasterrcnn'
    root = './drive/MyDrive'
    save_path = './results/week3/detection_fasterrcnn_finetuning.txt'

    k_folds = 4
    num_epochs = 1
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    results = {}

    # dataset
    transform = torchvision.transforms.ToTensor()
    dataset = AICityDataset(video_file=os.path.join(root, 'vdo.avi'),
                            ann_file=os.path.join(root, 'ai_challenge_s03_c010-full_annotation.xml'),
                            transform=transform)

    # Define the K-fold Cross Validator
    k_folds = KFold(n_splits=k_folds, shuffle=False)
    for fold, (train_ids, test_ids) in enumerate(k_folds.split(dataset)):
        print(f'FOLD {fold}')
        print('--------------------------------')
        train_sampler = torch.utils.data.SubsetRandomSampler(test_ids)
        test_sampler = torch.utils.data.SubsetRandomSampler(train_ids)
        # define data loaders
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=4, sampler=train_sampler, num_workers=2,
                                                   collate_fn=utils.collate_fn)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=4, sampler=test_sampler, num_workers=2,
                                                  collate_fn=utils.collate_fn)

        # Create model
        model = get_model(model_name, fine_tune=True, num_classes=len(train_loader.dataset.classes))
        model.to(device)

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

        for epoch in range(num_epochs):
            train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)
            lr_scheduler.step()
            evaluate(model, test_loader, device, save_path)
        results[fold] = map

    # Print fold results
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    print('--------------------------------')
    sum_folds = 0.0
    for key, value in results.items():
        print(f'Fold {key}: {value} %')
        sum_folds += value
    print(f'Average: {sum_folds / len(results.items())} %')


