import os

import argparse
import numpy as np
from pathlib import Path

import torch
from test_lib import test
from utils.general import check_file
from PIL import Image
from resnet_classifier.infer import RoedeerInference
import torchvision.transforms as T

inference = RoedeerInference("resnet_classifier/best.model")
conf_tresh_for_two_stage = 0.5

def second_stage_classifier(img_batch, predictions, conf_thres, paths):

    index = 0
    for img, prediction, path in zip(img_batch, predictions, paths):
        n_predictions, _ = prediction.shape

        month_path = path.replace("images", "months").replace("JPG", "month")
        with open(month_path, 'r') as f:
            month = int(f.read())


        _, h_img, w_img = img.shape
        for i in range(0, n_predictions):
            if prediction[i, 4] < conf_tresh_for_two_stage:
                break

            # get bounding box corners
            x0, y0, x1, y1, _, _ = prediction[i, :]
            x0, y0, x1, y1 = [int(e.item()) for e in [x0, y0, x1, y1]]

            # get bounding box width and height
            w_bb = x1 - x0
            h_bb = y1 - y0

            # get center coordinate
            xc = x0 + w_bb // 2
            yc = y0 + h_bb // 2

            # compute coordinates with 10% margin
            r = max((w_bb // 2) * 1.1, (h_bb // 2) * 1.1)
            x0 = xc - r
            x1 = xc + r
            y0 = yc - r
            y1 = yc + r
            x0 = int(max(x0, 0))
            y0 = int(max(y0, 0))
            x1 = int(min(x1, w_img))
            y1 = int(min(y1, h_img))
           
            # Transform to image
            cropped = img[:,y0:y1,x0:x1]
            pil_img = T.ToPILImage()(cropped)

            # run second stage inference
            identifyable_prediction, confidence = inference.infer(pil_img, month)

            # write inference result
            if identifyable_prediction == 0:
                prediction[i, 4] = torch.from_numpy(np.array([confidence])).to(prediction)
                prediction[i, 5] = torch.from_numpy(np.array([2.0])).to(prediction)

    return predictions
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--data', default="data/roedeer.yaml", help='*.data path')
    parser.add_argument('--batch-size', type=int, default=1, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=714, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--conf-thres-second-stage', type=float, default=0.5, help='object confidence threshold for applying second stage')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='IOU threshold for NMS')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='roedeer-exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--v5-metric', action='store_true', help='assume maximum recall as 1.0 in AP calculation')
    opt = parser.parse_args()
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.data = check_file(opt.data)  # check file
    print(opt)
    #check_requirements()

test(opt, 
     opt.data,
     "weights/roedeer_t4.pt", #weights
     opt.batch_size,
     opt.img_size,
     opt.conf_thres,
     opt.iou_thres,
     opt.save_json,
     opt.single_cls,
     opt.augment,
     opt.verbose,
     save_txt=opt.save_txt | opt.save_hybrid,
     save_hybrid=opt.save_hybrid,
     save_conf=opt.save_conf,
     trace=not opt.no_trace,
     v5_metric=opt.v5_metric,
     second_stage_classifier=second_stage_classifier
     )
