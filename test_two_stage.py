import os

import argparse
import numpy as np
from pathlib import Path

from yolov7.test_lib import test
from yolov7.utils.general import check_file

def second_stage_classifier(img, predictions):
    print(img, img.shape)
    print(predicitions)
    return predicitions
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--data', default="yolov7/data/roedeer.yaml", help='*.data path')
    parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=714, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
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
     "yolov7/weights/roedeer_t4.pt", #weights
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
