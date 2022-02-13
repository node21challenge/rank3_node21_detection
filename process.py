
import SimpleITK
import numpy as np
import json
from typing import Dict
import glob

from pandas import DataFrame
from scipy.ndimage import center_of_mass, label
from medpy.io import load, header

from evalutils import DetectionAlgorithm
from postprocessing import NMS, preds_sort, bagging, retina_bags
from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)

import logging
import os
from collections import OrderedDict
from pathlib import Path
import torch
import itertools

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from process_launchers import *
from detectron2.modeling import GeneralizedRCNNWithTTA
from tools.train_net import build_evaluator, Trainer
from coco_json import NODE_2021
from coco_json_test import NODE_2021_TEST
from pycocotools.coco import COCO


class Maskrcnnnodecontainer(DetectionAlgorithm):
    def __init__(self, input_dir, output_dir, args=None):
        super().__init__(
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
            input_path=Path(input_dir),
            output_file=Path(os.path.join(output_dir, 'nodules.json'))
        )

        self.input_path, self.output_file = input_dir, output_dir
        self.coco_json_output_path = 'datasets'
        self.args_test = args

    def train_launcher_maskr(self, args):
        NODE_2021(self.input_path, self.coco_json_output_path)
        launch(
            train_main_maskr,
            args.num_gpus,
            num_machines=args.num_machines,
            machine_rank=args.machine_rank,
            dist_url=args.dist_url,
            args=(args,),
        )

    def train_launcher_retina(self, args):
        NODE_2021(self.input_path, self.coco_json_output_path)
        launch(
            train_main_retina,
            args.num_gpus,
            num_machines=args.num_machines,
            machine_rank=args.machine_rank,
            dist_url=args.dist_url,
            args=(args,),
        )

    def predict(self, *, input_image: SimpleITK.Image) -> DataFrame:
        image_data = SimpleITK.GetArrayFromImage(input_image)
        spacing = input_image.GetSpacing()
        image_data = np.array(image_data)
        if len(image_data.shape) == 2:
            image_data = np.expand_dims(image_data, 0)

        NODE_2021_TEST(image_data, self.coco_json_output_path)

        args = self.args_test
        ##  ensemble predictions
        # 1.1 retina win 100
        launch(
            retina_test_main_100,
            args.num_gpus,
            num_machines=args.num_machines,
            machine_rank=args.machine_rank,
            dist_url=args.dist_url,
            args=(args,),
        )
        # 1.2 retina win 995
        launch(
            retina_test_main_995,
            args.num_gpus,
            num_machines=args.num_machines,
            machine_rank=args.machine_rank,
            dist_url=args.dist_url,
            args=(args,),
        )
        # 1.3 retina win 99
        launch(
            retina_test_main_99,
            args.num_gpus,
            num_machines=args.num_machines,
            machine_rank=args.machine_rank,
            dist_url=args.dist_url,
            args=(args,),
        )
        # 1.x ensemble retina
        folders = [
            './retina_outputs_100',
            './retina_outputs_995',
            './retina_outputs_99', ]
        pred_paths = []
        for folder in folders:
            pred_paths.append(os.path.join(folder, 'inference/coco_instances_results.json'))
        pred_jsons = []
        for pred_path in pred_paths:
            with open(pred_path) as f:
                pred_jsons.append(json.load(f))
        final_jsons = [retina_bags(pred_jsons)]

        # 2.1. maskr win 100
        launch(
            maskr_test_main_100,
            args.num_gpus,
            num_machines=args.num_machines,
            machine_rank=args.machine_rank,
            dist_url=args.dist_url,
            args=(args,),
        )
        # 2.2. maskr win 995
        launch(
            maskr_test_main_995,
            args.num_gpus,
            num_machines=args.num_machines,
            machine_rank=args.machine_rank,
            dist_url=args.dist_url,
            args=(args,),
        )
        # 2.3. maskr win 99
        launch(
            maskr_test_main_99,
            args.num_gpus,
            num_machines=args.num_machines,
            machine_rank=args.machine_rank,
            dist_url=args.dist_url,
            args=(args,),
        )

        # retina + maskrcnn ensemble
        folders = [
            './maskr_outputs_100',
            './maskr_outputs_995',
            './maskr_outputs_99',]
        pred_paths = []
        for folder in folders:
            pred_paths.append(os.path.join(folder, 'inference/coco_instances_results.json'))
        # pred_jsons = []
        for pred_path in pred_paths:
            with open(pred_path) as f:
                final_jsons.append(json.load(f))
        pred_dicts = preds_sort(final_jsons)
        coco_inferences = bagging(pred_dicts, nms_th=0.2)
        print(coco_inferences)

        # post aligning prediction results
        coco_inferences_dict = {}
        for ann in coco_inferences:
            x, y, w, h = ann['bbox']
            score = ann['score']
            key = ann["image_id"]
            # area = w * h
            # # post smoothing to reduce false postive rate
            # if area < area_range[0] and score < 0.5:
            #     pass
            # if area > area_range[1] and score < 0.5:
            #     pass
                # 3 abs score supress
            # else:
                # if score >= score_th:
            if key in coco_inferences_dict.keys():
                coco_inferences_dict[key]['boxes'].append([x, y, x + w, y + h])
                coco_inferences_dict[key]['scores'].append(score)
            else:
                nest_dict = {}
                nest_dict['boxes'] = [[x, y, x + w, y + h]]
                nest_dict['scores'] = [score]
                coco_inferences_dict[key] = nest_dict

        results = []
        for i in range(len(image_data)):
            if i in coco_inferences_dict.keys():
                prediction = coco_inferences_dict[i]
                np_prediction = {str(key): [np.array(i) for i in val]
                                for key, val in prediction.items()}
                np_prediction['slice'] = len(np_prediction['boxes']) * [i]
                results.append(np_prediction)

        predictions = self.merge_dict(results)
        data = self.format_to_GC(predictions, spacing)
        print(data)
        return data

    def format_to_GC(self, np_prediction, spacing) -> Dict:
        '''
        Convenient function returns detection prediction in required grand-challenge format.
        See:
        https://comic.github.io/grandchallenge.org/components.html#grandchallenge.components.models.InterfaceKind.interface_type_annotation


        np_prediction: dictionary with keys boxes and scores.
        np_prediction[boxes] holds coordinates in the format as x1,y1,x2,y2
        spacing :  pixel spacing for x and y coordinates.

        return:
        a Dict in line with grand-challenge.org format.
        '''
        # For the test set, we expect the coordinates in millimeters.
        # this transformation ensures that the pixel coordinates are transformed to mm.
        # and boxes coordinates saved according to grand challenge ordering.
        x_y_spacing = [spacing[0], spacing[1], spacing[0], spacing[1]]
        boxes = []
        for i, bb in enumerate(np_prediction['boxes']):
            box = {}
            box['corners'] = []
            x_min, y_min, x_max, y_max = bb * x_y_spacing
            x_min, y_min, x_max, y_max = round(x_min, 2), round(y_min, 2), round(x_max, 2), round(y_max, 2)
            bottom_left = [x_min, y_min, np_prediction['slice'][i]]
            bottom_right = [x_max, y_min, np_prediction['slice'][i]]
            top_left = [x_min, y_max, np_prediction['slice'][i]]
            top_right = [x_max, y_max, np_prediction['slice'][i]]
            box['corners'].extend([top_right, top_left, bottom_left, bottom_right])
            box['probability'] = round(float(np_prediction['scores'][i]), 2)
            boxes.append(box)

        return dict(type="Multiple 2D bounding boxes", boxes=boxes, version={"major": 1, "minor": 0})

    def merge_dict(self, results):
        merged_d = {}
        for k in results[0].keys():
            merged_d[k] = list(itertools.chain(*[d[k] for d in results]))
        return merged_d

    def save(self):
        with open(str(self._output_file), "w") as f:
            json.dump(self._case_results[0], f)

    # TODO: Copy this function for your processor as well!
    def process_case(self, *, idx, case):
        '''
        Read the input, perform model prediction and return the results.
        The returned value will be saved as nodules.json by evalutils.
        process_case method of evalutils
        (https://github.com/comic/evalutils/blob/fd791e0f1715d78b3766ac613371c447607e411d/evalutils/evalutils.py#L225)
        is overwritten here, so that it directly returns the predictions without changing the format.

        '''
        # Load and test the image for this case
        input_image, input_image_file_path = self._load_input_image(case=case)
        print(input_image_file_path)
        # Detect and score candidates
        scored_candidates = self.predict(input_image=input_image)

        # Write resulting candidates to nodules.json for this case
        return scored_candidates


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)

    if args.train_maskr:
        Maskrcnnnodecontainer(args.input_dir, args.output_dir).train_launcher_maskr(args)
    if args.train_retina:
        Maskrcnnnodecontainer(args.input_dir, args.output_dir).train_launcher_retina(args)
    # test mode
    else:
        Maskrcnnnodecontainer(args.input_dir, args.output_dir, args=args).process()
