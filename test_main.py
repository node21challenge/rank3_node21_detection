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
from detectron2.modeling import GeneralizedRCNNWithTTA
from tools.train_net import build_evaluator, Trainer, maskr_setup, retina_setup


def maskr_test_main_100(args):
    cfg = maskr_setup(args)
    cfg.defrost()
    cfg.INPUT.WINDOW = (0, 1)
    cfg.MODEL.WEIGHTS = 'maskr_final_100.pth'
    cfg.OUTPUT_DIR = './maskr_outputs_100'
    cfg.freeze()

    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=args.output_dir).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=args.retrain)

    res = Trainer.test(cfg, model)
    if cfg.TEST.AUG.ENABLED:
        res.update(Trainer.test_with_TTA(cfg, model))
    if comm.is_main_process():
        verify_results(cfg, res)
        return res

def maskr_test_main_995(args):
    cfg = maskr_setup(args)
    cfg.defrost()
    cfg.INPUT.WINDOW = (0.005, 0.995)
    cfg.MODEL.WEIGHTS = 'maskr_final_995.pth'
    cfg.OUTPUT_DIR = './maskr_outputs_995'
    cfg.freeze()

    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=args.output_dir).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=args.retrain)

    res = Trainer.test(cfg, model)
    if cfg.TEST.AUG.ENABLED:
        res.update(Trainer.test_with_TTA(cfg, model))
    if comm.is_main_process():
        verify_results(cfg, res)
        return res

def maskr_test_main_99(args):
    cfg = maskr_setup(args)
    cfg.defrost()
    cfg.INPUT.WINDOW = (0.01, 0.99)
    cfg.MODEL.WEIGHTS = 'maskr_final_99.pth'
    cfg.OUTPUT_DIR = './maskr_outputs_99'
    cfg.freeze()

    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=args.output_dir).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=args.retrain)

    res = Trainer.test(cfg, model)
    if cfg.TEST.AUG.ENABLED:
        res.update(Trainer.test_with_TTA(cfg, model))
    if comm.is_main_process():
        verify_results(cfg, res)
        return res


def retina_test_main_100(args):
    cfg = retina_setup(args)
    cfg.defrost()
    cfg.INPUT.WINDOW = (0, 1)
    cfg.MODEL.WEIGHTS = 'retina_final_100.pth'
    cfg.OUTPUT_DIR = './retina_outputs_100'
    cfg.freeze()

    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=args.output_dir).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=args.retrain)

    res = Trainer.test(cfg, model)
    if cfg.TEST.AUG.ENABLED:
        res.update(Trainer.test_with_TTA(cfg, model))
    if comm.is_main_process():
        verify_results(cfg, res)
        return res


def retina_test_main_995(args):
    cfg = retina_setup(args)
    cfg.defrost()
    cfg.INPUT.WINDOW = (0.005, 0.995)
    cfg.MODEL.WEIGHTS = 'retina_final_995.pth'
    cfg.OUTPUT_DIR = './retina_outputs_995'
    cfg.freeze()

    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=args.output_dir).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=args.retrain)

    res = Trainer.test(cfg, model)
    if cfg.TEST.AUG.ENABLED:
        res.update(Trainer.test_with_TTA(cfg, model))
    if comm.is_main_process():
        verify_results(cfg, res)
        return res


def retina_test_main_99(args):
    cfg = retina_setup(args)
    cfg.defrost()
    cfg.INPUT.WINDOW = (0.01, 0.99)
    cfg.MODEL.WEIGHTS = 'retina_final_99.pth'
    cfg.OUTPUT_DIR = './retina_outputs_99'
    cfg.freeze()

    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=args.output_dir).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=args.retrain)

    res = Trainer.test(cfg, model)
    if cfg.TEST.AUG.ENABLED:
        res.update(Trainer.test_with_TTA(cfg, model))
    if comm.is_main_process():
        verify_results(cfg, res)
        return res
