# We are importing our own Trainer Module here to use the COCO validation evaluation during training.
# Otherwise no validation eval occurs.
import os

from detectron2.data import build_detection_test_loader, DatasetMapper
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, SemSegEvaluator
from ferrite.LossEvalHook import LossEvalHook


class CocoTrainer(DefaultTrainer):
  @classmethod
  def build_evaluator(cls, cfg, dataset_name, output_folder=None):
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    return SemSegEvaluator(dataset_name, cfg, True, output_folder)

  @classmethod
  def build_evaluator(cls, cfg, dataset_name, output_folder=None):
      if output_folder is None:
          output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
      return COCOEvaluator(dataset_name, cfg, True, output_folder)

  def build_hooks(self):
    hooks = super().build_hooks()
    hooks.insert(-1,LossEvalHook(
        self.cfg.TEST.EVAL_PERIOD,
        self.model,
        build_detection_test_loader(
            self.cfg,
            self.cfg.DATASETS.TEST[0],
            DatasetMapper(self.cfg,True)
        )
    ))
    return hooks