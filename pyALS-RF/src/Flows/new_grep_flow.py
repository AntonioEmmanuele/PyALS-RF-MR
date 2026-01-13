import logging
# import numpy as np
from .NEW_GREP.NEW_GREP import NEW_GREP
from .NEW_GREP.NEW_ResiliencyBasedGREP2 import NEW_ResiliencyBasedGREP2
from ..ctx_factory import load_configuration_ps, create_classifier
from ..ConfigParsers.PsConfigParser import *
from distutils.dir_util import mkpath # type: ignore
import os

""" 
    This is the code of the new-grep flow.
"""
def new_grep_flow(  ctx: dict, quantization_type,
                    in_pruning, fraction: float, global_impact_method: str,
                    ncpus, cost_criterion: str, minredundancy: int, maxloss: float, 
                    validation_epoch: float,  report_dir:str , pruning_dir: str):
    logger = logging.getLogger("pyALS-RF")
    logger.info("Running the NEW-GREP pruning flow.")
    load_configuration_ps(ctx)
    create_classifier(ctx)
    
    
    trimmer = NEW_ResiliencyBasedGREP2(ctx.obj["classifier"], fraction, maxloss, minredundancy, ncpus, validation_epoch, global_impact_method)
    if quantization_type != None:
        trimmer.classifier.set_thds_type(quantization_type)
    trimmer.trim(NEW_GREP.get_cost_criterion(cost_criterion))
    trimmer.store_pruning_conf(os.path.join(pruning_dir, "pruning_cfg.json5"))
    trimmer.save_reports(report_dir)
    trimmer.dump_pruning_val_idx(pruning_dir)
    trimmer.export_per_class_csv(pruning_dir)
    trimmer.dump_prediction_vectors(pruning_dir)


""" 
    Redundancy Initialization of the new_grep
"""
def dump_pv_on_file(  ctx: dict, quantization_type,
                        outpath):
    
    logger = logging.getLogger("pyALS-RF")
    logger.info("Running the NEW-GREP pruning flow.")
    load_configuration_ps(ctx)
    create_classifier(ctx)
    
    
    trimmer = NEW_ResiliencyBasedGREP2(ctx.obj["classifier"], fraction, maxloss, minredundancy, ncpus, validation_epoch, global_impact_method)
    
    if quantization_type != None:
        trimmer.classifier.set_thds_type(quantization_type)
    trimmer.trim(NEW_GREP.get_cost_criterion(cost_criterion))
    trimmer.store_pruning_conf(os.path.join(pruning_dir, "pruning_cfg.json5"))
    trimmer.save_reports(report_dir)
    trimmer.dump_pruning_val_idx(pruning_dir)
    trimmer.export_per_class_csv(pruning_dir)
    trimmer.dump_prediction_vectors(pruning_dir)
