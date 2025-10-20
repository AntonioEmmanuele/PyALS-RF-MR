"""
Copyright 2021-2023 Salvatore Barone <salvatore.barone@unina.it>

This is free software; you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free
Software Foundation; either version 3 of the License, or any later version.

This is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
more details.

You should have received a copy of the GNU General Public License along with
RMEncoder; if not, write to the Free Software Foundation, Inc., 51 Franklin
Street, Fifth Floor, Boston, MA 02110-1301, USA.
"""
import os, numpy as np, logging
import os
from shutil import copy2 as copy_file
from pyalslib import YosysHelper
from jinja2 import Environment, FileSystemLoader
from .HDLGenerator import HDLGenerator
from ..Model.Classifier import Classifier
from ..Flows.GREP.GREP import GREP
from .GREPHdlGenerator import GREPHdlGenerator
import json5
from math import floor

def mkpath(path):
    """Create directories recursively, similar to distutils.dir_util.mkpath."""
    os.makedirs(path, exist_ok=True)

class MrHDLGenerator(HDLGenerator):
    
    def __init__(self, classifier : Classifier, yshelper : YosysHelper, destination : str):
        super().__init__(classifier, yshelper, destination)

    def copyfiles_exact(self, ax_dest : str):
        copy_file(self.source_dir + self.vhdl_bnf_source, f"{ax_dest}/src")
        copy_file(self.source_dir + self.vhdl_luts_source, f"{ax_dest}/src")
        copy_file(self.source_dir + self.vhdl_reg_source, f"{ax_dest}/src")
        copy_file(self.source_dir + self.vhdl_decision_box_source, f"{ax_dest}/src")
        copy_file(self.source_dir + self.vhdl_swapper_block_source, f"{ax_dest}/src")
        copy_file(self.source_dir + self.vhdl_simple_voter_source, f"{ax_dest}/src")
        copy_file(self.source_dir + self.vhdl_sorting_network_source, f"{ax_dest}/src")
    
    def copyfiles_mr_axc(self, ax_dest : str):
        copy_file(self.source_dir + self.vhdl_bnf_source, f"{ax_dest}/src")
        copy_file(self.source_dir + self.vhdl_luts_source, f"{ax_dest}/src")
        copy_file(self.source_dir + self.vhdl_reg_source, f"{ax_dest}/src")
        copy_file(self.source_dir + self.vhdl_decision_box_source, f"{ax_dest}/src")
        copy_file(self.source_dir + self.vhd_mr_solver_source, f"{ax_dest}/src")

                  
    def generate_exact_implementation(self, **kwargs):
        dest = f"{self.destination}/exact/"
        mkpath(self.destination)
        mkpath(dest)
        mkpath(f"{dest}/src")
        self.copyfiles_exact(dest)
        
        features = [{"name": f["name"], "nab": 0} for f in self.classifier.model_features]
        trees_name = [t.name for t in self.classifier.trees]
        env = Environment(loader = FileSystemLoader(self.source_dir))
        
        trees_inputs = {}
        data_width = None
        for tree in self.classifier.trees:
            boxes = self.get_dbs(tree)
            if data_width == None: 
                data_width = boxes[0]["box"].get_data_width()
            inputs = self.implement_decision_boxes(tree, boxes, f"{dest}/src")
            self.implement_assertions(tree, boxes, f"{dest}/src", kwargs['lut_tech'])
            trees_inputs[tree.name] = inputs
        
        self.generate_classifier(f"{dest}/src", features, trees_inputs, env, data_width)
        self.generate_rejection_module(f"{dest}/src", env)
        self.generate_majority_voter(f"{dest}/src", env)
    
    def generate_axhdl(self, **kwargs):
        
        # Generate the destination folder of the approximate classifier.
        dest = f"{self.destination}/mr_axc/"
        mkpath(self.destination)
        mkpath(dest)
        mkpath(f"{dest}/src")
        # Copy the files of the classifier
        self.copyfiles_mr_axc(dest)
        
        features = [{"name": f["name"], "nab": 0} for f in self.classifier.model_features]
        trees_name = [t.name for t in self.classifier.trees]
        env = Environment(loader = FileSystemLoader(self.source_dir))
        trees_inputs = {}

        with open(kwargs['pruning_configuration'], "r") as f:
            pc = json5.load(f)
        
        with open(kwargs['approximate_configuration'], "r") as f:
            axc = json5.load(f)
            
        # Set the pruning configuration
        GREP.set_pruning_conf(self.classifier, pc)    

        # For each tree, implement the decision boxes and the assertion functions.
        data_width = None
        for tree in self.classifier.trees:
            boxes = self.get_dbs(tree)
            if data_width is None: 
                data_width = boxes[0]["box"].get_data_width()
            inputs = self.implement_decision_boxes(tree, boxes, f"{dest}/src")
            self.implement_assertions(tree, boxes, f"{dest}/src", kwargs['lut_tech'])
            trees_inputs[tree.name] = inputs
            
        # self.generate_rejection_module(f"{dest}/src", env)
        # self.generate_majority_voter(f"{dest}/src", env)
        self.generate_classifier_mr(f"{dest}/src", features, trees_inputs, env, data_width, axc)
    
    def generate_classifier_mr(self, dest, features, trees, env, data_width, approximate_cfg):
        # print(approximate_cfg)
        # print(trees)
        # exit(1)
        """
        Generates the VHDL classifier using MR solvers (no majority voter).
        The threshold is computed as floor(len(approximate_cfg[0]) / 2) + 1.
        """
        # Compute the threshold value from the first approximate configuration
        thd_val = floor(len(approximate_cfg[0]) / 2) + 1

        # Load the classifier template
        classifier_template = env.get_template(self.vhd_mr_classifier_source_template)

        # Render the template with all required data
        classifier = classifier_template.render(
            data_width=data_width,
            trees=trees,
            features=features,
            classes=self.classifier.classes_name,
            candidates=self.classifier.classes_name,
            approximate_configuration=approximate_cfg,
            threshold_val=thd_val
        )

        # Write to destination file
        with open(f"{dest}/classifier.vhd", "w") as out_file:
            out_file.write(classifier)

        print(f"[INFO] Generated classifier with MR solvers → {dest}/classifier.vhd")
        print(f"[INFO] Threshold set to {thd_val}")
            

