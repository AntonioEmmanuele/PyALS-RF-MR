"""
Copyright 2021-2025 Salvatore Barone <salvatore.barone@unina.it>
                    Antonio Emmanuele <antonio.emmanuele@unina.it> 

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
import logging, numpy as np
import copy
import json5
import os
import csv
from multiprocessing import cpu_count
from tabulate import tabulate
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from ...Model.Classifier import *
from ...Model.DecisionTree import *
from ...plot import boxplot
from enum import Enum
from scipy.stats import norm # For cut-offs.
from enum import Enum, unique

class NEW_GREP:
        

    @unique
    class CostCriterion(Enum):
        depth = 1      # higher the depth higher the cost
        activity = 2   # lower the frequency of activation higher the cost
        combined = 3   # both the previous, combined; thus, leaves with the same costs in terms of depth but with lower frequency of activations cost more!

    @staticmethod
    def cc_to_name(criterion):
        # Se arriva direttamente un Enum
        if isinstance(criterion, NEW_GREP.CostCriterion):
            return criterion.name
        
        # Se arriva come intero (1, 2, o 3)
        try:
            return NEW_GREP.CostCriterion(criterion).name
        except ValueError:
            return "NoValidCriterion"
    
    """ Initialize the GREP Object
        classifier: Classifier object to be pruned.
        pruning_set_fraction: fraction of the test set to be used as pruning set.
        max_loss: maximum accuracy loss, accepted during the pruning.
        min_resiliency: minimum resiliency to be guaranteed for each sample.
        ncpus: number of CPUs to be used.
        validation_epoch: Percentage of pruning set samples after which accuracy is estimated during trimming.
    """
    def __init__(self, classifier : Classifier, pruning_set_fraction : float = 0.5, max_loss : float = 5.0, min_resiliency : int = 0, ncpus : int = cpu_count(), validation_epoch : float = 0.1):
        self.classifier = classifier
        self.pruning_set_fraction = pruning_set_fraction
        self.max_loss = max_loss
        self.min_resiliency = np.int64(min_resiliency)
        self.ncpus = min(ncpus, len(self.classifier.trees))
        self.validation_epoch = validation_epoch
    
    """ 
        Store the pruning configuration into a JSON5 file.
    """
    def store_pruning_conf(self, outfile : str):
        with open(outfile, "w") as f:
            json5.dump(self.pruning_configuration, f, indent=2)
    """
        Execute a backup of the boolean network of trees, before applying any pruning.
        This is useful to restore the original classifier if a pruning is not accepted.
    """
    def backup_bns(self):
        self.bns_backup = { t.name : copy.deepcopy(t.boolean_networks) for t in self.classifier.trees }

    """ 
        Restore the boolean network for trees from the backup.
    """ 
    def restore_bns(self):
        for t in self.classifier.trees:
            t.boolean_networks = self.bns_backup[t.name]

    """This function generates the three different splits required by the algorithm.
            validation_set_size     -> Percentage of the test set used as validation_set.
            increase_pruning_size   -> Append the validation set to the pruning set. 
            x_pruning, y_pruning -> These are set internally to the entire training set.
                                    This is done in order to cover all the possible leaves
                                    when computing the redundancy. 
            x_val, y_val, x_test, y_test -> These are splitted by leveraging the test_set.
    """
    def split_test_dataset(self, validation_set_size : float = 0.5, increase_pruning_size: bool = True):
        self.x_pruning = self.classifier.x_train # Use the training set for pruning
        self.y_pruning = self.classifier.y_train
        indexes = np.arange(len(self.classifier.x_test))
        self.x_val, self.x_test, self.y_val, self.y_test, self.idx_val, self.idx_test = train_test_split(self.classifier.x_test, self.classifier.y_test, indexes, train_size = validation_set_size) # Use stratify = self.classifier.x_test.ravel() ensures that all classess are considered. 
        if increase_pruning_size : 
            self.x_pruning = np.vstack((self.x_pruning, self.x_val))
            self.y_pruning = np.vstack((self.y_pruning, self.y_val))

    """ 
        This function receives in input the multicore parameter of the three possible sets (e.g. pruning, val, test)
        and the corresponding class label. 
        It performs an accuracy evaluation on that set.     
    """
    def evaluate_accuracy(self, args, y): 
        outcomes = np.sum(self.pool.starmap(Classifier.compute_score, args), axis = 0)
        return np.sum(np.argmax(o) == y and not self.classifier.check_draw(o)[0] for o, y in zip(outcomes, y)) / len(y) * 100  
    
    
    """ 
        This function simply performs tree visiting, returning the votes vector for each input sample.     
    """
    def visit(self, args): 
        return np.sum(self.pool.starmap(Classifier.compute_score, args), axis = 0)
    
    """ 
        Given a set of votes vectors, outcomes, this function returns the accuracy on that set     
    """
    def acc_eval(self, outcomes, y):
        return (np.sum(np.argmax(o) == y and not self.classifier.check_draw(o)[0] for o, y in zip(outcomes, y)) / len(y) * 100 )[0]

    """ Given in input a list of votes vectors, and the set of correct class labels,
    this function estimates the per_class_accuracy. """
    def eval_per_class_acc(self, outcomes, y):
        y_true = np.asarray(y).reshape(-1)
        y_pruning_arr = np.asarray(self.y_pruning).reshape(-1)
        model_classes = np.unique(y_pruning_arr)
        per_class_acc = {}
        outcomes_linear = np.array(
            [
                np.argmax(o) if not self.classifier.check_draw(o)[0] else -1
                for o in outcomes
            ]
        )
        
        if outcomes_linear.shape[0] != y_true.shape[0]:
            raise ValueError(
                f"Mismatch: outcomes has {outcomes_linear.shape[0]} samples, "
                f"but y has {y_true.shape[0]} samples."
            )

        for c in model_classes:
            idx = (y_true == c)
            total = int(np.sum(idx))

            if total == 0:
                per_class_acc[c] = {
                    "accuracy": None,
                    "count": 0
                }
                continue

            correct = int(np.sum(outcomes_linear[idx] == y_true[idx]))

            per_class_acc[c] = {
                "accuracy": correct / total * 100.0,
                "count": total
            }

        return per_class_acc
    
    def compute_per_class_loss(self, per_class_acc_exact, per_class_acc_axc):
        per_class_loss = {}

        # We iterate over all classes found in either dictionary
        all_classes = set(per_class_acc_exact.keys()) | set(per_class_acc_axc.keys())

        for c in all_classes:
            exact_info = per_class_acc_exact.get(c, {"accuracy": None, "count": 0})
            axc_info   = per_class_acc_axc.get(c,   {"accuracy": None, "count": 0})

            # If class has zero samples, loss = 0
            if exact_info["count"] == 0:
                per_class_loss[c] = 0
                continue

            acc_exact = exact_info["accuracy"]
            acc_axc   = axc_info["accuracy"]

            # Handle cases where one of them is None
            if acc_exact is None or acc_axc is None:
                per_class_loss[c] = 0
            else:
                per_class_loss[c] = acc_exact - acc_axc

        return per_class_loss

    """ This function is called before launching the trim functionality.
        For each sample in the pruning set, it evaluates its redundancy. 
        Moreover, it produces in output the set os samples (samples info, sorted in decreased order of redundancy.) 
        For each sample, the set of leaves related to it are sorted.
        Post-Conditions 
            self.samples_info -> Dictionary mantaining:
                Key: string_of_sample_x : { redundancy: <red_value>, leaves_reaches: <ReachedLeaves>}
            self.initial_redundancy -> List of tuples.
                (x_sample, redundancy)
            self.leaves_info -> Vec
            
    """
    def evaluate_redundancy(self):
        logger = logging.getLogger("pyALS-RF")
        # keeps the initial redundancy of each sample (it's easier to sort a list of tuples)
        self.initial_redundancy = [] 
        # for each sample, keeps its residual redundancy and the list of prunable minterm
        self.samples_info = {NEW_GREP.sample_to_str(x) : { "r": 0, "leaves" : []} for x in self.x_pruning} 
        # for each leaf, of each tree, keeps the list of sample belonging to that leaf
        self.leaves_info = {} 
        self.val_accuracy_pruning_set = 0

        # compute the tree visit for each of the sample
        logger.info("Computing the pruning capabilities")
        # Compute the redundancy of the sample.
        tree_visiting_outcomes = self.pool.starmap(NEW_GREP.compute_redundancy, self.args_evaluate_pruning) 
        logger.info("Done")

        # Flatten the return value of the starmap.
        per_sample_visits = []
        for sample_blocks in zip(*tree_visiting_outcomes):   
            merged = [leaf_info for block in sample_blocks for leaf_info in block]
            per_sample_visits.append(merged)

        # per_sample_visits is a list of tuples holding for each value the tuple (tree_name, sample_name, one_hot_prediction)
        for x, y, visiting_outcomes in tqdm(
                zip(self.x_pruning, self.y_pruning, per_sample_visits),
                total=len(self.y_pruning),
                desc="Computing redundancy",
                bar_format="{desc:30} {percentage:3.0f}% |{bar:40}{r_bar}{bar:-10b}",
                leave=False):

            logger.debug(f"Visiting outcomes: {visiting_outcomes}")
            # Score is the vector holding for each label the number of votes received !
            # compure the score of that sample, i.e. its output prediction vector.
            score = np.sum(i[3] for i in visiting_outcomes)
            logger.debug(f"Score: {score}") 
            # If the value was correctly predicted.
            if np.argmax(score) == y[0] and not self.classifier.check_draw(score)[0]:
                self.val_accuracy_pruning_set += 1
                # in this case compute the number of prunable leaves
                r = np.sort(np.array(score, copy=True))[::-1]
                # Estimate the redundancy.
                self.samples_info[NEW_GREP.sample_to_str(x)]["r"] = (r[0] - r[1] - 1) // 2
                logger.debug(f"Initial redundancy: {self.samples_info[NEW_GREP.sample_to_str(x)]['r']}")
                # Append this also to the initial redundancy vector. 
                self.initial_redundancy.append((x, self.samples_info[NEW_GREP.sample_to_str(x)]["r"]))
                # among all leaves resulting from trees visiting, the leaves that can be pruned are those resulting in correct classification
                self.samples_info[NEW_GREP.sample_to_str(x)]["leaves"] = [ i[:3] for i in visiting_outcomes if np.argmax(i[3]) == y[0] ]
                logger.debug(f"Candidate leaves: {self.samples_info[NEW_GREP.sample_to_str(x)]['leaves']}")
                # for each of the leaves resulting in correct classification, the list of samples resulting in that leaf is updated
                # in order to have the complete list of samples activating each leaf in the forest
                for leaf in self.samples_info[NEW_GREP.sample_to_str(x)]["leaves"]: # note that leaf is actually a tuple (tree name, class name, leaf)
                    if leaf not in self.leaves_info:
                        self.leaves_info[leaf] = {"cost" : 0.0, "samples" : []}
                    self.leaves_info[leaf]["samples"].append(x)
                    logger.debug(f"Adding {x} to the list of samples resulting in {leaf}")
        
        # After populating it, sort the samples depending on their redundancy value.
        self.initial_redundancy.sort(key=lambda x: x[1], reverse = True)
        logger.info(f"Found {len(self.leaves_info)} candidate leaves")
        self.original_number_of_leaves = NEW_GREP.gen_total_leaf_count(self.classifier)
        logger.info(f"Number of leaves in the ensemble: {self.original_number_of_leaves}")
        self.val_accuracy_pruning_set = self.val_accuracy_pruning_set * 100 / len(self.y_pruning)     
        logger.info(f"Accuracy (on the pruning set): {self.val_accuracy_pruning_set}%")

    """ This function perform the sorting operation of the leaves.
        in short, the field info, of samples_info, will contain the leaves sorted accordingly to a specific criterion
    """
    def sort_leaves_by_cost(self, cost_criterion : CostCriterion):
        logger = logging.getLogger("pyALS-RF")
        # compute the cost of each leaf first, based on depth and activations
        for leaf, info in self.leaves_info.items():
            literals = len(leaf[2].split("and"))
            activations = len(info["samples"])
            if cost_criterion == NEW_GREP.CostCriterion.depth:
                info["cost"] = literals
            elif cost_criterion == NEW_GREP.CostCriterion.activity:
                info["cost"] = 1 / activations
            elif cost_criterion == NEW_GREP.CostCriterion.combined:
                info["cost"] = literals / activations # leaves with the same costs in terms of literals but with less activity cost more!
            logger.debug(f"Cost of {leaf} is {literals}/{activations}={info['cost']}")
        # now, for each of the activing sample, sort the list of leaves based on their cost
        for info in self.samples_info.values():
            leaves_and_their_cost = [ (leaf, self.leaves_info[leaf]["cost"]) for leaf in info["leaves"] ]
            logger.debug(f"Sorting leaves\n{leaves_and_their_cost}")
            leaves_and_their_cost.sort(key=lambda x: x[1], reverse = True)
            info["leaves"] = [ l[0] for l in leaves_and_their_cost]
            logger.debug(f"Sorted leaves\n{info['leaves']}")
    
    def redundancy_boxplot(self, outfile):
        boxplot([ i[1] for i in self.initial_redundancy ], "", "Redundancy", outfile, figsize = (2, 4), annotate = False, integer_only= True)

    """ Return the cost of the boolean network of the entire ensemble. """  
    def get_cost(self):
        return sum( NEW_GREP.get_bns_cost(t) for t in self.classifier.trees )
    
    @staticmethod
    def sample_to_str(x):
        return ';'.join(str(i) for i in x.tolist())
    
    @staticmethod
    def get_cost_criterion(criterion : str):
        return { "depth" : NEW_GREP.CostCriterion.depth, "activity" : NEW_GREP.CostCriterion.activity, "combined" : NEW_GREP.CostCriterion.combined}[criterion]
    
    """ For an input sample, this function returns:
        1. the tree_name
        2. the class label (i.e. class name)
        3. the leaf boolean expression.
        4. The class label in one hot.  """
    @staticmethod
    def tree_visit_with_leaf(tree : DecisionTree, attributes):
        boxes_output = tree.get_boxes_output(attributes)
        prediction_as_one_hot = np.array([eval(a["sop"], boxes_output) for a in tree.boolean_networks ], dtype=int)
        for class_name, assertions in tree.class_assertions.items():
            for leaf in assertions:
                if eval(leaf, boxes_output):
                    return tree.name, class_name, leaf, prediction_as_one_hot
    
    """ Return the number of leaves in the ensemble """
    @staticmethod 
    def gen_total_leaf_count(classifier): 
        leaves_cnt = 0
        for tree in classifier.trees: 
            for class_name, assertions in tree.class_assertions.items():
                for leaf in assertions:
                    leaves_cnt += 1
        return leaves_cnt
    
    """ Function used to estimate the cost, in terms of and gates, of the boolean network of a specific classifier. 
        This fun is intended to be invoked on a single decision tree.
    """
    @staticmethod          
    def get_bns_cost(tree : DecisionTree):
        literal_cost = 0
        for network in tree.boolean_networks:
            for minterm in network["minterms"]:
                literal_cost += len(minterm.split(" and "))
        return literal_cost
    
    """ Function used for setting the pruning configuration, pruning_conf, on a specific classifier. 
        Internally, for each tree in the ensemble, it calls the set_pruning function, setting the pruning
        configuration on that specific tree. 
    """
    @staticmethod
    def set_pruning_conf(classifier : Classifier, pruning_conf):
        total_leaves_pruned = 0
        for t in classifier.trees:
            total_leaves_pruned += NEW_GREP.set_pruning(t, pruning_conf)
        return total_leaves_pruned
    
    """ Function used to set the pruning configuration ON A SPECIFIC DECISION TREE CLASSIFIER.
        This is intended to be invoked only by the set_pruning_conf function.
    """
    @staticmethod
    def set_pruning(tree : DecisionTree, pruning_configuration, use_espresso : bool = False):
        logger = logging.getLogger("pyALS-RF")
        nl = '\n'
        #tree.boolean_networks = []
        logger.debug(f"Setting pruning configuration for {tree.name}")
        pruned_leaves = 0
        for bn, (class_name, assertions) in zip(tree.boolean_networks, tree.class_assertions.items()):
            pruned = [assertion for class_label, tree_name, assertion in pruning_configuration if tree_name == tree.name and class_label == class_name ] 
            pruned_leaves += len(pruned)
            kept_assertions = [ assertion for assertion in assertions if assertion not in pruned ]
            logger.debug(f"Pruning on tree {tree.name}, class {class_name}: {len(kept_assertions)} leaves kept out of {len(bn['minterms'])}")          
            kept_assertions, sop, hdl_expression = tree.define_boolean_expression(kept_assertions, use_espresso)
            bn['minterms'] = kept_assertions
            bn['sop'] = sop
            bn['hdl_expression'] = hdl_expression
            #tree.boolean_networks.append({"class" : class_name, "minterms" : kept_assertions, "sop" : sop, "hdl_expression" : hdl_expression})
        logger.debug(f'Tree {tree.name} pruning configuration:\n{tabulate([[bn["class"], f"{nl}".join(bn["minterms"]), bn["sop"].replace(" or ", f" or{nl}"), bn["hdl_expression"].replace(" or ", f" or{nl}")] for bn in tree.boolean_networks], headers=["class", "minterms", "SoP", "HDL"], tablefmt="grid")}')    
        return pruned_leaves
    
    """ Given in input:  
        - a tree
        - a tree id : ( integer provided by the user)
        - a class label
        This function returns the list of leaves for a specific class.
    """
    @staticmethod
    def get_pruning_conf_by_class(tree: DecisionTree, tree_id, class_label: int):
        pruned_leaves = [ ]
        for bn in tree.boolean_networks:
            if int(bn["class"]) == class_label:
                for leaf in bn["minterms"]:
                    pruned_leaves.append((str(class_label), str(tree_id), leaf))
        return pruned_leaves

    """ Internal static method used to visit the trees using the pruning samples. 
        This function is intended to be invoked within the evaluate_redundancy function.
    """
    @staticmethod
    def compute_redundancy(trees, dataset):
        return [[ NEW_GREP.tree_visit_with_leaf(t, x) for t in trees ] for x in dataset ]
    
    """ Given in input a classifier and a set of leaves indexes to prune, this function 
        returns pruning configuration. 
        pruned_leaves_idx_per_tree is a dictionary ( or a tree indexed list), containing
        for each tree the pruned leaves for each class.
     """
    @staticmethod
    def get_pruning_cfg_from_leaves_idx(classifier, pruned_leaves_idx_per_tree):
        pruning_cfg = []
        # For each tree.
        for tree_id, tree in enumerate(classifier.trees):
            pruned_leaves_per_class = pruned_leaves_idx_per_tree[tree_id]
            tree_pruning_cfg = []
            # For each class
            for considered_class, pruned_leaves in pruned_leaves_per_class.items():
                # For each pruned leaf per class.
                for pruned_leaf in pruned_leaves:
                    tree_pruning_cfg.append((str(considered_class), str(tree_id), tree.leaves[pruned_leaf]["sop"]))
            pruning_cfg.extend(tree_pruning_cfg)
        return pruning_cfg

    def compare(self):
        data = []
        self.restore_bns()
        exact_outcome = np.sum(self.pool.starmap(Classifier.compute_score, self.args_evaluate_validation), axis = 0)
        NEW_GREP.set_pruning_conf(self.classifier, self.pruning_configuration)
        pruned_outcome = np.sum(self.pool.starmap(Classifier.compute_score, self.args_evaluate_validation), axis = 0)
        for eo, po, x, y in zip(exact_outcome, pruned_outcome, self.x_val, self.y_val):
            if np.argmax(eo) != np.argmax(po) or Classifier.check_draw(eo) != Classifier.check_draw(po):
                data.append((x, y, eo, Classifier.check_draw(eo), po, Classifier.check_draw(po)))
        if data:
            print(tabulate(data, headers = ["Sample", "Class", "O.out", "O.draw", "P.out", "P.draw"], showindex="always"))
    
    
    def dump_pruning_val_idx(self, outdir):
        np.savetxt(os.path.join(outdir, "val_idx.txt"), self.idx_val, fmt = "%d")
        np.savetxt(os.path.join(outdir, "test_idx.txt"), self.idx_test, fmt = "%d")

    def trim(self, cost_criterion : CostCriterion):
        logger = logging.getLogger("pyALS-RF")
        self.cost_criterion = cost_criterion
        self.criterion_name = NEW_GREP.cc_to_name(self.cost_criterion)
        logger.info(f"Pruning set fraction: {self.pruning_set_fraction}")
        # Generate the pruning, validation and test splits.
        self.split_test_dataset(self.pruning_set_fraction)
        logger.info(f"Pruning set: {len(self.x_pruning)} samples")
        logger.info(f"Validation set: {len(self.x_val)} samples")
        logger.info(f"Test set size: {len(self.x_test)} samples")
        self.p_tree = self.classifier.p_tree 
        # Initialize the multicore parameters for inference.
        self.args_evaluate_pruning = [[t, self.x_pruning] for t in self.p_tree]
        self.args_evaluate_validation = [[t, self.x_val] for t in self.p_tree]
        self.args_evaluate_test = [[t, self.x_test] for t in self.p_tree]
        self.pool = self.classifier.pool
        logger.info("Computing baseline VALIDATION accuracy")
        # self.baseline_acc_validation = self.evaluate_accuracy(self.args_evaluate_validation, self.y_val)[0]
        baseline_out_val = self.visit(self.args_evaluate_validation)
        self.baseline_acc_validation = self.acc_eval(baseline_out_val, self.y_val)
        self.baseline_per_class_acc_validation = self.eval_per_class_acc(baseline_out_val, self.y_val)
        logger.info(f"Baseline accuracy (on validation set) : {self.baseline_acc_validation}%")
       
        logger.info("Computing baseline TEST accuracy")
        #self.baseline_accuracy_test = self.evaluate_accuracy(self.args_evaluate_test, self.y_test)[0]
        baseline_out_test = self.visit(self.args_evaluate_test)
        self.baseline_accuracy_test = self.acc_eval(baseline_out_test, self.y_test)
        self.baseline_per_class_acc_test = self.eval_per_class_acc(baseline_out_test, self.y_test)

        logger.info(f"Baseline accuracy (on test set) : {self.baseline_accuracy_test}%")
        self.original_cost = self.get_cost()
        logger.info(f"Original cost: {self.original_cost}")
        self.val_accuracy = self.baseline_acc_validation
        self.loss_val = 0
        logger.info("Performing Boolean networks backup")
        self.backup_bns()
        # Initialize the redundancy
        self.evaluate_redundancy()
        self.sort_leaves_by_cost(cost_criterion)
        self.pruning_configuration = []
        # This parameter keeps track of the temporary pruning configuration.
        # After a temporary pruning configuration is validated after an approximation epoch,
        # then it is assigned to the pruning configuration.
        self.temporary_pruning_configuration = []


    def export_per_class_csv(self, pruning_dir):
        """
        Export a CSV in wide format: one row and multiple columns,
        where each class contributes several class-specific metric columns.
        """

        # Collect all classes that appear in any dictionary
        all_classes = (
            set(self.baseline_per_class_acc_validation.keys()) |
            set(self.val_per_class_acc.keys()) |
            set(self.val_per_class_loss.keys()) |
            set(self.baseline_per_class_acc_test.keys()) |
            set(self.test_per_class_acc.keys()) |
            set(self.test_per_class_loss.keys())
        )

        # Helper: extract accuracy from dictionary or None
        def _get_acc(d):
            if d is None:
                return None
            if isinstance(d, dict):
                return d.get("accuracy", None)
            return d

        # Helper: extract count from dictionary or 0
        def _get_count(d):
            if d is None:
                return 0
            if isinstance(d, dict):
                return d.get("count", 0)
            return 0

        # =========================================
        # Build CSV header (wide format)
        # =========================================
        fieldnames = []

        for c in sorted(all_classes):
            fieldnames.extend([
                f"Class_{c}_Count",
                f"Class_{c}_AccTestExact",
                f"Class_{c}_AccTestAxc",
                f"Class_{c}_AccTestLoss",
                f"Class_{c}_AccValExact",
                f"Class_{c}_AccValAxc",
                f"Class_{c}_AccValLoss"
            ])

        # =========================================
        # Build the single output row
        # =========================================
        row = {}

        for c in sorted(all_classes):

            # Validation metrics
            val_exact = self.baseline_per_class_acc_validation.get(c)
            val_axc   = self.val_per_class_acc.get(c)
            val_loss  = self.val_per_class_loss.get(c, 0.0)

            # Test metrics
            test_exact = self.baseline_per_class_acc_test.get(c)
            test_axc   = self.test_per_class_acc.get(c)
            test_loss  = self.test_per_class_loss.get(c, 0.0)

            # Combine validation + test count
            class_count = _get_count(val_exact) + _get_count(test_exact)

            # Fill CSV columns for this class
            row[f"Class_{c}_Count"]          = class_count
            row[f"Class_{c}_AccTestExact"]   = _get_acc(test_exact) or 0
            row[f"Class_{c}_AccTestAxc"]     = _get_acc(test_axc) or 0
            row[f"Class_{c}_AccTestLoss"]    = test_loss
            row[f"Class_{c}_AccValExact"]    = _get_acc(val_exact) or 0
            row[f"Class_{c}_AccValAxc"]      = _get_acc(val_axc) or 0
            row[f"Class_{c}_AccValLoss"]     = val_loss

            # If no samples exist for this class → force zero loss
            if class_count == 0:
                row[f"Class_{c}_AccTestLoss"] = 0
                row[f"Class_{c}_AccValLoss"]  = 0

        # =========================================
        # Write CSV (one row only)
        # =========================================
        with open(os.path.join(pruning_dir, "per_class_reports.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(row)
    
    def dump_prediction_vectors(self, out_dir):
        """
        Save validation and test accuracy outcome vectors into .txt files.
        Each line contains the textual representation of one outcome array.
        """

        os.makedirs(out_dir, exist_ok=True)

        val_path = os.path.join(out_dir, "val_acc_outcomes.txt")
        test_path = os.path.join(out_dir, "test_acc_outcomes.txt")

        # Save validation outcomes
        with open(val_path, "w") as f:
            for arr in self.val_acc_outcomes:
                f.write(np.array2string(np.asarray(arr), separator=' ') + "\n")

        # Save test outcomes
        with open(test_path, "w") as f:
            for arr in self.test_acc_outcomes:
                f.write(np.array2string(np.asarray(arr), separator=' ') + "\n")

        # print(f"Saved validation outcomes to: {val_path}")
        # print(f"Saved test outcomes to: {test_path}")