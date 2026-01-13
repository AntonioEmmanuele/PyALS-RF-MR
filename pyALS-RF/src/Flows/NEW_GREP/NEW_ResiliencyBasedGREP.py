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
import logging
from multiprocessing import cpu_count
from tqdm import tqdm
from ...Model.Classifier import Classifier
from .NEW_GREP import NEW_GREP
import numpy as np
import os
import csv

class NEW_ResiliencyBasedGREP(NEW_GREP):
    
    """ Static methods used to estimate the global impact of a pruning procedure."""
    @staticmethod
    def can_prune_max(residual_redundancy, min_resiliency):
        return np.max(residual_redundancy) >= min_resiliency 

    @staticmethod
    def can_prune_mean(residual_redundancy, min_resiliency):
        return np.mean(residual_redundancy) >= min_resiliency
    
    @staticmethod
    def can_prune_min(residual_redundancy, min_resiliency):
        return np.min(residual_redundancy) >= min_resiliency
    

    def __init__(self, classifier : Classifier, pruning_set_fraction : float = 0.5, max_loss : float = 5.0, min_resiliency : int = 0, ncpus : int = cpu_count(), validation_epoch : float = 0.1, prune_method: str = "max"):
        self.global_impact_evaluation = {
            "max":  NEW_ResiliencyBasedGREP.can_prune_max,
            "mean": NEW_ResiliencyBasedGREP.can_prune_mean,
            "min":  NEW_ResiliencyBasedGREP.can_prune_min,
        }

        super().__init__(classifier, pruning_set_fraction, max_loss, min_resiliency, ncpus, validation_epoch)
        self.global_impact_fn = self.global_impact_evaluation[prune_method]
        self.global_impact_attr = prune_method
        
        
    def save_reports(self, outdir):
        logger = logging.getLogger("pyALS-RF")
        os.makedirs(outdir, exist_ok=True)
        csv_path = os.path.join(outdir, "pruning_summary.csv")
        
        
        row = {
            "MaxLoss":                 self.max_loss,
            "MinRedundancy":           self.min_resiliency,
            "EpochPercentage":         self.validation_epoch,
            "CostCriterion":           self.criterion_name,
            "GlobalImpact":            self.global_impact_attr,

            # valori che nel CSV appaiono come [x], quindi prendo il primo elemento
            "Baseline_Validation_Acc": self.baseline_acc_validation,
            "Final_Validation_Acc":    self.val_accuracy,
            "Validation_Loss":         self.loss_val,
            "Baseline_Test_Acc":       self.baseline_accuracy_test,
            "Final_Test_Acc":          self.test_accuracy,
            "Test_Loss":               self.loss_test,

            "Original_Cost":           self.original_cost,
            "Final_Cost":              self.final_cost,
            "Expected_Savings":        self.expected_savings,
            "Pruned_Leaves_Count":     len(self.pruning_configuration),
            
            "Redundant_Samples_Count": len(self.initial_redundancy),
            
        }


        file_exists = os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames = list(row.keys()), delimiter = ';')
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
        logger.info(f"Pruning summary saved to {csv_path}")

    """  
        This function evaluates whether the residual redundancy of remaining samples is sufficient to perform a split 
        operation.
    """
    def can_prune_global_impact(self, residual_redundancy):
        return self.global_impact_fn(residual_redundancy, self.min_resiliency)

    
    """ Algorithm:  
        Inputs:
        R               -> samples with redundancy
        leaves          -> leaves for each samples
        leaf_samples    -> samples for each different leaf
        redundancy_thd  -> minimum redundancy for each sample
        epoch_max       -> Number of samples after which accuracy is evaluated on the validation set.
        validation_set  -> Set of samples used to validate the accuracy.
        acc_validation  -> Accuracy of the original classifier on the validation set 

        temporary_pruning_cfg = []
        pruning_cfg = []

        while len(R) > 0: 
            sample, actual_redundancy = pop(R)
            leaves_for_sample = leaves[sample]
            if actual_redundancy < rendundancy_thds:
                break
            epoch_counter += 1
            for leaf in leaves:
                other_samples = leaf_samples[leaf]
                residual_redundancy = R[other_samples]
                if leaf not in temporary_pruning_cfg and can_prune_global_impact(residual_redundancy):
                    pending_evaluation = True
                    temporary_pruning_cfg.append(leaf)
                    actual_redundancy = actual_redundancy - 1
                    for s in  other_samples: 
                        R[s] = R[s] - 1
                    sort_descending(R)
                    if actual_redundancy < redundancy_thd:
                        break
            if epoch_counter == epoch_max: 
                epoch_counter = 0
                pending_evaluation = False
                classifier_pruned = prune(temporary_pruning_cfg)
                acc_pruned = evaluate_accuracy(classifier_pruned, validation_set)
                loss = acc_validation - acc_pruned
                if loss <= accepted_loss:
                    pruning_cfg = temporary_pruning_cfg 
                else:
                    break
                    
        if pending_evaluation:
            pending_evaluation = False
            classifier_pruned = prune(temporary_pruning_cfg)
            acc_pruned = evaluate_accuracy(classifier_pruned, validation_set)
            loss = acc_validation - acc_pruned
            if loss <= accepted_loss:
                pruning_cfg = temporary_pruning_cfg 
            else:
                break
    """
    def trim(self, cost_criterion : NEW_GREP.CostCriterion):
        super().trim(cost_criterion)
        logger = logging.getLogger("pyALS-RF")
        self.cost_criterion = cost_criterion
        self.pruning_configuration = []
        
        epoch_counter =  0 
        temporary_validation_acc = 0.0
        temporary_validation_loss = 0.0
        self.loss_val = 0.0

        processing_list = self.initial_redundancy[:]
        epoch_max = int(len(processing_list) * self.validation_epoch)
        logger.info(f"Epoch Max {epoch_max}")
        
        pbar = tqdm(total=len(processing_list), desc="Redundancy-based hedge trimming...", bar_format="{desc:30} {percentage:3.0f}% |{bar:40}{r_bar}{bar:-10b}", leave=False)
        
        # For each initialized sample.
        while len(processing_list) > 0:
            x, _ = processing_list.pop(0)
            pbar.update(1)
            # Get the actual redundancy, and the set of leaves reached by the sample
            actual_redundancy = self.samples_info[NEW_GREP.sample_to_str(x)]["r"]
            active_leaves = self.samples_info[NEW_GREP.sample_to_str(x)]["leaves"]        
            # If the redundancy of the sample is enough
            if actual_redundancy >= self.min_resiliency:
                # Increase the epoch counter    
                epoch_counter += 1
                # For each leaf related to the sample
                for tree_name, class_name, leaf in tqdm(active_leaves, total = len(active_leaves), desc="Evaluating leaves...", bar_format="{desc:30} {percentage:3.0f}% |{bar:40}{r_bar}{bar:-10b}", leave=False):
                    # Get the ID of the leaf
                    leaf_id = (class_name, tree_name, leaf)
                    # Get all the other samples related to that list.
                    samples = self.leaves_info[(tree_name, class_name, leaf)]["samples"]
                    # Get the CURRENT redundancy of that samples.
                    residual_redundancy = [ self.samples_info[NEW_GREP.sample_to_str(x)]["r"] for x in samples ]
                    # If the leaf is not ALREADY in the pruning_cfg, and its global impact is valid, then approximate it
                    if leaf_id not in self.temporary_pruning_configuration and self.can_prune_global_impact(residual_redundancy):
                        # Each time a new leaf if added into the current_pruning_conf, the pending flag is enabled
                        pending_axc_acceptance = True 
                        # Append to the temporary pruning configuration.
                        self.temporary_pruning_configuration.append(leaf_id)
                        logger.debug(f"Adding {leaf_id} to the list of pruned assertions.")
                        # Decrease the actual redundancy of that leaf.
                        actual_redundancy -= 1
                        # Update the redundancy of the samples.
                        self.update_redundancy(samples)
                        # Update the redundancy of the current sample.
                        self.samples_info[NEW_GREP.sample_to_str(x)]["r"] = actual_redundancy
                        # Get the set of sorted samples info.
                        
                        processing_list = [
                            (x, self.samples_info[NEW_GREP.sample_to_str(x)]["r"])
                            for x, _ in processing_list
                        ]
                        processing_list.sort(key=lambda item: item[1], reverse=True)
                        # If the actual redundancy is less than the minimum resiliency, stop approximating for the sample.
                        if actual_redundancy < self.min_resiliency:
                            break
            else:
                logger.info(f"[INFO] Exiting due to terminated actual redundancy {actual_redundancy}")
                break
            
            # If the number of approximated samples equals epoch_max, validate the accuracy loss.
            # If acceptable, the temporary_pruning_conf becames the actual pruning conf.
            if epoch_counter > epoch_max:
                logger.info("Validanting accuracy on validation set")
                # Set only the temporary pruning configuration.
                NEW_GREP.set_pruning_conf(self.classifier, self.temporary_pruning_configuration)
                temporary_validation_acc    = self.evaluate_accuracy(self.args_evaluate_validation, self.y_val)[0]
                temporary_validation_loss   = self.baseline_acc_validation - temporary_validation_acc
                # The pending flag is disabled each time a configuration has been evaluated.
                pending_axc_acceptance = False
                # If the loss on the validation set is above the maximum allowed, stop pruning.
                if temporary_validation_loss > self.max_loss:
                    logger.info(f"[INFO] Exiting due to terminated MAX_LOSS_VIOLATION {temporary_validation_loss}")
                    break
                else:
                    
                    # Otherwise, reset the counter, update the actual pruning configuration, and the validation loss parameters
                    self.loss_val = temporary_validation_loss
                    self.val_accuracy = temporary_validation_acc
                    # Take only the current slide of the temporary pruning configuration.
                    # Shallow Copy should be enough, no need for a deep copy.
                    self.pruning_configuration = self.temporary_pruning_configuration[:len(self.temporary_pruning_configuration)] 
                    epoch_counter = 0        
        pbar.close()
        # if there is still a configuration to be evaluated.
        if pending_axc_acceptance: 
            # Set only the temporary pruning configuration.
            NEW_GREP.set_pruning_conf(self.classifier, self.temporary_pruning_configuration)
            temporary_validation_acc    = self.evaluate_accuracy(self.args_evaluate_validation, self.y_val)[0]
            temporary_validation_loss   = self.baseline_acc_validation - temporary_validation_acc
            # The pending flag is disabled each time a configuration has been evaluated.
            pending_axc_acceptance = False
            # If the loss is acceptable, then incorporate the new configuration.
            if temporary_validation_loss <= self.max_loss:
                # Otherwise, reset the counter, update the actual pruning configuration, and the validation loss parameters
                self.loss_val = temporary_validation_loss
                self.val_accuracy = temporary_validation_acc
                # Take only the current slide of the temporary pruning configuration.
                # Shallow Copy should be enough, no need for a deep copy.
                self.pruning_configuration = self.temporary_pruning_configuration[:len(self.temporary_pruning_configuration)] 
                logger.info(f"Current validation accuracy : {temporary_validation_acc} Loss : {temporary_validation_loss}")
        # A non-optimal solution, violating the vinculum of the maximum allowed loss on the validation set, could be present here.
        # For this reason, we restore the BNS 
        self.restore_bns()
        # Finally, we can set the pruning conf. We're sure that this does not violate the maximum allowed loss on the validation set.
        NEW_GREP.set_pruning_conf(self.classifier, self.pruning_configuration)
        self.final_cost = self.get_cost()
        
        logger.info(f" [VALIDATION SET] Baseline accuracy : {self.baseline_acc_validation}%, Accuracy AxC Classifier : {self.val_accuracy}%, Loss : {self.loss_val}%")
        
        # Accuracy evaluation on the Test test
        logger.info("Computing TEST Acc of the AxC Classifier")
        self.test_accuracy = self.evaluate_accuracy(self.args_evaluate_test, self.y_test)[0]
        self.loss_test = self.baseline_accuracy_test - self.test_accuracy
        logger.info(f" [TEST SET] Baseline accuracy : {self.baseline_accuracy_test}%, Accuracy AxC Classifier : {self.test_accuracy}%, Loss : {self.loss_test}%")       
        self.expected_savings = (1 - self.final_cost / self.original_cost) * 100
        logger.info(f"Final cost: {self.final_cost}. Expected saving is {self.expected_savings}%")
        logger.info(f"Logging execution..")

    
    """ Decrease the redundancy of a set of samples, after a leaf voting for their correct label, has been pruned."""
    def update_redundancy(self, samples):
        logger = logging.getLogger("pyALS-RF")
        for x in samples:
            self.samples_info[NEW_GREP.sample_to_str(x)]["r"] -= 1
            logger.debug(f"\tDecreasing resiliency for sample {x}. Residual redundancy: {self.samples_info[NEW_GREP.sample_to_str(x)]['r']}. Cost now is {self.get_cost()}.")
