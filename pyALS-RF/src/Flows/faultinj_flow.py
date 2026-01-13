"""
Copyright 2021-2024 Salvatore Barone <salvatore.barone@unina.it>
                    Antonio Emmanuele <antonio.emmanuele@unina.it>
This is free software; you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free
Software Foundation; either version 3 of the License, or any later version.

This is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
more details.

You should have received a copy of the GNU General Public License along with
pyALS-RF; if not, write to the Free Software Foundation, Inc., 51 Franklin
Street, Fifth Floor, Boston, MA 02110-1301, USA.
"""

import logging, joblib, numpy as np
from tqdm import tqdm
from ..ctx_factory import load_configuration_ps, create_classifier, store_flow
from ..ConfigParsers.PsConfigParser import *
from ..Model.Classifier import Classifier
from ..Model.FaultCollection import FaultCollection
import os 
import json5
import copy
import numpy as np
import struct
from .GREP.GREP import GREP
import random
from time import time
import pandas as pd

# Mapping for struct formats and sizes
TYPE_INFO = {
    "f64": { "fmt": ">d", "bytes": 8, "np_type": np.float64 },
    "f32": { "fmt": ">f", "bytes": 4, "np_type": np.float32 },
    "int16": { "fmt": ">h", "bytes": 2, "np_type": np.int16 },
    "int8":  { "fmt": ">b", "bytes": 1, "np_type": np.int8 },
}


""" Begin bit string manipulation functions. **** """
# # ------------------------------------------------------------
# # Generic: value → bitstring
# # ------------------------------------------------------------
# def value_to_bitstring(value, feat_repr):
#     """
#     Converts an input value to its bitstring representation
#     based on the feature representation (f64, f32, int16, int8).
#     """
#     info = TYPE_INFO[feat_repr]

#     # Cast to correct numpy type
#     np_value = info["np_type"](value)

#     # Pack into raw bytes using the correct struct format
#     packed = struct.pack(info["fmt"], np_value)

#     # Convert bytes into a bitstring (MSB → LSB)
#     bitstring = "".join(f"{byte:08b}" for byte in packed)

#     return bitstring

def value_to_bitstring(value, feat_repr):
    info = TYPE_INFO[feat_repr]

    
    np_value = info["np_type"](value).item()

    packed = struct.pack(info["fmt"], np_value)

    bitstring = "".join(f"{byte:08b}" for byte in packed)
    return bitstring

def bitstring_to_value(bitstring, feat_repr):
    info = TYPE_INFO[feat_repr]

    packed = bytes(
        int(bitstring[i:i+8], 2)
        for i in range(0, len(bitstring), 8)
    )

    return struct.unpack(info["fmt"], packed)[0]



# ------------------------------------------------------------
# Modify bits of a bitstring
# ------------------------------------------------------------
def modify_bitstring(bitstring, positions, values):
    """
    Modifies given bit positions (0 = MSB) with new values (0/1).
    """
    bit_list = list(bitstring)

    for pos, val in zip(positions, values):
        bit_list[pos] = str(val)

    return "".join(bit_list)


# ------------------------------------------------------------
# Fault Injection (generic)
# ------------------------------------------------------------
def inject_fault_feature(value, bit_positions, fixed_values, feat_repr):
    """
    Applies controlled bit-flips to a feature value.

    value:         The original feature value.
    bit_positions: List of bit indices to override (0 = MSB).
    fixed_values:  Replacement bit values (0 or 1).
    feat_repr:     One of ["f64", "f32", "int16", "int8"].
    """
    # print(value)
    # print(feat_repr)
    # Convert original value to bitstring
    bitstring = value_to_bitstring(value, feat_repr)

    # Inject bit errors
    injected = modify_bitstring(bitstring, bit_positions, fixed_values)

    # Convert back to numeric form
    return bitstring_to_value(injected, feat_repr)


def inject_fault_input(classifier: Classifier, faults, x_test, feat_repr = "f32"):
    # For each fault {feat_idx : {bit_flipped: value}}
    for f in faults:
        feature_idx = list(f.keys())[0]
        bit_to_flip = list(f[feature_idx].keys())[0]
        fixed_value = f[feature_idx][bit_to_flip]
        # Reconvert in integers
        feature_idx = int(feature_idx)
        bit_to_flip = int(bit_to_flip)
        fixed_value = int(fixed_value)
        # For each possible input alter the specific feature of the fault.
        for idx, ipt in enumerate(x_test):
            x_test[idx][feature_idx] = inject_fault_feature(value = ipt[feature_idx], bit_positions = [bit_to_flip], fixed_values = [fixed_value], feat_repr=feat_repr)


""" End bit string manipulation functions. **** """

""" 
    This function is used to generate the set of possible fault sites of a classifier.
    ctx : Possible context.
    pruning_cfg: Pruning configuration used in the case of an approximate classifier.
    working_mode: 
    error_margin: e parameter of the Levesgue formula.
    confidence_level: Confidence level of the Levesgue parameter used for SFI.
    individual_probability: Parameter used to specify the probability of a single individual to manifest a fault.
    out_dir: Output directory.
    ncpus: Number of different CPUs used in the generation process ( currently not used.)
"""
def gen_fault_collection(ctx, pruning_cfg = None, feat_repr = "f32", working_mode = 0, error_margin = 0.01, confidence_level = 0.99, individual_prob = 0.5, out_dir = "./", ncpus = 1):
    # Initialize the logger 
    logger = logging.getLogger("pyALS-RF")
    logger.info("Generating the fault collection.")
    load_configuration_ps(ctx)
    create_classifier(ctx)    
    classifier = ctx.obj["classifier"]
    if pruning_cfg is not None:
        with open(pruning_cfg, "r") as f:
            pruning_readed = json5.load(f)
        GREP.set_pruning_conf(classifier = classifier, pruning_conf = pruning_readed)
    # Generate the fault collection
    f = FaultCollection(classifier, feat_repr = feat_repr)
    # Distinct the two working modes.
    if working_mode == 0: # Sample from the entire fault universe
        f.sample_faults(type_of_faults = 0, error_margin = error_margin, confidence_level = confidence_level, individual_prob = individual_prob)
    elif working_mode == 1: # Sample from different fault universes
        f.sample_faults(type_of_faults = 1, error_margin = error_margin, confidence_level = confidence_level, individual_prob = individual_prob)
        f.sample_faults(type_of_faults = 2, error_margin = error_margin, confidence_level = confidence_level, individual_prob = individual_prob)
        f.sample_faults(type_of_faults = 3, error_margin = error_margin, confidence_level = confidence_level, individual_prob = individual_prob)
    elif working_mode >= 2 and working_mode <= 6:
        f.sample_faults(type_of_faults = working_mode - 1, error_margin = error_margin, confidence_level = confidence_level, individual_prob = individual_prob)
    else:
        assert 1 == 0, "Invalid configuration of the fault parameter"
    f.faults_to_json5_list(classifier = classifier,  out_path = out_dir)

# Execute a faulted visit for each different fault in the folder input_faults.
# Classess probability vectors are saved into the output dir specified by the 
# "output" parameter in input. Each file (1 for each category of fault so 3 different)
# files, containts a list, for each fault for type in input fault, of the probabilities vector.
# i.e. for 10 feature faults and 50 input samples the feat out file contains 50 vector for each different
# fault (i.e. 10 different features.)
# The input faults is a list containing for each fault different features used during experiments. 
def fault_visit(ctx, output, input_faults, samples_idx,  ncpus, num_samples = 50, pruning_cfg = None, nabs = None):
    assert not (nabs != None and pruning_cfg != None), "Mode not supported, use just one axc technique!"
    # Initialize the logger 
    logger = logging.getLogger("pyALS-RF")
    logger.info("Runing the Fault Visit.")
    logger.info(f"Output: {output}")
    logger.info(f"Input Faults: {input_faults}")
    logger.info(f"Path of samples {samples_idx}")
    logger.info(f"Pruning cfg: {pruning_cfg}")
    logger.info(f"NABS : {nabs}")
    if nabs is not None:    # For NABS this function is called only inside the other functions
        load_configuration_ps(ctx)
        create_classifier(ctx)    
    classifier = ctx.obj["classifier"]
    # Read the pruning_cfg to approximate the classifier.
    if pruning_cfg is not None:
        with open(pruning_cfg, "r") as f:
            pruning_readed = json5.load(f)
        GREP.set_pruning_conf(classifier = classifier, pruning_conf = pruning_readed)
    elif nabs is not None:
        classifier.set_nabs(nabs) # REMEMBER THAT THE FUNCTION DOES NOT RESET THE NABS !
    # Read the test set.
    if samples_idx is not None:
        with open(samples_idx, "r") as file:
            indexes = json5.load(file)
        x_test = copy.deepcopy(classifier.x_test[indexes])
        y_test = copy.deepcopy(classifier.y_test[indexes])
    else:
        x_test = copy.deepcopy(classifier.x_test[0 : num_samples])
        y_test = copy.deepcopy(classifier.y_test[0 : num_samples])
    original_pred = classifier.predict(x_test)
    """ ************************************************ """
    # Feature Faults
    feat_path   = os.path.join(input_faults, "feat_faults.json5") 
    out_path_vectors    = os.path.join(output, "feat_faults_vectors.json5")
    # Load the feat JSON5 file
    with open(feat_path, "r") as file:
        loaded_faults = json5.load(file)
    nro_feat_faults = len(loaded_faults)
    x_test_temp = x_test
    fault_vect_list  = []
    # For each fault
    for f in tqdm(loaded_faults, desc = "Visiting with feature faults"):
        # Inject faults into the inputs
        inject_fault_input(classifier = classifier, faults = [f], x_test = x_test_temp)
        # Visit
        faulted_vec = classifier.predict(x_test_temp, disable_tqdm = True)
        # Save values 
        fault_vect_list.append(faulted_vec.tolist())
        # # Restore, x_test_temps mantaints at each cycle always the pointer to the original array.
        x_test_temp = copy.deepcopy(x_test)
    # Save to json5 the list of vectors
    with open(out_path_vectors, "w") as f:
        json5.dump(fault_vect_list, f, indent = 2)
    if len(fault_vect_list) > 0:
        feat_perc_detected, feat_perc_crit, feat_list_prob_det, feat_list_prob_crit, feat_ctr_det, feat_ctr_crit = compute_faults_prob(original_pred = original_pred, fault_vect_list = fault_vect_list)
    else:
        feat_perc_detected, feat_perc_crit, feat_list_prob_det, feat_list_prob_crit, feat_ctr_det, feat_ctr_crit = 0,0,0,0,0,0
    
    """ ************************************************ """
    # For DBs faults.
    dbs_path   = os.path.join(input_faults, "dbs_faults.json5") 
    out_path_vectors    = os.path.join(output, "dbs_faults_vectors.json5")
    fault_vect_list = []
    # Load the feat JSON5 file
    with open(dbs_path, "r") as file:
        loaded_faults = json5.load(file)
    nro_dbs_faults = len(loaded_faults)
    # For each fault
    for f in tqdm(loaded_faults, desc = "Visiting with DBs faults"):
        # Inject faults into the inputs
        old_dbs = classifier.inject_tree_boxes_faults_fb(f)
        # Visit
        faulted_vec = classifier.predict(x_test, disable_tqdm = True)
        # Save values 
        fault_vect_list.append(faulted_vec.tolist())
        # Restore
        classifier.restore_dbs(old_dbs)
    
    # Save to json5 the list of vectors
    with open(out_path_vectors, "w") as f:
        json5.dump(fault_vect_list, f, indent = 2)
    if len(fault_vect_list) > 0:
        dbs_perc_detected, dbs_perc_crit, dbs_list_prob_det, dbs_list_prob_crit, dbs_ctr_det, dbs_ctr_crit = compute_faults_prob(original_pred = original_pred, fault_vect_list = fault_vect_list)
    else:
        dbs_perc_detected, dbs_perc_crit, dbs_list_prob_det, dbs_list_prob_crit, dbs_ctr_det, dbs_ctr_crit = 0,0,0,0,0,0
    """ ************************************************ """
    # For BNs faults
    bns_path   = os.path.join(input_faults, "bns_faults.json5") 
    out_path_vectors    = os.path.join(output, "bns_faults_vectors.json5")
    fault_vect_list = []
    # Load the feat JSON5 file
    with open(bns_path, "r") as file:
        loaded_faults = json5.load(file)
    nro_bns_faults = len(loaded_faults)
    old_bns = classifier.store_bns()
    # For each fault
    for f in tqdm(loaded_faults, desc = "Visiting with BNs faults"):
        # Inject faults into the inputs
        classifier.inject_bns_faults(f)
        # Visit
        faulted_vec = classifier.predict(x_test, disable_tqdm = True)
        # Save values 
        fault_vect_list.append(faulted_vec.tolist())
        # Restore
        classifier.restore_bns(old_bns)
    # Save to json5 the list of vectors
    with open(out_path_vectors, "w") as f:
        json5.dump(fault_vect_list, f, indent = 2)
    bns_perc_detected, bns_perc_crit, bns_list_prob_det, bns_list_prob_crit, bns_ctr_det, bns_ctr_crit = compute_faults_prob(original_pred = original_pred, fault_vect_list = fault_vect_list)
    if len(fault_vect_list) > 0:
        bns_perc_detected, bns_perc_crit, bns_list_prob_det, bns_list_prob_crit, bns_ctr_det, bns_ctr_crit = compute_faults_prob(original_pred = original_pred, fault_vect_list = fault_vect_list)
    else:
        bns_perc_detected, bns_perc_crit, bns_list_prob_det, bns_list_prob_crit, bns_ctr_det, bns_ctr_crit = 0, 0, 0, 0, 0, 0
    logger.info(f"FEAT. DET:  {feat_perc_detected} CRIT: {feat_perc_crit}")
    logger.info(f"DBS. DET:  {dbs_perc_detected} CRIT: {dbs_perc_crit}")
    logger.info(f"BNS: DET:  {bns_perc_detected} CRIT: {bns_perc_crit}")
    summary_path = os.path.join(output, "summary.json5")
    with open(summary_path, "w") as f:
        json5.dump(
            {
                # Percentage per fault.
                "Feat_Perc_Det" : feat_perc_detected,
                "Feat_Perc_Crit" : feat_perc_crit,
                "DBS_Perc_Det" :    dbs_perc_detected,
                "DBS_Perc_Crit" : dbs_perc_crit,
                "BNS_Perc_Det" :  bns_perc_detected,
                "BNS_Perc_Crit" : bns_perc_crit,
                # Counters and number of injected faults.
                "Feat_Ctr_Det" : feat_ctr_det,
                "Feat_Ctr_Crit": feat_ctr_crit,
                "DBS_Ctr_Det"  : dbs_ctr_det,
                "DBS_Ctr_Crit" : dbs_ctr_crit,
                "BNS_Ctr_Det"  : bns_ctr_det,
                "BNS_Ctr_Crit" : bns_ctr_crit,
                "Nro_Feat_Inj" : nro_feat_faults,
                "Nro_DBS_Inj"  : nro_dbs_faults,
                "Nro_BNS_Inj"  : nro_bns_faults,
                # Probabilities per entire test sample...
                "Feat_Mean_Det_Prob" : np.mean(feat_list_prob_det),
                "DBS_Mean_Det_Prob" : np.mean(dbs_list_prob_det),
                "BNS_Mean_Det_Prob" : np.mean(bns_list_prob_det),
                "Feat_Mean_Crit_Prob" : np.mean(feat_list_prob_crit),
                "DBS_Mean_Crit_Prob" : np.mean(dbs_list_prob_crit),
                "BNS_Mean_Crit_Prob" : np.mean(bns_list_prob_crit),
                
                "Feat_Mean_Det_List" : feat_list_prob_det,
                "DBS_Mean_Det_List" : dbs_list_prob_det,
                "BNS_Mean_Det_List" : bns_list_prob_det,
                "Feat_Mean_Crit_List" : feat_list_prob_crit,
                "DBS_Mean_Crit_List" : dbs_list_prob_crit,
                "BNS_Mean_Crit_List" : bns_list_prob_crit
            },
            f,
            indent = 2
        )

# Function used to dump class probabilities vectors without any fault. 
def dump_unfaulted_class_vector(ctx, output, test_idx, pruning_conf, ncpus, num_samples = 50):
    # Initialize the logger 
    logger = logging.getLogger("pyALS-RF")
    logger.info("Runing the TMR flow.")
    load_configuration_ps(ctx)
    create_classifier(ctx)    
    classifier = ctx.obj["classifier"]
    # Apply the pruning
    if pruning_conf is not None:
        with open(pruning_conf, "r") as f:
            pruning_readed = json5.load(f)
        GREP.set_pruning_conf(classifier = classifier, pruning_conf = pruning_readed)
    # Select the test idx
    if test_idx is None:
        x_test = copy.deepcopy(classifier.x_test[0 : num_samples])
    else: 
        with open(test_idx, "r") as f :
            idx = json5.load(f)
        x_test = copy.deepcopy(classifier.x_test[idx])
    
    vectors = classifier.predict(x_test)
    vectors = vectors.tolist()
    # Save to json5 the list of vectors
    with open(os.path.join(output, "class_vec_no_faults.json5"), "w") as file:
        json5.dump(vectors, file, indent = 2)

""" Test function used during development. """

def test_bns_faulted_inference(ctx, conf, input_faults, num_samples = 50):
    # Initialize the logger 
    logger = logging.getLogger("pyALS-RF")
    logger.info("Runing the TMR flow.")
    load_configuration_ps(ctx)
    create_classifier(ctx)    
    classifier = ctx.obj["classifier"]
    x_test = copy.deepcopy(classifier.x_test[0 : num_samples])
    y_test = copy.deepcopy(classifier.y_test[0 : num_samples])
    
    path   = os.path.join(input_faults, "bns_faults.json5") 
    #out_path_vectors    = os.path.join(output, "dbs_faults_vectors.json5")
    fault_vect_list = []
    # Load the feat JSON5 file
    with open(path, "r") as file:
        loaded_faults = json5.load(file)
    original_pred = classifier.predict(x_test, disable_tqdm = True)
    old_bns = classifier.store_bns()
    # For each fault
    for f in tqdm(loaded_faults, desc = "Visiting with BNs faults"):
        # Inject faults into the inputs
        classifier.inject_bns_faults(f)
        # Visit
        faulted_vec = classifier.predict(x_test, disable_tqdm = True)
        # Save values 
        fault_vect_list.append(faulted_vec.tolist())
        # count_neq = 0
        # for (x,y) in zip(faulted_vec, original_pred):
        #     # print(f"Original {y}")
        #     # print(f"Faulted {x}")
        #     if not np.array_equal(x,y):
        #         count_neq += 1
        # #print(f"Neq {count_neq} Eq {len(original_pred) - count_neq}")
        # # Restore
        classifier.restore_bns(old_bns)
        nv = classifier.predict(x_test, disable_tqdm = True)
        for x,y in zip(original_pred, nv):
            if not np.array_equal(x,y):
                print(f"Restoring NOT WORKING")
                exit(1)
                assert 1 == 0
        # temp_vec = classifier.predict(x_test, disable_tqdm = True)
        # is_ok = True
        # for x,y in zip(temp_vec, original_pred):
        #     if not np.array_equal(x,y):
        #         is_ok = False
        #         break 
        # if not is_ok:
        #     logger.info("ERROR IN RESTORING")
        #     exit(1)
        # else:
        #     logger.info("RESTORE IS OK. W IL MARSUPIO")
        #exit(1)
    bns_perc_detected, bns_perc_crit, bns_list_prob_det, bns_list_prob_crit, bns_ctr_detected, bns_ctr_crit = compute_faults_prob(original_pred = original_pred, fault_vect_list = fault_vect_list)

    logger.info(f"The Percentage of detected faults is {bns_perc_detected}")
    logger.info(f"The Percentage of critical faults is {bns_perc_crit}")
    return 0



def test_dbs_faulted_inference(ctx, conf, input_faults, num_samples = 50):
    # Initialize the logger 
    logger = logging.getLogger("pyALS-RF")
    logger.info("Runing the TMR flow.")
    load_configuration_ps(ctx)
    create_classifier(ctx)    
    classifier = ctx.obj["classifier"]
    x_test = copy.deepcopy(classifier.x_test[0 : num_samples])
    y_test = copy.deepcopy(classifier.y_test[0 : num_samples])
    
    # For DBs faults.
    dbs_path   = os.path.join(input_faults, "dbs_faults.json5") 
    #out_path_vectors    = os.path.join(output, "dbs_faults_vectors.json5")
    fault_vect_list = []
    # Load the feat JSON5 file
    with open(dbs_path, "r") as file:
        loaded_faults = json5.load(file)
    original_pred = classifier.predict(x_test, disable_tqdm = True)

    # For each fault
    for f in tqdm(loaded_faults, desc = "Visiting with DBs faults"):
        # Inject faults into the inputs
        old_dbs = classifier.inject_tree_boxes_faults_fb(f)
        # Visit
        faulted_vec = classifier.predict(x_test, disable_tqdm = True)
        # Save values 
        fault_vect_list.append(faulted_vec)
        # Restore
        classifier.restore_dbs(old_dbs)
        temp_vec = classifier.predict(x_test, disable_tqdm = True)
        is_ok = True
        for x,y in zip(temp_vec, original_pred):
            if not np.array_equal(x,y):
                is_ok = False
                break 
        if not is_ok:
            logger.info("ERROR IN RESTORING")
            exit(1)
        else:
            logger.info("RESTORE IS OK. W IL MARSUPIO")
    detected = 0
    critical = 0
    for faulted_vec in fault_vect_list:
        to_detect = True 
        to_add_crit = True
        for o, v in zip(original_pred, faulted_vec):
            if not np.array_equal(o, v):
                if to_detect:
                    detected += 1
                    to_detect = False
                if np.argmax(o) != np.argmax(v):
                    if to_add_crit :
                        critical += 1
                        to_add_crit = False
    logger.info(f"The Percentage of detected faults is {detected / len(fault_vect_list)}")
    logger.info(f"The Percentage of critical faults is {critical / len(fault_vect_list)}")
    return 0

def compute_faults_prob(original_pred, fault_vect_list):
    detected = 0
    critical = 0
    list_prob_det = []
    list_prob_crit = []
    # Initialize the logger 
    for faulted_vec in fault_vect_list:
        to_detect = True 
        to_add_crit = True
        counter_prob_det = 0
        counter_prob_crit = 0
        for o, v in zip(original_pred, faulted_vec):
            if not np.array_equal(o, v):
                if to_detect:
                    detected += 1
                    to_detect = False
                counter_prob_det += 1
                if np.argmax(o) != np.argmax(v):
                    if to_add_crit :
                        critical += 1
                        to_add_crit = False
                    counter_prob_crit += 1
        list_prob_det.append(counter_prob_det / len(original_pred))
        list_prob_crit.append(counter_prob_crit / len(original_pred))

    return detected / len(fault_vect_list), critical / len(fault_vect_list), list_prob_det, list_prob_crit, detected, critical


def sample_test_set(ctx, perc, num,  out, ncpus):
    load_configuration_ps(ctx)
    create_classifier(ctx)    
    classifier = ctx.obj["classifier"]
    idx = [ i for i in range(0, len(classifier.x_test))]
    if perc is not None:
        split = int(len(classifier.x_test) * perc)
    else: 
        split = num
    samples =  random.sample(idx, split)
    path = os.path.join(out, "Test_Samples_FI.json5")
    with open(path, "w") as file:
        json5.dump(samples, file, indent = 2)

def gen_fault_coll_ps(ctx, ps_dir, val_path, working_mode = 0, error_margin = 0.01, confidence_level = 0.95, individual_prob = 0.5, out_dir = "./", ncpus = 1):
    logger = logging.getLogger("pyALS-RF")
    logger.info("Runing the TMR flow.")
    load_configuration_ps(ctx)
    create_classifier(ctx)   

    classifier = ctx.obj["classifier"]
    if ps_dir is not None:
        ps_dir_cfg = ps_dir
    else:
        ps_dir_cfg = ctx.obj["configuration"].outdir    
    
    with open(ps_dir_cfg, "r") as file:
        raw_content = file.read()
    corrected_content = raw_content.replace("{x:", '{"x":').replace(", f:", ', "f":').replace(", g:", ', "g":')
    
    # Convertire la stringa corretta in un dizionario Python
    try:
        data = json.loads(corrected_content)
    except json.JSONDecodeError as e:
        assert 1 == 0, "Unable to correct the errors in json5" 
    
    if val_path != None:
        validation_indexes = np.loadtxt(val_path, dtype = int)
        validation_samples_x = [classifier.x_test[i] for i in validation_indexes]
        validation_samples_y = [classifier.y_test[i] for i in validation_indexes]
        logger.info(f"Validating accuracy on new samples")
        base_acc =  classifier.evaluate_accuracy(validation_samples_x, validation_samples_y, disable_tqdm = False)
        logger.info(f"Baseline accuracy {base_acc}")
        acc_lossess = []
        nabs_array = []
        savings = []
        # Order each configuration by accuracy loss.
        for solution in tqdm(data, desc = "Evaluating accuracy on the test set"):
            nabs = solution["x"]
            savings.append(int(solution["f"][1]))
            nabs_array.append(nabs)
            nabs_dict = {f["name"]: n for f, n in zip(classifier.model_features, nabs[:len(classifier.model_features)])}
            classifier.set_nabs(nabs_dict)
            accuracy = classifier.evaluate_accuracy(validation_samples_x, validation_samples_y, disable_tqdm = True)
            loss = base_acc - accuracy
            logger.info(f"Accuracy for new solution {accuracy} Loss: {loss}")
            acc_lossess.append(loss)
            classifier.reset_nabs_configuration()
        sorted_idx = np.argsort(acc_lossess)
        min_loss_idx = np.argmin(acc_lossess)
        #loss_ranges = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        ranges_values = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 5.0)]
        range_full    = [False for r in range(len(ranges_values))]
        selected_lossess = 0
        break_outer = False
        loss_array_sel = []
        for loss in sorted_idx:
            for range_id in range(0,len(ranges_values)):
                if acc_lossess[loss] >= ranges_values[range_id][0] and  acc_lossess[loss] < ranges_values[range_id][1] and not range_full[range_id] :
                    range_full[range_id] = True
                    selected_lossess += 1
                    loss_array_sel.append(loss)
                if selected_lossess == len(ranges_values):
                    break_outer = True
                    break
            if break_outer :
                break    
        found_lossess = [acc_lossess[sel] for sel in loss_array_sel]
        found_nabs = [nabs_array[sel] for sel in loss_array_sel] 
        found_savings = [savings[sel] for sel in loss_array_sel]

    else:
        # Simply load the first term.
        found_lossess = []
        found_nabs = []
        found_savings = []
        for solution  in tqdm(data, desc = "Evaluating accuracy on the test set"):
            found_lossess.append(int(solution["f"][0]))
            found_savings.append(int(solution["f"][1]))
            found_nabs.append(solution["x"])
    # Generate an index containing the configuration of each approximated variant
    index_list = []
    cfg_paths = []
    index_path = os.path.join(out_dir, "index.json5")
    for l,n, sav in zip(found_lossess, found_nabs, found_savings):
        # The out dir of each configuration is in the same folder of the index file
        cpath = os.path.join(out_dir, f"cfg_{l:.2f}")
        if not os.path.exists(cpath):
            os.makedirs(cpath)
        cfg_paths.append(cpath)
        index_list.append({ "Loss": l, "Nab" : n, "Out:": cpath , "Savings" : sav})
    with open(index_path, "w") as f:
        json5.dump(index_list, f ,indent = 2)
    # Generate the configuration for each NAB
    for l,n,cp in tqdm(zip(found_lossess, found_nabs, cfg_paths), desc = "Sampling and Saving Fault Configurations"):
        fc = FaultCollection(classifier, n)
        if working_mode == 0: # Sample from the entire fault universe
            fc.sample_faults(type_of_faults = 0, error_margin = error_margin, confidence_level = confidence_level, individual_prob = individual_prob)
        elif working_mode == 1: # Sample from different fault universes
            fc.sample_faults(type_of_faults = 1, error_margin = error_margin, confidence_level = confidence_level, individual_prob = individual_prob)
            fc.sample_faults(type_of_faults = 2, error_margin = error_margin, confidence_level = confidence_level, individual_prob = individual_prob)
            fc.sample_faults(type_of_faults = 3, error_margin = error_margin, confidence_level = confidence_level, individual_prob = individual_prob)
        elif working_mode >= 2 and working_mode < 5:
            fc.sample_faults(type_of_faults = working_mode - 1, error_margin = error_margin, confidence_level = confidence_level, individual_prob = individual_prob)
        else:
            assert 1 == 0, "Invalid configuration of the fault parameter"
        fc.faults_to_json5_list(classifier = classifier,  out_path = cp)
        

def ps_faultinj_visit(ctx, output, ps_index, input_faults, samples_idx,  ncpus, num_samples = 50):
    load_configuration_ps(ctx)
    create_classifier(ctx)    
    classifier = ctx.obj["classifier"]
    logger = logging.getLogger("pyALS-RF")
    logger.info("Runing the Fault Visit with PRECISION SCALING")
    logger.info(f"Ps index {ps_index}")
    with open(ps_index, "r") as f:
        ps_cfgs = json5.load(f)
    logger.info("Executing Faulted visits.")
    for pscfg in ps_cfgs:
        nabs = [int(nb) for nb in pscfg["Nab"]]
        nabs_dict = {f["name"]: n for f, n in zip(classifier.model_features, nabs[:len(classifier.model_features)])}
        ipt_faults = pscfg["Out:"] # A little bit tired of writing functions... so direct embedding :)
        out_dir = pscfg["Out:"]
        fault_visit(ctx, out_dir, ipt_faults, samples_idx,  ncpus, num_samples, pruning_cfg = None, nabs = nabs_dict) # Tooo tired to think about reenginering.
        classifier.reset_nabs_configuration()


def compute_probabilities(outcome):
    """Convert tree votes into probabilities."""
    outcome = np.array(outcome, dtype=float)
    total = outcome.sum()
    assert total > 0, "Outcome vector cannot be all zeros!"
    return outcome / total


def margin_from_votes(outcome, y_true):
    """
    p(y_true) - max_{c != y_true} p(c)
    """
    probs = compute_probabilities(outcome)
    p_true = probs[y_true]
    # Compute the class as a probability detector.
    p_best_rival = np.max(np.delete(probs, y_true))

    return p_true - p_best_rival

def l2_error_norm(probs, y_true):
    """
    Compute the L2 error norm between predicted probabilities
    and the one-hot vector of the true class.
    
    outcome: vettore di voti degli alberi (o probabilità)
    y_true: indice della classe vera
    """

    # One-hot vector
    t = np.zeros_like(probs)
    t[y_true] = 1.0

    # L2 norm of (p - t)
    err = np.linalg.norm(probs - t, ord=2)

    return float(err)

def margin(probs, y_true):
    """
    p(y_true) - max_{c != y_true} p(c)
    """
    p_true = float(probs[y_true])
    # Compute the class as a probability detector.
    p_best_rival = float(np.max(np.delete(probs, y_true)))

    return p_true - p_best_rival


def cross_entropy_from_votes(outcome):
    """
    Cross Entropy: 
    H = - sum p_k * log(p_k)
    """
    probs = compute_probabilities(outcome)
    
    # Avoid log(0)
    nonzero = probs[probs > 0]

    return -np.sum(nonzero * np.log(nonzero))

def cross_entropy(probs):
    # Avoid log(0)
    nonzero = probs[probs > 0]
    return -np.sum(nonzero * np.log(nonzero))

def gini_index_from_votes(outcome):
    """
    Gini Index: G = 1 - sum p_k^2
    """
    probs = compute_probabilities(outcome)
    return 1 - np.sum(probs ** 2)

def gini_index(probs):
    return 1 - np.sum(probs ** 2)

def sample_ranking(ctx, conf, quantization_type, ncpus, out_path):
    load_configuration_ps(ctx)
    create_classifier(ctx)    
    
    classifier: Classifier = ctx.obj["classifier"]
    if quantization_type != None:
        classifier.set_thds_type(quantization_type)

    logger = logging.getLogger("pyALS-RF")
    logger.info("Runing the Sample Ranking command.")

    # Perform sampling on the values.
    pvs = classifier.predict_many(classifier.x_test)
    
    # Per sample value of margin, gini, CE.
    margins = []
    margins_abs = []
    gini = []
    cross_entropy_vals = []
    l2_norms = []
    for pv, y_true in tqdm(zip(pvs, classifier.y_test), 
                       total=len(classifier.y_test), 
                       desc="Computing metrics"):
        probs = compute_probabilities(pv)
        mx = margin(probs, y_true)
        margins.append(mx)
        margins_abs.append(abs(mx))
        gini.append(gini_index(probs))
        ce = cross_entropy(probs)
        cross_entropy_vals.append(0.0 if abs(ce) < 1e-12 else ce)
        l2_norms.append(l2_error_norm(probs, y_true))

    
    # Sort prediction indexes.
    marginabs_indices_sorted = np.argsort(margins_abs)
    gini_indices_sorted = np.argsort(gini)[::-1]
    entropy_indices_sorted = np.argsort(cross_entropy_vals)[::-1]
    l2_norms_indices_sorted = np.argsort(l2_norms)[::-1]

    logger.info("Dumping indexes...")
    margin_idx_path = os.path.join(out_path, "margin_idx_sorted.txt")
    gini_idx_path = os.path.join(out_path, "gini_idx_sorted.txt")
    entropy_idx_path = os.path.join(out_path, "ce_idx_sorted.txt")
    l2_norm_paths = os.path.join(out_path, "l2_idx_sorted.txt")

    np.savetxt(margin_idx_path, marginabs_indices_sorted, fmt="%d")
    np.savetxt(gini_idx_path, gini_indices_sorted, fmt="%d")
    np.savetxt(entropy_idx_path, entropy_indices_sorted, fmt="%d")
    np.savetxt(l2_norm_paths, l2_norms_indices_sorted, fmt="%d")

    logger.info("Dumping values...")
    margin_val_path = os.path.join(out_path, "margin_abs_vals.txt")
    margin_val2_path = os.path.join(out_path, "margin_vals.txt")
    gini_val_path = os.path.join(out_path, "gini_vals.txt")
    entropy_val_path = os.path.join(out_path, "CE_vals.txt")
    l2_val_path = os.path.join(out_path, "l2_vals.txt")
    np.savetxt(margin_val_path, margins_abs, fmt="%6f")
    np.savetxt(margin_val2_path, margins, fmt="%6f")
    np.savetxt(gini_val_path, gini, fmt="%6f")
    np.savetxt(entropy_val_path, cross_entropy_vals, fmt="%6f")
    np.savetxt(l2_val_path, l2_norms, fmt="%6f")
    

def relate_faults_to_input(original_pred, faulted_pred):
    detected = 0
    critical = 0
    # Initialize an empty list of faults detected for each different sample.
    faults_per_sample_det = [ [] for _ in original_pred]
    faults_per_sample_crit = [ [] for _ in original_pred]
    # For each fault list.
    for fault_idx, faulted_vec in enumerate(faulted_pred):
        # Flag usced to detect a critical and non-critical fault
        # Indeed they must be detected only once !
        to_detect = True
        to_detect_crit = True
        # For each prediction
        sample_cnt = 0
        for o,f in zip(original_pred, faulted_vec):
            if not np.array_equal(o, f):
                # Increase the counter of detected faults.
                if to_detect:
                    detected += 1
                    to_detect = False
                # Append to the list of faults detected for each sample, the 
                faults_per_sample_det[sample_cnt].append(fault_idx)
                # If it is a critical fault
                if np.argmax(o) != np.argmax(f):
                    if to_detect_crit :
                        to_detect_crit = False
                        critical += 1
                    # In any case, append to the list of faults detected for each different sample.
                    faults_per_sample_crit[sample_cnt].append(fault_idx)
            sample_cnt += 1
    
    return detected, critical, faults_per_sample_det, faults_per_sample_crit 

def fault_visit_exhaustive( ctx, 
                            output, 
                            feat_repr, input_faults_path,  
                            ncpus = os.cpu_count()):
    
    # assert samples_idx_path is not None, "The command requires a samples path."
    assert input_faults_path is not None, "The command requires input fauls."
    load_configuration_ps(ctx)
    create_classifier(ctx)   
    
    # Initialize the logger 
    logger = logging.getLogger("pyALS-RF")
    logger.info("Runing the Fault Visit.")
    logger.info(f"Output: {output}")
    logger.info(f"Quantization CFG Path: {feat_repr}")
    logger.info(f"Input Faults: {input_faults_path}")
    logger.info(f"CPU Count {ncpus}")
    
    classifier : Classifier = ctx.obj["classifier"]
    # Alter the feature representation of the model.
    if feat_repr != None:
        classifier.set_thds_type(feat_repr)
        classifier.set_samples_repr(feat_repr)
    else:
        classifier.set_thds_type("f32")
        classifier.set_samples_repr("f32")
    
    logger.info("Inizializing predictions of the original ensemble.")
    original_pred_vecs = classifier.predict_many(classifier.x_test)
    # cp = 0
    # for p, y in zip(original_pred_vecs, classifier.y_test):
    #     if np.argmax(p)  == y and not classifier.check_draw(p)[0]:
    #         cp += 1
    # logger.info(f"CP: {cp} {(cp / len(classifier.y_test)) * 100.0}")
    # exit()
    
    """ ************************************************ """
    # Feature Faults
    feat_path   = os.path.join(input_faults_path, "feat_faults.json5") 
    # Load the feat JSON5 file
    with open(feat_path, "r") as file:
        loaded_faults = json5.load(file)
    x_test_temp = copy.deepcopy(classifier.x_test)
    fault_vect_list  = []
    nro_feat_faults = len(loaded_faults)
    # For each fault
    for f in tqdm(loaded_faults, desc = "Visiting with feature faults"):
        # Inject faults into the inputs
        inject_fault_input(classifier = classifier, faults = [f], x_test = x_test_temp, feat_repr= feat_repr)
        # Visit
        faulted_vec = classifier.predict_many(x_test_temp)
        # Save values 
        fault_vect_list.append(faulted_vec.tolist())
        # # Restore, x_test_temps mantaints at each cycle always the pointer to the original array.
        x_test_temp = copy.deepcopy(classifier.x_test)
    
    
    if len(fault_vect_list) > 0:
        feat_detected_cnt, feat_critical_cnt, feat_per_sample_det_faults, feat_per_sample_crit_faults = relate_faults_to_input(original_pred=original_pred_vecs, faulted_pred= fault_vect_list)
        feat_det_perc = feat_detected_cnt / len(fault_vect_list) * 100
        feat_crit_perc = feat_critical_cnt / len(fault_vect_list) * 100
        # feat_perc_detected, feat_perc_crit, _, _, feat_ctr_det, feat_ctr_crit = compute_faults_prob(original_pred = original_pred_vecs, fault_vect_list = fault_vect_list)

    else:
        feat_detected_cnt, feat_critical_cnt, feat_per_sample_det_faults, feat_per_sample_crit_faults = 0, 0, [], []
        feat_det_perc = 0.0 
        feat_crit_perc = 0.0 
    
    logger.info(f"[FEAT] Faults Detected: {feat_detected_cnt} PERC: {feat_det_perc}")
    logger.info(f"[FEAT] Critical faults detected: {feat_critical_cnt} PERC: {feat_crit_perc}")

    
    # """ ************************************************ """
    # For DBs faults.
    dbs_path   = os.path.join(input_faults_path, "dbs_faults.json5") 
    fault_vect_list = []
    # Load the feat JSON5 file
    with open(dbs_path, "r") as file:
        loaded_faults = json5.load(file)
    nro_dbs_faults = len(loaded_faults)
    
    
    # For each fault
    for f in tqdm(loaded_faults, desc = "Visiting with DBs faults"):    
        # Inject faults into the inputs
        old_dbs = classifier.inject_tree_boxes_faults_fb(f)
        # Visit
        faulted_vec = classifier.predict_many(classifier.x_test)
        # Save values 
        fault_vect_list.append(faulted_vec.tolist())
        # Restore
        classifier.restore_dbs(old_dbs)

    if len(fault_vect_list) > 0:
        dbs_detected_cnt, dbs_critical_cnt, dbs_per_sample_det_faults, dbs_per_sample_crit_faults = relate_faults_to_input(original_pred=original_pred_vecs, faulted_pred= fault_vect_list)
        dbs_det_perc = dbs_detected_cnt / len(fault_vect_list) * 100
        dbs_crit_perc = dbs_critical_cnt / len(fault_vect_list) * 100
        # dbs_perc_detected, dbs_perc_crit, _, _, dbs_ctr_det, dbs_ctr_crit = compute_faults_prob(original_pred = original_pred_vecs, fault_vect_list = fault_vect_list)

    else:
        dbs_detected_cnt, dbs_critical_cnt, dbs_per_sample_det_faults, dbs_per_sample_crit_faults = 0, 0, [], []
        dbs_det_perc = 0.0
        dbs_crit_perc = 0.0
    
    logger.info(f"[DBS] Faults Detected: {dbs_detected_cnt} PERC: {dbs_det_perc}")
    logger.info(f"[DBS] Critical faults detected: {dbs_critical_cnt} PERC: {dbs_crit_perc}")
    
    # exit()
    """ ************************************************ """
    # For BNs faults
    bns_path   = os.path.join(input_faults_path, "bns_faults.json5") 
    fault_vect_list = []
    # Load the feat JSON5 file
    with open(bns_path, "r") as file:
        loaded_faults = json5.load(file)
    nro_bns_faults = len(loaded_faults)
    old_bns = classifier.store_bns()
    # For each fault
    for f in tqdm(loaded_faults, desc = "Visiting with BNs faults"):
        # Inject faults into the inputs
        classifier.inject_bns_faults(f)
        # Visit
        faulted_vec = classifier.predict_many(classifier.x_test)
        # Save values 
        fault_vect_list.append(faulted_vec.tolist())
        # Restore
        classifier.restore_bns(old_bns)
    
    if len(fault_vect_list) > 0:
        bns_detected_cnt, bns_critical_cnt, bns_per_sample_det_faults, bns_per_sample_crit_faults = relate_faults_to_input(original_pred=original_pred_vecs, faulted_pred= fault_vect_list)
        bns_det_perc = bns_detected_cnt / len(fault_vect_list) * 100
        bns_crit_perc = bns_critical_cnt / len(fault_vect_list) * 100
        # bns_perc_detected, bns_perc_crit, _, _, bns_ctr_det, bns_ctr_crit = compute_faults_prob(original_pred = original_pred_vecs, fault_vect_list = fault_vect_list)

    else:
        bns_detected_cnt, bns_critical_cnt, bns_per_sample_det_faults, bns_per_sample_crit_faults = 0, 0, [], []
        bns_det_perc = 0.0
        bns_crit_perc = 0.0

    logger.info(f"[BNS] Faults Detected: {bns_detected_cnt} PERC: {bns_det_perc}")
    logger.info(f"[BNS] Critical faults detected: {bns_critical_cnt} PERC: {bns_crit_perc}")

    """ ************************************************ """
    # For THDs faults.
    thd_path   = os.path.join(input_faults_path, "thd_faults.json5") 
    fault_vect_list = []
    # Load the feat JSON5 file
    with open(thd_path, "r") as file:
        loaded_faults = json5.load(file)
    nro_thds_faults = len(loaded_faults)
    
    # For each fault
    for f in tqdm(loaded_faults, desc = "Visiting with ThDs faults"):
        # Inject faults into the inputs
        old_dbs = classifier.inject_tree_thds_faults(f)
        # Visit
        faulted_vec = classifier.predict_many(classifier.x_test)
        # Save values 
        fault_vect_list.append(faulted_vec.tolist())
        # Restore
        classifier.restore_thds(old_dbs)

        # tv = classifier.predict_many(classifier.x_test)
        # thds_detected_cnt, thds_critical_cnt, thds_per_sample_det_faults, thds_per_sample_crit_faults = relate_faults_to_input(original_pred=original_pred_vecs, faulted_pred= [tv])
        # if thds_detected_cnt > 0:
        #     assert 1 == 0, "Restoring not working"

    if len(fault_vect_list) > 0:
        thds_detected_cnt, thds_critical_cnt, thds_per_sample_det_faults, thds_per_sample_crit_faults = relate_faults_to_input(original_pred=original_pred_vecs, faulted_pred= fault_vect_list)
        thds_det_perc = thds_detected_cnt / len(fault_vect_list) * 100
        thds_crit_perc = thds_critical_cnt / len(fault_vect_list) * 100
        # thds_perc_detected, thds_perc_crit, _, _, thds_ctr_det, thds_ctr_crit = compute_faults_prob(original_pred = original_pred_vecs, fault_vect_list = fault_vect_list)

    else:
        thds_detected_cnt, thds_critical_cnt, thds_per_sample_det_faults, thds_per_sample_crit_faults = 0, 0, [], []
        thds_det_perc = 0.0
        thds_crit_perc = 0.0
    
    logger.info(f"[THDs] Faults Detected: {thds_detected_cnt} PERC: {thds_det_perc}")
    logger.info(f"[THDs] Critical faults detected: {thds_critical_cnt} PERC: {thds_crit_perc}")

    """ ************************************************ """
    # For Reg IN faults.
    regIn_path   = os.path.join(input_faults_path, "reg_feat_faults.json5") 
    fault_vect_list = []
    # Load the feat JSON5 file
    with open(regIn_path, "r") as file:
        loaded_faults = json5.load(file)
    nro_regIn_faults = len(loaded_faults)
    
    
    # For each fault
    for f in tqdm(loaded_faults, desc = "Visiting with ReinIn faults"):    
        # Inject faults into the inputs
        old_dbs = classifier.inject_tree_regin_faults(f)
        # Visit
        faulted_vec = classifier.predict_many(classifier.x_test)
        # Save values 
        fault_vect_list.append(faulted_vec.tolist())
        # Restore
        classifier.restore_dbs(old_dbs)
        tv = classifier.predict_many(classifier.x_test)
        # thds_detected_cnt, thds_critical_cnt, thds_per_sample_det_faults, thds_per_sample_crit_faults = relate_faults_to_input(original_pred=original_pred_vecs, faulted_pred= [tv])
        # if thds_detected_cnt > 0:
        #     print(thds_detected_cnt)
        #     print(old_dbs)
        #     assert 1 == 0, "Restoring not working"

    if len(fault_vect_list) > 0:
        regIn_detected_cnt, regIn_critical_cnt, regIn_per_sample_det_faults, regIn_per_sample_crit_faults = relate_faults_to_input(original_pred=original_pred_vecs, faulted_pred= fault_vect_list)
        regIn_det_perc = regIn_detected_cnt / len(fault_vect_list) * 100
        regIn_crit_perc = regIn_critical_cnt / len(fault_vect_list) * 100
        # regIn_perc_detected, regIn_perc_crit, _, _, regIn_ctr_det, regIn_ctr_crit = compute_faults_prob(original_pred = original_pred_vecs, fault_vect_list = fault_vect_list)

    else:
        regIn_detected_cnt, regIn_critical_cnt, regIn_per_sample_det_faults, regIn_per_sample_crit_faults = 0, 0, [], []
        regIn_det_perc = 0.0
        regIn_crit_perc = 0.0
    
    logger.info(f"[RegIn] Faults Detected: {regIn_detected_cnt} PERC: {regIn_det_perc}")
    logger.info(f"[RegIn] Critical faults detected: {regIn_critical_cnt} PERC: {regIn_crit_perc}")
    
    report_dict = {
        "TotalFeatInj": nro_feat_faults,
        # **** Features

        # Counters
        "FeatDetected": feat_detected_cnt,
        "FeatCtrNonCrit": feat_detected_cnt - feat_critical_cnt,
        "FeatCritical": feat_critical_cnt,

        # Percentages over the total injected
        "FeatDetectedPerc": feat_detected_cnt / nro_feat_faults * 100,
        "FeatNonCritPerc": (feat_detected_cnt - feat_critical_cnt) / nro_feat_faults * 100,
        "FeatCritPerc": feat_crit_perc,
        # Percentages over the detected
        "FeatNonCritPercDet": (feat_detected_cnt - feat_critical_cnt) / feat_detected_cnt * 100,
        "FeatCritPercDet": feat_critical_cnt / feat_detected_cnt * 100,

        # **** DBS
        "TotalDBSInj": nro_dbs_faults,
        # Counters
        "DBSDetected": dbs_detected_cnt,
        "DBSCtrNonCrit": dbs_detected_cnt - dbs_critical_cnt,
        "DBSCritical": dbs_critical_cnt,
        # Percentages over the total injected
        "DBSDetectedPerc": dbs_detected_cnt / nro_dbs_faults * 100,
        "DBSNonCritPerc": (dbs_detected_cnt - dbs_critical_cnt) / nro_dbs_faults * 100,
        "DBSCritPerc": dbs_crit_perc,
        # Percentages over the detected
        "DBSNonCritPercDet": (dbs_detected_cnt - dbs_critical_cnt) / dbs_detected_cnt * 100,
        "DBSCritPercDet": dbs_critical_cnt / dbs_detected_cnt * 100,

        # **** BNS
        "TotalBNSInj": nro_bns_faults,
        # Counters
        "BNSDetected": bns_detected_cnt,
        "BNSCtrNonCrit": bns_detected_cnt - bns_critical_cnt,
        "BNSCritical": bns_critical_cnt,
        # Percentages over the total injected
        "BNSDetectedPerc": bns_detected_cnt / nro_bns_faults * 100,
        "BNSNonCritPerc": (bns_detected_cnt - bns_critical_cnt) / nro_bns_faults * 100,
        "BNSCritPerc": bns_crit_perc,

        # Percentages over the detected
        "BNSNonCritPercDet": (bns_detected_cnt - bns_critical_cnt) / bns_detected_cnt * 100,
        "BNSCritPercDet": bns_critical_cnt / bns_detected_cnt * 100,

        # **** THD
        "TotalTHDInj": nro_thds_faults,
        # Counters
        "THDDetected": thds_detected_cnt,
        "THDCtrNonCrit": thds_detected_cnt - thds_critical_cnt,
        "THDCritical": thds_critical_cnt,
        # Percentages over the total injected
        "THDDetectedPerc": thds_detected_cnt / nro_thds_faults * 100,
        "THDNonCritPerc": (thds_detected_cnt - thds_critical_cnt) / nro_thds_faults * 100,
        "THDCritPerc": thds_crit_perc,
        # Percentages over the detected
        "THDNonCritPercDet": (thds_detected_cnt - thds_critical_cnt) / thds_detected_cnt * 100,
        "THDCritPercDet": thds_critical_cnt / thds_detected_cnt * 100,

        # **** RegIn
        "TotalRegINInj": nro_regIn_faults,
        # Counters
        "RegINDetected": regIn_detected_cnt,
        "RegINCtrNonCrit": regIn_detected_cnt - regIn_critical_cnt,
        "RegINCritical": regIn_critical_cnt,
        # Percentages over the total injected
        "RegINDetectedPerc": regIn_detected_cnt / nro_regIn_faults * 100,
        "RegINNonCritPerc": (regIn_detected_cnt - regIn_critical_cnt) / nro_regIn_faults * 100,
        "RegINCritPerc": regIn_crit_perc,
        # Percentages over the detected
        "RegINNonCritPercDet": (regIn_detected_cnt - regIn_critical_cnt) / regIn_detected_cnt * 100,
        "RegINCritPercDet": regIn_critical_cnt / regIn_detected_cnt * 100,
    }
    out_report_path = f"{output}/report_inj.csv"
    print(out_report_path)
    report_df = pd.DataFrame(report_dict, index=[0])
    report_df.to_csv(out_report_path, index=False)
    
    

    feat_out_path =f"{output}/feat_faults_per_sample.txt"
    with open(feat_out_path, "w") as f:
        json.dump(feat_per_sample_det_faults, f, indent=1)
    feat_crit_out_path =f"{output}/feat_CRITfaults_per_sample.txt"
    with open(feat_crit_out_path, "w") as f:
        json.dump(feat_per_sample_crit_faults, f, indent=1)

    dbs_out_path =f"{output}/dbs_faults_per_sample.txt"
    with open(dbs_out_path, "w") as f:
        json.dump(dbs_per_sample_det_faults, f, indent=1)
    dbs_crit_out_path =f"{output}/dbs_CRITfaults_per_sample.txt"
    with open(dbs_crit_out_path, "w") as f:
        json.dump(dbs_per_sample_crit_faults, f, indent=1)

    bns_out_path =f"{output}/bns_faults_per_sample.txt"
    with open(bns_out_path, "w") as f:
        json.dump(bns_per_sample_det_faults, f, indent=1)
    bns_crit_out_path =f"{output}/bns_CRITfaults_per_sample.txt"
    # np.savetxt(bns_crit_out_path, bns_per_sample_crit_faults, fmt = "%d")
    with open(bns_crit_out_path, "w") as f:
        json.dump(bns_per_sample_crit_faults, f, indent=1)

    thds_out_path =f"{output}/thds_faults_per_sample.txt"
    with open(thds_out_path, "w") as f:
        json.dump(thds_per_sample_det_faults, f, indent=1)
    thds_crit_out_path =f"{output}/thds_CRITfaults_per_sample.txt"
    with open(thds_crit_out_path, "w") as f:
        json.dump(thds_per_sample_crit_faults, f, indent=1)

    regIn_out_path =f"{output}/regIn_faults_per_sample.txt"
    with open(regIn_out_path, "w") as f:
        json.dump(regIn_per_sample_det_faults, f, indent=1)
    regIn_crit_out_path =f"{output}/regIn_CRITfaults_per_sample.txt"
    with open(regIn_crit_out_path, "w") as f:
        json.dump(regIn_per_sample_crit_faults, f, indent=1)