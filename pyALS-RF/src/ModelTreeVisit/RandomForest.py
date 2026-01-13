import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score
import joblib
from sklearn.tree import export_text
import io
import sys
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from typing import List,Optional
import matplotlib.pyplot as plt
from tqdm import tqdm
from .DecisionTree import Decision_Tree_Classifier
from pyalslib import list_partitioning
from multiprocessing import cpu_count, Pool

class Random_Forest_Classifier:
    
    def __init__(self, ncpus: int  = 1, estimators : Optional[List[DecisionTreeClassifier]] = None):
        self.ncpus = ncpus
        self.n_classes = 0
        if estimators == None:
            self.n_estimators = 0
            self.estimators = []
        else:
            self.n_estimators = len(estimators)
            self.estimators = estimators
        
    def init_multicore(self, ncpus):
        assert len(self.n_estimators) > 0, "This method should be invoked only with an initialized Ensemble !"
        self.p_tree = list_partitioning(self.trees, ncpus)
        self.ncpus = ncpus
        self.pool = Pool(ncpus)
        
    @staticmethod
    def convert_sklearn_model( sklearn_random_forest_model: RandomForestClassifier):
        rf = Random_Forest_Classifier()
        for est in sklearn_random_forest_model.estimators_:
            rf.estimators.append(Decision_Tree_Classifier.convert_sklearn_tree_to_nodes(est))
        rf.n_estimators = len(rf.estimators)
        rf.n_classes = sklearn_random_forest_model.n_classes_
        return rf 
    
    def import_pyals_dataset_description(self, dataset_description):
        # Save other parameters
        if dataset_description is not None:
            self.classes_name = dataset_description.classes_name
            # In case the dataset is splitted (copied into the outdir folder during training)
            # then use the original separator
            if dataset_description.separated_training : 
                self.csv_separator = dataset_description.separator
            # Otherwise use the standard ";"
            else:
                self.csv_separator = ";"
            self.out_column = dataset_description.outcome_col
        else:
            self.classes_name = self.model_classes
            self.csv_separator = ";"
            self.out_column = -1
        
    def read_test_set(self, dataset_csv, no_header = False):
        self.dataframe = pd.read_csv(dataset_csv, sep = self.csv_separator)
        # Todo : Remove the assumption of the last column being the label
        attribute_name = list(self.dataframe.keys())[:-1]
        out_col = self.dataframe.keys()[-1]
        # assert len(attribute_name) == len(self.model_features), f"Mismatch in features vectors. Read {len(attribute_name)} features, buth PMML says it must be {len(self.model_features)}!"
        # f_names = [ f["name"] for f in self.model_features]
        if not no_header:
            #name_matches = [ a == f for a, f in zip(attribute_name, f_names) ] # SUBSTITUTED FOR TIC TAC TOE ENDGAME
            # name_matches = [ a.replace('-', '_') == f.replace('-', '_') for a, f in zip(attribute_name, f_names) ]
            # assert all(name_matches), f"Feature mismatch at index {name_matches.index(False)}: {attribute_name[name_matches.index(False)]} != {f_names[name_matches.index(False)]}"
            self.x_test = self.dataframe.loc[:, self.dataframe.columns != out_col].values
            self.y_test = self.dataframe.loc[:, self.dataframe.columns == out_col].values
            # for arg in self.args:
            #     arg[1] = self.x_test
        else: # TEMPORARY, TO GENERALIZE IN FUTURE
            self.x_test = self.dataframe.iloc[:, : -1].values
            self.y_test = self.dataframe.iloc[:, -1].values
            # for arg in self.args:
            #     arg[1] = self.x_test


    def read_training_set(self, dataset_csv, no_header = False):
        self.dataframe = pd.read_csv(dataset_csv, sep = self.csv_separator)
        # Todo : Remove the assumption of the last column being the label
        attribute_name = list(self.dataframe.keys())[:-1]
        out_col = self.dataframe.keys()[-1]
        #f_names = [ f["name"] for f in self.model_features]
        if not no_header:
            #name_matches = [ a == f for a, f in zip(attribute_name, f_names) ] # SUBSTITUTED FOR TIC TAC TOE ENDGAME
            #name_matches = [ a.replace('-', '_') == f.replace('-', '_') for a, f in zip(attribute_name, f_names) ]
            #assert all(name_matches), f"Feature mismatch at index {name_matches.index(False)}: {attribute_name[name_matches.index(False)]} != {f_names[name_matches.index(False)]}"
            self.x_train = self.dataframe.loc[:, self.dataframe.columns != out_col].values
            self.y_train = self.dataframe.loc[:, self.dataframe.columns == out_col].values
            # for arg in self.args:
            #     arg[1] = self.x_test
        else: # TEMPORARY, TO GENERALIZE IN FUTURE
            self.x_train = self.dataframe.iloc[:, : -1].values
            self.y_train = self.dataframe.iloc[:, -1].values
            # for arg in self.args:
            #     arg[1] = self.x_test

    # Get the output vector.
    def predict_proba(self, X):
        pv = [0 for x in range(self.n_classes)]
        for t in self.estimators:
            pv[t.predict(X)] += 1
        return pv
    
    # Return the class of a input sample
    def predict(self, X):
        
        return np.argmax(self.predict_proba(X))
    
    @staticmethod
    def _visit_single_tree(ptrees, X):
        tree_lists = [[] * len(ptrees) ] 
        for tid, tree in enumerate(ptrees): 
            tree_lists[tid] = tree.visit(X)
        return tree_lists
    
    # Perform a tree visiting operation. 
    def visit(self, X):
        if self.n_estimators == 1:
            return np.array([self.estimators[0].visit(X)])
        else:
            args = [[t, X] for t in self.p_tree]
            return self.pool.starmap(Random_Forest_Classifier._visit_single_tree, args)
            
    # Visit the set of samples by updating also the set of "samples per leaf", i.e. the set of samples
    # falling within the leaf.
    def visit_with_sample_update(self, X):
        leaves = []
        for tree in self.estimators:
            leaf = tree.visit(X)
            leaf.samples_in_leaf.append(X)
            leaves.append(leaf)
        return leaves
    
    # Given a set of leaves (one per each DT in the ensemble), this function
    # transforms such set into a prediction vector ( analoguos to the predict_proba)
    # This function is useful when transforming a set of leaves (obtained through the RF visit function)
    # into a prediction vector.
    def leaf_to_pv(self, leaves):
        pv = [0 for x in range(self.n_classes)]
        for leaf in leaves:
            pv[np.argmax(leaf.value)] += 1
        return pv
    
    # Returns the total number of nodes for each tree, in the ensemble.
    def get_total_nodes(self):
        total_internal = 0
        total_leaves = 0
        for tree in self.estimators:
            total_internal += tree.n_internal_nodes
            total_leaves += tree.n_leaf_nodes
        return total_internal, total_leaves
    

