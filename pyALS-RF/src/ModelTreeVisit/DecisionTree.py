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
import heapq

# Notes: 
# 1- The set of leaves can not be completely removed.
#       Indeed, if a leaf is not pruned during the exploration of a sample, 
#       then its value is not removed.
#   - A possible solution to this problem may consist in mantaining, for each leaf, the set of consecutive indexes, i.e. the set of 
#   indexes which will be removed. In this way, when a leaf is removed, then it will be removed from all the other remaining indexes.
#   - Another possible solution, is to ensure that a leaf is already pruned, maybe, mantaining an attribute in the data-structure.
#       In this way the algorithm will automatically skip leaves which are already pruned. 
# 2- Add model quantization support. 
# 3- Add multicore support...
# 4- Find a better way to update the resiliency of samples...

""" Class representing a standard Decision Tree Node. 
"""
class TreeNode:
    
    def __init__(self, id = 0, threshold=None, feature=None, value=None):
        self.left = None
        self.right = None
        self.parent = None
        self.threshold = threshold
        self.feature = feature
        self.value = value  
        self.id = 0
        self.pruned = False

    def is_leaf(self):
        return self.left is None and self.right is None

    def get_depth(self):
        current_node = self
        depth = 0
        while current_node != None:
            current_node = current_node.parent
            depth += 1
        return depth
    
    def __eq__(self, other):
        if other == None:
            return False
        if self.id != other.id:
            return False
        return True
    

class Decision_Tree_Classifier:
    
    def __init__(self, nodes, n_internal_nodes, n_leaf_nodes):
        self.nodes = nodes
        self.n_internal_nodes = n_internal_nodes
        self.n_leaf_nodes = n_leaf_nodes
    
    
    def visit(self, X):
        """
        Batch visiting of the classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features) or (n_features,)
            Input samples.

        Returns
        -------
        leaf_nodes : list[TreeNode]
            leaf_nodes[i] è la foglia raggiunta dal campione X[i].
            Se X è monodimensionale, leaf_nodes avrà lunghezza 1.
        """
        X = np.asarray(X)

        # Normalizing the Shape to 2D (1, n_features)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Ensuring that X is contiguous in memory
        X = np.ascontiguousarray(X)

        n_samples, _ = X.shape

        # Vector to store, for each sample, the reached leaf.
        leaf_nodes = [None] * n_samples

        # Frontier: This is a vector holding for, each node, the set of samples to process
        #           for that specific node.
        root = self.nodes[0]
        frontier = [(root, np.arange(n_samples, dtype=int))]

        # While frontier -> while there are still nodes to process.
        while frontier:
            # Initialize the next frontied
            next_frontier = []
            # For each node, and sample
            for node, idxs in frontier:
                # If the node is leaf, inference is terminated.
                if node.is_leaf():
                    for i in idxs:
                        leaf_nodes[i] = node
                    continue
                # Otherwise perform the comparison.
                feat = node.feature
                thr = node.threshold

                # Take the sample and feathre and check whether you should go left/right
                # NOTE THAT THIS IS PERFORMED FOR ALL THE SAMPLES !
                x_feat = X[idxs, feat]
                go_left = x_feat <= thr

                # Split vettoriale degli indici
                left_idxs = idxs[go_left]
                right_idxs = idxs[~go_left]

                if left_idxs.size > 0:
                    next_frontier.append((node.left, left_idxs))
                if right_idxs.size > 0:
                    next_frontier.append((node.right, right_idxs))
            # Normalize the next frontier
            frontier = next_frontier

        return leaf_nodes
    
    def predict_proba(self, X):
        """
        Compute class probability estimates for the given input samples.

        This method performs a batch traversal of the decision tree using
        `visit(X)`, which returns the leaf node reached by each sample.
        For every leaf, its stored value vector (typically class counts or
        class probabilities) is extracted and stacked into a single NumPy
        array. This enables efficient vectorized processing and avoids
        Python-level loops wherever possible.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_features,)
            Input samples for which class probabilities are to be predicted.
            A one-dimensional array is treated as a single-sample batch.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Probability estimates for each sample. The number of classes
            corresponds to the dimensionality of `leaf.value` in the stored
            tree. If leaf values represent class counts rather than
            probabilities, the caller may normalize the rows as needed.

        Notes
        -----
        - This method is optimized for batch inference. It leverages the
        vectorized batch traversal in `visit(X)` and performs a single
        `np.argmax` operation for all samples collectively.
        - No dictionaries or hash maps are used, in accordance with the
        design constraints of the custom tree representation.
        """
        leaves = self.visit(X)  # list of TreeNode objects

        # Stack all leaf.value vectors into a 2D array
        proba = np.stack([np.asarray(leaf.value).ravel() for leaf in leaves], axis=0)
        return proba
    
    def predict(self, X):
        """
        Predict class labels for the given input samples.

        This method relies on `predict_proba(X)` to compute the per-sample
        probability vectors and selects the class with the highest probability
        via `np.argmax`. Both single-sample and batch inputs are supported,
        with all computations performed in a vectorized fashion.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_features,)
            Input samples whose class labels are to be predicted.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Integer-encoded class predictions for each input sample.
        """
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    # Convert a Scikit-Learn Model to the used internal representation
    @staticmethod
    def convert_sklearn_tree_to_nodes(tree_model):
        tree = tree_model.tree_
        n_nodes = tree.node_count
        children_left = tree.children_left
        children_right = tree.children_right
        thresholds = tree.threshold
        features = tree.feature
        values = tree.value

        # Crea tutti i nodi
        nodes = [TreeNode() for _ in range(n_nodes)]
        n_internal_nodes = 0
        n_leaf_nodes = 0
        # Popola ogni nodo
        for i in range(n_nodes):
            if children_left[i] != children_right[i]:  # nodo interno
                nodes[i].threshold = thresholds[i]
                nodes[i].feature = features[i]
                nodes[i].left = nodes[children_left[i]]
                nodes[i].right = nodes[children_right[i]]
                nodes[children_left[i]].parent = nodes[i]
                nodes[children_right[i]].parent = nodes[i]
                nodes[i].id = i 
                n_internal_nodes +=1
            else: 
                nodes[i].value = values[i]
                nodes[i].left = None
                nodes[i].right = None
                nodes[i].id = i
                n_leaf_nodes += 1
        return Decision_Tree_Classifier(nodes, n_internal_nodes, n_leaf_nodes)
    
    """
    Compare the traversal of a Decision_Tree_Classifier with the one of the SKLEARN DT.
    """
    def compare_tree_paths(self, x, sklearn_tree):

        node_sklearn = 0
        node_custom = self.nodes[0]
        tree = sklearn_tree.tree_

        step = 0
        while True:
            # Se entrambi sono in una foglia
            if (tree.children_left[node_sklearn] == tree.children_right[node_sklearn]) and node_custom.is_leaf():
                print("✅ Same Leaf.")
                print(tree.value[node_sklearn])
                print(node_custom.value)
                return True

            # Se uno è foglia e l'altro no → errore
            if node_custom.is_leaf() or (tree.children_left[node_sklearn] == tree.children_right[node_sklearn]):
                print("❌ Divergence: One is leaf and the other is not.")
                return False

            # Confronta feature e threshold
            feat_skl = tree.feature[node_sklearn]
            feat_cus = node_custom.feature
            thresh_skl = tree.threshold[node_sklearn]
            thresh_cus = node_custom.threshold

            if feat_skl != feat_cus or not np.isclose(thresh_skl, thresh_cus, atol=1e-7):
                print(f"❌ Divergence in parameters (step {step}):")
                print(f"   sklearn: feat={feat_skl}, thresh={thresh_skl}")
                print(f"   custom : feat={feat_cus}, thresh={thresh_cus}")
                return False

            val = x[feat_skl]
            print(f"Step {step} - feat[{feat_skl}] = {val} vs soglia = {thresh_skl}")

            # Calcola la direzione
            go_left_skl = val <= thresh_skl
            go_left_cus = val <= thresh_cus

            if go_left_skl != go_left_cus:
                print(f"❌ Divergence in direction (step {step}):")
                print(f"   sklearn → {'sinistra' if go_left_skl else 'destra'}")
                print(f"   custom  → {'sinistra' if go_left_cus else 'destra'}")
                return False

            # Procedi al nodo successivo
            node_sklearn = tree.children_left[node_sklearn] if go_left_skl else tree.children_right[node_sklearn]
            node_custom = node_custom.left if go_left_cus else node_custom.right
            step += 1   

    # Perform a pruning operation to a leaf node.
    def prune_leaf(self, leaf_to_prune : TreeNode):
        # Identify the Parent Node.
        parent_node = leaf_to_prune.parent
        # Identify whether the parent is a left or right child
        # This is done in order to remove the need of 
        if parent_node.left.id == leaf_to_prune.id:
            sibling_node = parent_node.right
        else:
            sibling_node = parent_node.left

        # Remove both the leaf and its parent from the DT.
        self.nodes.remove(parent_node)
        self.nodes.remove(leaf_to_prune)

        # If the parent is no root
        if parent_node.parent != None:
            # Find the grandparent and attach the sibling substituing the father of the pruned leaf
            grandparent = parent_node.parent
            # Find where to attach the sibling node.
            if grandparent.left.id == parent_node.id:
                grandparent.left = sibling_node
            else:
                grandparent.right = sibling_node
            sibling_node.parent = grandparent
        else: # In case the parent is the root node, then the sibling is the new root.
            self.nodes.remove(sibling_node)
            self.nodes.insert(0, sibling_node)
            sibling_node.parent = None
            grandparent = None
        self.n_internal_nodes -= 1 
        self.n_leaf_nodes -= 1
        # Indicate that the leaf has been pruned.
        leaf_to_prune.pruned = True
        return grandparent, parent_node, sibling_node
    
    def reattach_leaf(self, grandparent_node, parent_node, sibling_node, leaf_removed):
        # If the parent was not the Root of the tree.
        if parent_node.parent != None:
            # Find where to attach the sibling node.
            if grandparent_node.left.id == sibling_node.id:
                grandparent_node.left = parent_node
            else:
                grandparent_node.right = parent_node
            sibling_node.parent = parent_node
            self.nodes.append(parent_node)
            self.nodes.append(leaf_removed)
        # Otherwise
        else:
            self.nodes.remove(sibling_node)
            self.nodes.insert(0, parent_node)
            self.nodes.insert(1, sibling_node)
            self.nodes.insert(2, leaf_removed)
            sibling_node.parent = parent_node
        self.n_internal_nodes += 1 
        self.n_leaf_nodes += 1


    # *************** THESE ARE NOT BATCH FUNCTIONS FOR VISITING THE TREE.

    # Perform a visiting of the classifier.
    def visit_no_batch(self, X):
        current_node = self.nodes[0]
        # Until I'm not in a leaf
        while current_node.left != None:
            if X[current_node.feature] <= current_node.threshold:
                current_node = current_node.left
            else:
                current_node = current_node.right
        return current_node
    
    # Predict the class probability ( the vector of probabilities for each class for the reached leaf during visiting)
    def predict_proba_no_batch(self, X):
        return self.visit_no_batch(X).value
    
    # return the predicted class of the leaf ( the argmax of the vector probability)
    def predict_no_batch(self, X):
        return np.argmax(self.predict_proba_no_batch(X))



