"""
Have Fun!
- 189 Course Staff
"""
from collections import Counter

import numpy as np
from numpy import genfromtxt
import scipy.io
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score
from pydot import graph_from_dot_data
import io

import random
random.seed(246810)
np.random.seed(246810)

eps = 1e-5  # a small number

class DecisionTree:

    #def __init__(self, max_depth=5, feature_labels=None, X = np.array([[]]), y = np.array([])):
    def __init__(self, max_depth=5, feature_labels=None):
        #self.X = X
        #self.y = y
        self.max_depth = max_depth
        self.features = feature_labels
        self.left, self.right = None, None  # for non-leaf nodes
        self.split_idx, self.thresh = None, None  # for non-leaf nodes
        self.data, self.pred = None, None  # for leaf nodes

    @staticmethod
    def entropy(y):
        if y.size == 0:
            return 0
        unique_elements, counts_all_classes = np.unique(y, return_counts=True)
        proportions = counts_all_classes / y.size
        entropy = -np.sum(proportions * np.log2(proportions))
        return entropy

    #feature_idx is added here 
    def information_gain2(self, X, y, feature_idx, thresh):
        initial_entropy = self.entropy(y)
        left_indices = np.where(X[:, feature_idx] <= thresh)[0]
        right_indices = np.where(X[:, feature_idx] > thresh)[0]
        
        left_entropy = self.entropy(y[left_indices])
        right_entropy = self.entropy(y[right_indices])

        left_weight = len(left_indices) / len(y)
        right_weight = len(right_indices) / len(y)
        entropy_after = left_weight * left_entropy + right_weight * right_entropy
        info_gain = initial_entropy - entropy_after
        return info_gain
    ####### abandoned ##################
    
    def information_gain(self, X, y, feature_idx, threshold):
        initial_entropy = self.entropy(y)
        
        # Boolean masks for splitting
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask  # Elements not in the left are in the right
        
        left_entropy = self.entropy(y[left_mask])
        right_entropy = self.entropy(y[right_mask])

        left_weight = np.sum(left_mask) / len(y)
        right_weight = 1 - left_weight  # More efficient than recalculating

        entropy_after = left_weight * left_entropy + right_weight * right_entropy
        info_gain = initial_entropy - entropy_after
        return info_gain


    @staticmethod
    def gini_impurity(X, y, thresh):
        # TODO
        pass

    @staticmethod
    def gini_purification(X, y, thresh):
        # TODO
        pass

    def split(self, X, y, idx, thresh):
        X0, idx0, X1, idx1 = self.split_test(X, idx=idx, thresh=thresh)
        y0, y1 = y[idx0], y[idx1]
        return X0, y0, X1, y1

    def split_test(self, X, idx, thresh):
        idx0 = np.where(X[:, idx] <= thresh)[0]
        idx1 = np.where(X[:, idx] > thresh)[0]
        X0, X1 = X[idx0, :], X[idx1, :]
        return X0, idx0, X1, idx1

    def fit(self, X, y):
        if y.size == 0:
            self.pred = -1
        elif y.size == 1:
            self.pred = y[0]
        elif self.max_depth == 0:
            majority_value = self.majority(y)
            self.pred = majority_value
        else:
            d = np.atleast_2d(X)[0].size
            all_info_gains = []
            #all_split_ways = []
            all_thresholds = []
            all_features = []

            # helper: given y = np.array([0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0]), 
            # output Pairs of indices with difference of 1: [(0, 1), (5, 6), (6, 7), (9, 10)]
            diff_left = np.insert(np.diff(y), 0, 0)  
            diff_right = np.append(np.diff(y), 0)
            indices = np.where((diff_left != 0) | (diff_right != 0))[0]
            diffs = np.diff(indices)
            diffs_eq_1 = np.where(diffs == 1)[0]  
            index_pairs = [(indices[pos], indices[pos + 1]) for pos in diffs_eq_1]
            ############################################################################
            if len(index_pairs) == 0:
                self.pred = self.majority(y)
            else:
                sorted_indices = [np.argsort(X[:, i]) for i in range(X.shape[1])]
                sorted_X = [X[indices] for indices in sorted_indices]
                sorted_y = [y[indices] for indices in sorted_indices]
                for feature_idx in range(d):
                    #sorted_indices = np.argsort(X[:, feature_idx])
                    X = sorted_X[feature_idx]
                    y = sorted_y[feature_idx]
                    X_col = X[:, feature_idx]
                    thresholds = self.compute_all_threshold(index_pairs, X_col)
                    for threshold in thresholds:
                        all_thresholds.append(threshold)
                        all_info_gains.append(self.information_gain(X, y, feature_idx, threshold))
                        #x0, y0, x1, y1 = self.split(X, y, feature_idx, threshold)
                        #all_split_ways.append([x0, y0, x1, y1])
                        all_features.append(feature_idx)
                    
                all_info_gains = np.array(all_info_gains)
                max_gain_index = np.argmax(all_info_gains)
                #opt_split_way = all_split_ways[max_gain_index]
                self.split_idx = all_features[max_gain_index]
                self.thresh = all_thresholds[max_gain_index]
                X = sorted_X[self.split_idx]
                y = sorted_y[self.split_idx]
                opt_split_way = self.split(X, y, self.split_idx, self.thresh)
                #self.left = DecisionTree(max_depth = self.max_depth - 1, X = opt_split_way[:2][0], y = opt_split_way[:2][1])
                #self.right = DecisionTree(max_depth = self.max_depth - 1, X = opt_split_way[2:][0], y = opt_split_way[2:][1])
                self.left = DecisionTree(max_depth = self.max_depth - 1)
                self.right = DecisionTree(max_depth = self.max_depth - 1)
                #print(opt_split_way)
                #print(f"feature is {self.split_idx}")
                #print(f"threshold is {self.thresh}")
                #self.left.fit(self.left.X, self.left.y)
                #self.right.fit(self.right.X, self.right.y)
                self.left.fit(opt_split_way[:2][0], opt_split_way[:2][1])
                self.right.fit(opt_split_way[2:][0], opt_split_way[2:][1])
            

    def predict(self, X):
        if self.pred == -1:
            #self.data = X
            #length_pred = self.count_size_2dArray(X)
            #self.pred = np.array([None for i in range(length_pred)])
            return None, None
        elif self.pred != None:
            self.data = X
            length_pred = self.count_size_2dArray(X)
            self.pred = np.array([self.pred for i in range(length_pred)])
            return np.atleast_2d(X), self.pred
        else:
            x0, idx0, x1, idx1 = self.split_test(X, self.split_idx, self.thresh)
            #left_data, left_pred = self.left.predict(X0)
            #right_data, right_pred = self.right.predict(X1)
            #self.data, self.pred = self.concatenate(left_data, left_pred, right_data, right_pred)
            #return np.atleast_2d(self.data), self.pred
            #self.left.data, self.left.pred = self.left.predict(x0)
            #self.right.data, self.right.pred = self.right.predict(x1)
            left_data, left_pred = self.left.predict(x0)
            right_data, right_pred = self.right.predict(x1)
            #return self.concatenate(self.left.data, self.right.data, self.left.pred, self.right.pred)
            return self.concatenate(left_data, right_data, left_pred, right_pred)


    # input: Example numpy array x = np.array([4, 5, 1, 3, 2, 8, 9, 7, 6, 0, 1])
    # input: Given list of pairs of indices index_pairs = [(0, 1), (5, 6), (6, 7), (9, 10)]
    # output: Means of values in x for each pair of indices: [4.5 8.5 8.  0.5]
    def compute_all_threshold(self, index_pairs, x):
        means = []
        for pair in index_pairs:
            start_idx, end_idx = pair
            values = x[start_idx:end_idx + 1] 
            mean_value = np.mean(values)
            means.append(mean_value)
        means_array = np.array(means)
        return means_array

    # concanate left data and right data, left labels and right labels, format is the same as design matrix and np.array y
    def concatenate(self, X1, X2, y1, y2):
        if X1 is None or X2 is None:
            return X1 or X2, y1 or y2
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        #print("Shape of X1:", X1.shape)  # Debugging print
        #print("Shape of X2:", X2.shape)
        X_combined = np.concatenate((X1, X2), axis=0)
        y_combined = np.concatenate((y1, y2))
        return X_combined, y_combined

    # calculate the majority number of an array 
    def majority(self, array):
        sum_of_elements = np.sum(array)
        majority_value = 1 if sum_of_elements > len(array) / 2 else 0
        return majority_value

    # calculate the number of Xi points in a leaf node 
    def count_size_2dArray(self, arr):
        if not isinstance(arr, np.ndarray):
            raise ValueError("Input must be a numpy array")
        if arr.ndim == 1:  # 1D array
            return 1
        elif arr.ndim == 2:  # 2D array
            return arr.shape[0]  # The number of rows (nested arrays)
        else:
            raise ValueError("Input must be either 1D or 2D numpy array")

    def __repr__(self):
        if self.max_depth == 0:
            return "%s (%s)" % (self.pred, self.labels.size)
        else:
            return "[%s < %s: %s | %s]" % (self.features[self.split_idx],
                                           self.thresh, self.left.__repr__(),
                                           self.right.__repr__())

class BaggedTrees(BaseEstimator, ClassifierMixin):

    def __init__(self, params=None, n=200):
        if params is None:
            params = {}
        self.params = params
        self.n = n
        self.decision_trees = [
            DecisionTreeClassifier(random_state=i, **self.params)
            for i in range(self.n)
        ]

    def fit(self, X, y):
       # for tree in self.decision_trees:
         #   tree.fit(X, y)
        pass

    def predict(self, X):

        #for tree in self.decision_trees:
        pass



class RandomForest(BaggedTrees):

    def __init__(self, params=None, n=200, m=1):
        if params is None:
            params = {}
        params['max_features'] = m
        self.m = m
        super().__init__(params=params, n=n)


class BoostedRandomForest(RandomForest):

    def fit(self, X, y):
        # TODO
        pass
    
    def predict(self, X):
        # TODO
        pass


def preprocess(data, fill_mode=True, min_freq=10, onehot_cols=[]):
    # fill_mode = False

    # Temporarily assign -1 to missing data
    data[data == b''] = '-1'

    # Hash the columns (used for handling strings)
    onehot_encoding = []
    onehot_features = []
    for col in onehot_cols:
        counter = Counter(data[:, col])
        for term in counter.most_common():
            if term[0] == b'-1':
                continue
            if term[-1] <= min_freq:
                break
            onehot_features.append(term[0])
            onehot_encoding.append((data[:, col] == term[0]).astype(float))
        data[:, col] = '0'
    onehot_encoding = np.array(onehot_encoding).T
    data = np.hstack(
        [np.array(data, dtype=float),
         np.array(onehot_encoding)])

    # Replace missing data with the mode value. We use the mode instead of
    # the mean or median because this makes more sense for categorical
    # features such as gender or cabin type, which are not ordered.
    if fill_mode:
        # TODO
        pass

    return data, onehot_features


def evaluate(clf):
    print("Cross validation", cross_val_score(clf, X, y))
    if hasattr(clf, "decision_trees"):
        counter = Counter([t.tree_.feature[0] for t in clf.decision_trees])
        first_splits = [
            (features[term[0]], term[1]) for term in counter.most_common()
        ]
        print("First splits", first_splits)


if __name__ == "__main__":
    #dataset = "titanic"
    dataset = "spam"
    params = {
        "max_depth": 5,
        # "random_state": 6,
        "min_samples_leaf": 10,
    }
    N = 100

    if dataset == "titanic":
        # Load titanic data
        path_train = 'datasets/titanic/titanic_training.csv'
        data = genfromtxt(path_train, delimiter=',', dtype=None)
        path_test = 'datasets/titanic/titanic_testing_data.csv'
        test_data = genfromtxt(path_test, delimiter=',', dtype=None)
        y = data[1:, 0]  # label = survived
        class_names = ["Died", "Survived"]

        labeled_idx = np.where(y != b'')[0]
        y = np.array(y[labeled_idx], dtype=float).astype(int)
        print("\n\nPart (b): preprocessing the titanic dataset")
        X, onehot_features = preprocess(data[1:, 1:], onehot_cols=[1, 5, 7, 8])
        X = X[labeled_idx, :]
        Z, _ = preprocess(test_data[1:, :], onehot_cols=[1, 5, 7, 8])
        assert X.shape[1] == Z.shape[1]
        features = list(data[0, 1:]) + onehot_features

    elif dataset == "spam":
        features = [
            "pain", "private", "bank", "money", "drug", "spam", "prescription",
            "creative", "height", "featured", "differ", "width", "other",
            "energy", "business", "message", "volumes", "revision", "path",
            "meter", "memo", "planning", "pleased", "record", "out",
            "semicolon", "dollar", "sharp", "exclamation", "parenthesis",
            "square_bracket", "ampersand"
        ]
        assert len(features) == 32

        # Load spam data
        path_train = 'datasets/spam_data/spam_data.mat'
        data = scipy.io.loadmat(path_train)
        X = data['training_data']
        y = np.squeeze(data['training_labels'])
        Z = data['test_data']
        class_names = ["Ham", "Spam"]

    else:
        raise NotImplementedError("Dataset %s not handled" % dataset)

    print("Features", features)
    print("Train/test size", X.shape, Z.shape)
    
    print("\n\nPart 0: constant classifier")
    print("Accuracy", 1 - np.sum(y) / y.size)

    # sklearn decision tree
    print("\n\nsklearn's decision tree")
    clf = DecisionTreeClassifier(random_state=0, **params)
    #clf = DecisionTree()
    clf.fit(X, y)
    evaluate(clf)
    out = io.StringIO()
    export_graphviz(
        clf, out_file=out, feature_names=features, class_names=class_names)
    # For OSX, may need the following for dot: brew install gprof2dot
    graph = graph_from_dot_data(out.getvalue())
    graph_from_dot_data(out.getvalue())[0].write_pdf("%s-tree-mine.pdf" % dataset)
    
    # TODO
