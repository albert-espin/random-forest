import abc
import random
from collections import Counter
from itertools import combinations
import numpy as np
from scipy.stats import mode
from sklearn.utils import resample
from data_functions import *


class TreeNode(object):

    """Base tree node class"""

    @abc.abstractmethod
    def __init__(self):

        """Constructor"""

        self.is_terminal = False


class BinaryDecisionNode(TreeNode):

    """Node of binary decision tree"""

    __slots__ = ("is_terminal", "feature_name", "split_value", "left_child", "right_child")

    def __init__(self, feature_name, split_value):

        """Constructor"""

        super().__init__()
        self.feature_name = feature_name
        self.split_value = split_value
        self.left_child = None
        self.right_child = None

    def evaluate(self, value):

        """Evaluate the node's condition given a value"""

        # numeric case: greater-than check
        if is_value_numeric(value):
            return value >= self.split_value

        # categorical case: is-value-contained check
        return value in self.split_value


class TerminalNode(TreeNode):

    """Terminal node of a tree"""

    __slots__ = ("is_terminal", "class_value")

    def __init__(self, class_value):

        """Constructor"""

        super().__init__()
        self.is_terminal = True
        self.class_value = class_value


class DataSplit(object):

    """Split of data (instances and class values) in two parts (left and right)"""

    __slots__ = ("left_instances", "left_class_values", "right_instances", "right_class_values")

    def __init__(self, left_instances, left_class_values, right_instances, right_class_values):

        self.left_instances = left_instances
        self.left_class_values = left_class_values
        self.right_instances = right_instances
        self.right_class_values = right_class_values


def get_bootstrap_sample(instances, class_values, sample_size):

    """Produce a bootstrap sample (random selection with replacement) for the passed instances and classes"""

    return resample(instances, class_values, replace=True, n_samples=sample_size)


def get_class_impurity(class_values):

    """Calculate the Gini index of impurity for the passed class values"""

    # get the frequency of each class
    class_frequency_dict = class_values.value_counts(normalize=True).to_dict()

    # penalize low class frequencies with the square, so that purity decreases
    purity = 0
    for class_value, frequency in class_frequency_dict.items():
        purity += frequency ** 2

    # impurity is the complementary of purity
    return 1 - purity


def get_pair_impurity(class_values0, class_values1):

    """Calculate the Gini index of impurity for the passed pair of lists of class values"""

    total_size = len(class_values0) + len(class_values1)

    # the impurity of the split is the weighted sum (by size proportion) of each set of class values in the split pair
    return len(class_values0) / total_size * get_class_impurity(class_values0) + len(class_values1) / total_size * get_class_impurity(class_values1)


def get_data_split_impurity(data_split):

    """Calculate the Gini index of impurity for the passed data split"""

    return get_pair_impurity(data_split.left_class_values, data_split.right_class_values)


def split_data(instances, class_values, split_feature, split_value):

    """Obtain a split of the passed data in the given feature at the given split point"""

    # numeric case: lower-than check
    if is_value_numeric(split_value):
        left_instances = instances.loc[instances[split_feature] < split_value]

    # categorical case: is-value-contained check
    else:
        left_instances = instances[-instances[split_feature].isin(split_value)]

    # discard the split if all instances are grouped together
    if len(left_instances) == 0 or len(left_instances) == len(instances):
        return None

    left_class_values = class_values[class_values.index.isin(left_instances.index)]
    right_instances = instances[~instances.index.isin(left_instances.index)]
    right_class_values = class_values[~class_values.index.isin(left_class_values.index)]

    return DataSplit(left_instances, left_class_values, right_instances, right_class_values)


def create_best_split_node(instances, class_values, split_feature_num):

    """Create the node representing the best split for the passed data, among the given number of randomly-selected features"""

    best_split_feature = None
    best_split_value = None
    best_data_split = None
    best_impurity = np.inf

    # randomly select the features for the split
    split_features = random.sample(instances.columns.tolist(), split_feature_num)

    for split_feature in split_features:

        # the split points for numeric features are the mid-points between the sorted values
        if split_feature in get_numeric_column_names(instances):
            unique_values = sorted(np.unique(instances[split_feature]))
            split_values = [(value0 + value1) / 2 for value0, value1 in zip(unique_values[:-1], unique_values[1:])]

        # for categorical features, the split points are every combination of membership among categories
        else:
            categories = np.unique(instances[split_feature])
            split_values = list()
            for i in range(0, len(categories)):
                split_values.extend(list(combinations(categories, i+1)))

        # compute the impurity of each split to check if it is the new best one (minimum impurity)
        for split_value in split_values:

            data_split = split_data(instances, class_values, split_feature, split_value)

            if data_split:
                impurity = get_data_split_impurity(data_split)

                if impurity < best_impurity:
                    best_impurity = impurity
                    best_split_feature = split_feature
                    best_split_value = split_value
                    best_data_split = data_split

    # create a node representing the split point
    node = BinaryDecisionNode(best_split_feature, best_split_value)

    return node, best_data_split


def build_random_tree(instances, class_values, split_feature_num):

    """Build a decision tree that performs random splits of the specified number of features"""

    unique_classes = np.unique(class_values)

    if len(unique_classes) == 0:
        return None

    # if all the instances belong to the same class, just add a terminal node with the class
    if len(unique_classes) == 1:
        return TerminalNode(unique_classes[0])

    # create the a node with the best split
    node, data_split = create_best_split_node(instances, class_values, split_feature_num)

    # if it was not possible to split with the randomly selected features, create a terminal node with the mode class
    if not data_split:
        return TerminalNode(mode(class_values)[0][0])

    # keep dividing the tree left and right
    left_child = build_random_tree(data_split.left_instances, data_split.left_class_values, split_feature_num)
    if left_child:
        node.left_child = left_child
    right_child = build_random_tree(data_split.right_instances, data_split.right_class_values, split_feature_num)
    if right_child:
        node.right_child = right_child

    return node


def get_tree_nodes(node):

    """Find all the nodes of the tree starting in the passed node"""

    # include the current node
    nodes = [node]

    # explore children
    if not node.is_terminal:
        if node.right_child:
            nodes.extend(get_tree_nodes(node.right_child))
        if node.left_child:
            nodes.extend(get_tree_nodes(node.left_child))

    return nodes


def get_features_by_importance(random_forest):

    """Determine the importance of each feature in the passed random forest"""

    feature_usages = list()

    # find the features used in all the nodes among trees
    for random_tree in random_forest:
        nodes = get_tree_nodes(random_tree)
        feature_usages.extend([node.feature_name for node in nodes if not node.is_terminal])

    # sort features by their frequency
    feature_counter = Counter(feature_usages)
    features_by_importance = sorted([(feature, round(count / len(feature_usages), 3)) for feature, count in zip(feature_counter.keys(), feature_counter.values())], key=lambda pair: pair[1], reverse=True)

    return features_by_importance


def build_random_forest_classifier(instances, class_values, tree_num, split_feature_num, bootstrap_sample_proportion):

    """Build a random forest classifier using the passed feature matrix and the class vector, and return the decision trees"""

    # trees forming the forest
    random_forest = list()

    # create decision trees
    for i in range(tree_num):

        # use a bootstrap sample of the data
        sample_size = round(len(instances) * bootstrap_sample_proportion)
        sample_instances, sample_class_values = get_bootstrap_sample(instances, class_values, sample_size)

        random_tree = build_random_tree(sample_instances, sample_class_values, split_feature_num)
        random_forest.append(random_tree)

    # features ordered by importance
    features_by_importance = get_features_by_importance(random_forest)
    print("Features by importance:", features_by_importance)

    return random_forest, features_by_importance


def predict_class_with_tree(instance, tree):

    """Used the passed decision tree to predict the class of the passed instance"""

    # go down through the tree nodes evaluating each non-terminal node's condition based on the instance features
    node = tree
    while not node.is_terminal:
        if node.evaluate(instance[node.feature_name]):
            node = node.right_child
        else:
            node = node.left_child

    # the predicted class is the one given by the reached terminal node
    return node.class_value


def evaluate_random_forest(random_forest, instances, class_values):

    """Predict the class value of the passed instances with the passed random forest, and compare with the ground truth values to assess the accuracy"""

    assigned_classes = list()
    correct_class_count = 0

    # assign a class for each instance
    for i, (_, instance) in enumerate(instances.iterrows()):

        # make all trees predict
        voted_classes = list()
        for tree in random_forest:
            voted_classes.append(predict_class_with_tree(instance, tree))

        # select the majority vote, and check if the classification is correct
        assigned_class = mode(voted_classes)[0][0]
        assigned_classes.append(assigned_class)
        if class_values.iloc[i] == assigned_class:
            correct_class_count += 1

    # compute the accuracy as the proportion of classifications that were correct
    accuracy = correct_class_count / len(instances)

    return accuracy, assigned_classes
