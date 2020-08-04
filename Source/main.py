import math
import os

import numpy as np
from pathlib import Path
from data_functions import *
from random_forest import *


# printing modifiers

PRINT_BOLD_COLOR = "\033[1m"
PRINT_DISABLE_COLOR = "\033[0m"


def clear_file(file_name):

    """Clear the contents of a file"""

    # clear any previous content in the file
    open(file_name, "w").close()


def print_and_write(message, file_name):

    """Print the passed string in the console and write it to the current global file"""

    print(message)
    file = open(file_name, "a")
    file.write(message + "\n")
    file.close()


def main():

    """ Main function"""

    # show the title
    print(PRINT_BOLD_COLOR + "Random Forest Classifier" + PRINT_DISABLE_COLOR + "\n")

    # path to the data sets
    parent_dir_path = os.path.dirname(os.getcwd())#str(Path(__file__).parents[1])
    data_dir_path = parent_dir_path + "/Data/"
    results_dir_path = parent_dir_path + "/Results/"

    # data sets to use
    data_set_names = ["contact-lenses", "labor", "hepatitis", "breast-cancer", "car"]

    # read the data sets as data frames
    data_frames = read_data_sets(data_dir_path, data_set_names)

    # pre-process the data sets
    data_frames = [pre_process_data_frame(data_frame, False) for data_frame in data_frames]

    for data_set_name, data_frame in zip(data_set_names, data_frames):

        # prepare the file where to write the results
        data_set_file_name = results_dir_path + data_set_name + ".txt"
        clear_file(data_set_file_name)

        print_and_write("Data set: {}".format(data_set_name), data_set_file_name)

        # proportion to define the size of bootstrap samples
        bootstrap_sample_proportion = 1

        # proportion of data to use as test
        test_proportion = max(0.1, 5 / len(data_frame))

        # split the data in training and test
        x_train, y_train, x_test, y_test = split_data_frame(data_frame, test_proportion)

        # total number of features
        feature_num = len(data_frame.columns)
        log_feature_num = round(math.log2(feature_num) + 1)
        root_feature_num = round(math.sqrt(feature_num))

        # combinations of parameters for the random forest (number of trees and number of splitting features)
        parameter_combinations = list()
        for tree_num in [50, 100]:
            split_feature_nums = [1, 3]
            if log_feature_num not in split_feature_nums:
                split_feature_nums.append(log_feature_num)
            if root_feature_num not in split_feature_nums:
                split_feature_nums.append(root_feature_num)
            for split_feature_num in split_feature_nums:
                parameter_combinations.append((tree_num, split_feature_num))
        parameter_combinations.sort()
        accuracies = list()
        feature_lists = list()

        # build a model for each set of parameters
        for (tree_num, split_feature_num) in parameter_combinations:

            print_and_write("\tTraining model for {} data set (number of trees: {}, number of splitting features: {})".format(data_set_name, tree_num, split_feature_num), data_set_file_name)

            # use the training set to build the classifier and get the features ordered by importance
            random_forest, features_by_importance = build_random_forest_classifier(x_train, y_train, tree_num, split_feature_num, bootstrap_sample_proportion)

            # show the features in order of importance
            print_and_write("\t\tFeatures (in order of importance):", data_set_file_name)
            for i, feature in enumerate(features_by_importance):
                print_and_write("\t\t\t({}) {}".format(i+1, feature), data_set_file_name)

            # evaluate the test set to assess the classifier's accuracy
            accuracy, _ = evaluate_random_forest(random_forest, x_test, y_test)
            accuracies.append(accuracy)

            feature_lists.append(features_by_importance)

            # show the accuracy
            print_and_write("\t\tTest accuracy: {}".format(round(accuracy, 3)), data_set_file_name)

        # compute the mean accuracy
        mean_accuracy = float(np.mean(accuracies))
        accuracy_std = float(np.std(accuracies))
        print_and_write("\tMean accuracy (among parameter setups): {} ± {}\n\n".format(round(mean_accuracy, 3), round(accuracy_std, 3)), data_set_file_name)

        # save the accuracy information in a separate tabular file
        results_data_frame = pd.DataFrame({"Tree num": [parameters[0] for parameters in parameter_combinations], "Split feature num": [parameters[1] for parameters in parameter_combinations], "Accuracy": accuracies, "Features by importance": feature_lists})
        accuracy_file_path = results_dir_path + data_set_name + "_accuracy.xlsx"
        clear_file(accuracy_file_path)
        writer = pd.ExcelWriter(accuracy_file_path)
        results_data_frame.to_excel(writer, "Matrix")
        writer.save()


if __name__ == "__main__":
    main()
