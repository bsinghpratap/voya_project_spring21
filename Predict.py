#!/usr/bin/env python
# coding: utf-8
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from sklearn.metrics import r2_score,  mean_absolute_percentage_error,  mean_squared_error,  mean_absolute_error
import joblib
import argparse
import pickle
from collections import Counter
import imblearn
from imblearn.over_sampling import SMOTE

RANDOM_STATE = 5

# JOB_TYPES = ["lda", "freq", "tfidf"]
# TARGET_VARIABLES = ["is_dps_cut", "d_environmental"]


def print_metrics_classif(y_real, y_predicted):
    accuracy = accuracy_score(y_real, y_predicted)
    precision = precision_score(y_real, y_predicted)
    recall = recall_score(y_real, y_predicted)
    f1 = f1_score(y_real, y_predicted)
    coh_kap_score = cohen_kappa_score(y_real, y_predicted)
    print("Accuracy: {:.4f}".format(accuracy))
    print("Precision: {:.4f}".format(precision))
    print("Recall: {:.4f}".format(recall))
    print("F1-score: {:.4f}".format(f1))
    print("Cohen-Kappa: {:.4f}".format(coh_kap_score))
    print(classification_report(y_real, y_predicted, target_names=["no_cut", "yes_cut"]))
    
    cm = confusion_matrix(y_real, y_predicted)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm.diagonal())
    
def train_and_validate_classification(x_vals_train, y_vals_train, x_vals_valid, y_vals_valid, max_depth, samp_strat=None, class_weight=None):
    if class_weight:
        rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, max_depth=max_depth, class_weight=class_weight)
    elif samp_strat:
        rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, max_depth=max_depth)
        smote = SMOTE(sampling_strategy=samp_strat, random_state=RANDOM_STATE)
        x_vals_train, y_vals_train = smote.fit_resample(x_vals_train, y_vals_train)
    
    # Fit the model
    rf.fit(X=x_vals_train, y=y_vals_train)

    # Prediction metrics
    y_vals_valid_predicted = rf.predict(x_vals_valid)
    coh_kap_score = cohen_kappa_score(y_vals_valid, y_vals_valid_predicted)
    return coh_kap_score, (rf, depth, samp_strat, class_weight)
    
    
def print_metrics_reg(y_real, y_predicted):
    r2 = r2_score(y_real, y_predicted)
    mape = mean_absolute_percentage_error(y_real, y_predicted)
    mse = mean_squared_error(y_real, y_predicted)
    mae = mean_absolute_error(y_real, y_predicted)
    print("R2: {:.4f}".format(r2))
    print("mape: {:.4f}".format(mape))
    print("mse: {:.4f}".format(mse))
    print("mae: {:.4f}".format(mae))
    
def train_and_validate_regression(x_vals_train, y_vals_train, x_vals_valid, y_vals_valid, max_depth, criteria):
    print("Max_Depth: {}".format(max_depth))
    print("Criteria: {}".format(criteria))
    rf = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, max_depth=max_depth, criterion=criteria)
    
    # Fit the model
    rf.fit(X=x_vals_train, y=y_vals_train)
    
    y_predicted = rf.predict(x_vals_valid)
    r2 = r2_score(y_vals_valid, y_predicted)
    return r2, (rf, depth, criteria)


# Parsing
parser = argparse.ArgumentParser()
JOB_TYPES = ["lda", "freq", "tfidf"]
TARGET_VARIABLES = ["is_dps_cut", "d_environmental"]

# Required or binary
parser.add_argument("--job_type", required=True, choices=JOB_TYPES, help="Feature type of prediction job")
parser.add_argument("--target", required=True, choices=TARGET_VARIABLES, help="Column name of prediction job")
parser.add_argument("--input_folder", required=True)
parser.add_argument("--output_folder",  required=True)

# Optional
parser.add_argument("-ws", "--window_size", type = int, required=False, help="For ppt sentence_lda, number of sentences in a document")

args = parser.parse_args()
print(args)

# Fetch settings
job_type = args.job_type
target = args.target
input_path = args.input_folder
output_folder = args.output_folder
if job_type == "lda":    
    window_size = args.window_size
    if not window_size:
        print("Window size not set for LDA. Quitting")
        quit()
else:
    window_size = None



print(job_type)
print(target)
print(window_size)
print("--------------------------------------------------------------")


# Variables
start_year = 2012
end_year = 2015
valid_year = 2016
test_year = 2017

for i in range(0,3):
    print("Processing {}_{} and {}_{}".format(start_year, end_year, valid_year, test_year))

    # Infer input file names
    if job_type == "lda":
        train_path = input_path + "_".join(["data", "train",str(window_size),str(start_year),str(end_year)]) + ".pkl"
        test_path = input_path + "_".join(["data","test","valid",str(window_size),str(valid_year),str(test_year)]) + ".pkl"
    else:
        train_path = input_path + "_".join(["baseline", "train",str(start_year),str(end_year)]) + ".pkl"
        test_path = input_path + "_".join(["baseline", "test",str(valid_year),str(test_year)]) + ".pkl"

    # Load data
    data_train = pd.read_pickle(train_path)
    data_test_valid = pd.read_pickle(test_path)

    # Reduce data based on target
    if target == "is_dps_cut":
        # Train
        data_train["is_dividend_payer"] = data_train["is_dividend_payer"].astype(bool)
        data_train = data_train[data_train["is_dividend_payer"] & data_train["is_dps_cut"].notnull()]
        data_train["is_dps_cut"] = data_train["is_dps_cut"].astype(int)

        # Test/Valid
        data_test_valid["is_dividend_payer"] = data_test_valid["is_dividend_payer"].astype(bool)
        data_test_valid = data_test_valid[data_test_valid["is_dividend_payer"] & data_test_valid["is_dps_cut"].notnull()]
        data_test_valid["is_dps_cut"] = data_test_valid["is_dps_cut"].astype(int)
    elif target == "d_environmental":
        # Train
        data_train = data_train[data_train["d_environmental"].notna()]

        # Test/Valid
        data_test_valid = data_test_valid[data_test_valid["d_environmental"].notna()]
    else:
        print("ERROR! invaid target")
        quit()

    # Split valid and test
    data_valid = data_test_valid[data_test_valid.year_x == valid_year]
    data_test = data_test_valid[data_test_valid.year_x == test_year]

    # Different training weights depending on job_type
    if job_type == "lda":
        train_weights = data_train.loc[:,"risk_topic_0":].to_numpy().tolist()
        valid_weights = data_valid.loc[:,"risk_topic_0":].to_numpy().tolist()
        test_weights = data_test.loc[:,"risk_topic_0":].to_numpy().tolist()
    else:
        if job_type == "freq":
            feature_regex = "freq*"
        elif job_type == "tfidf":
            feature_regex = "tfidf*"
        train_weights = data_train.filter(regex=feature_regex).to_numpy().tolist()
        valid_weights = data_valid.filter(regex=feature_regex).to_numpy().tolist()
        test_weights = data_test.filter(regex=feature_regex).to_numpy().tolist()
        


    # Labels always the same
    train_labels = data_train.loc[:,target].to_list()
    valid_labels = data_valid.loc[:,target].to_list()
    test_labels = data_test.loc[:,target].to_list()

    best_run_score = 0
    best_run_params = None
    depths = [3,5,7]
    if target == "is_dps_cut":   # Classification
        # Class weights for imbalanced data
        counter = Counter(train_labels)
        total = float(sum(list(counter.values())))
        class_weights = {0: total/float(counter.get(0)), 1: total/float(counter.get(1))}

        sample_strategies = [.1,.3,.5,.7,1]
        best_run_score = 0

        # Try with SMOTE and just classweights
        for depth in depths:
            score, params  = train_and_validate_classification(train_weights, train_labels, valid_weights, valid_labels, depth, class_weight=class_weights)
            if score > best_run_score:
                best_run_score = score
                best_run_params = params
            for samp_strat in sample_strategies:
                score, params = train_and_validate_classification(train_weights, train_labels, valid_weights, valid_labels, depth, samp_strat=samp_strat)
                if score > best_run_score:
                    best_run_score = score
                    best_run_params = params
        rf = best_run_params[0]

        # Report validation and testing scores
        print("Best validation score: {:.4f}, Params: ({}, {}, {})".format(best_run_score, best_run_params[1], best_run_params[2], best_run_params[3]))
        print_metrics_classif(test_labels, rf.predict(test_weights))
    else:   # Regression
        criteria = ["mse", "mae"]
        for depth in depths:
            for crit in criteria:
                score, params = train_and_validate_regression(train_weights, train_labels, valid_weights, valid_labels, depth, crit)
                if score > best_run_score:
                    best_run_score = score
                    best_run_params = params

        rf = best_run_params[0]
        print("Best validation score: {:.4f}, Params: ({}, {})".format(best_run_score, best_run_params[1], best_run_params[2]))
        print_metrics_reg(test_labels, rf.predict(test_weights))



    # Write RandomForest to disk
    output_rf_path = output_folder + "rf_"
    if job_type == "lda":
        output_rf_path += "_".join([job_type, target, str(window_size), str(start_year), str(end_year), str(valid_year), str(test_year)])
    else:
        output_rf_path += "_".join([job_type, target, str(start_year), str(end_year), str(valid_year), str(test_year)])
    output_rf_path += ".joblib"
    joblib.dump(rf, output_rf_path)

    start_year += 1
    end_year += 1
    valid_year += 1
    test_year += 1
