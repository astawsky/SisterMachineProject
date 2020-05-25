import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys, math
import glob
import pickle
import os
import scipy.stats as stats
from scipy import signal
import random
import sklearn
from sklearn.model_selection import train_test_split
import seaborn as sns
import itertools
import NewSisterCellClass as ssc
import pymc3 as pm
from scipy.special import kn as bessel_function_of_second_kind
from scipy.special import comb as combinations


def subtract_traj_averages(df, columns_names, log_vars):
    df_new = pd.DataFrame(columns=columns_names)
    for col in columns_names:
        if col in log_vars:
            df_new[col] = np.log(df[col]) - df['log_traj_avg_' + col]
        else:
            df_new[col] = df[col] - df['traj_avg_' + col]

    return df_new


def subtract_trap_averages(df, columns_names, log_vars):
    df_new = pd.DataFrame(columns=columns_names)
    for col in columns_names:
        if col in log_vars:
            df_new[col] = np.log(df[col]) - df['log_trap_avg_' + col]
        else:
            df_new[col] = df[col] - df['trap_avg_' + col]

    return df_new


def subtract_global_averages(df, columns_names, log_vars):
    df_new = pd.DataFrame(columns=columns_names)
    for col in columns_names:
        if col in log_vars:
            df_new[col] = np.log(df[col]) - np.mean(np.log(df[col]))
        else:
            df_new[col] = df[col] - np.mean(df[col])

    return df_new


def normal_entropy(std):
    # NOOOOOOOT This one came from the Normal page on Wikipedia
    return np.log(std*np.sqrt(2*np.pi*np.exp(1)))


def multivariate_normal_KL_divergence(mean_0, cov_0, mean_1, cov_1):
    # Definition of Mutual Information! if N_0 is the multivariate distribution and N_1 is the product of all the marginal distributions

    # D_{KL}(N_0||N_1), which is not necessarily the same as D_{KL}(N_1||N_0)!

    # This one came from the Multivariate Normal page on Wikipedia

    # sanity check, doesn't make sense to compare distributions of different dimensions
    if cov_0.shape != cov_1.shape:
        print('cov_0.shape != cov_1.shape, ERROR!')
        exit()
    elif cov_0.shape[0] != cov_0.shape[1]:
        # if it is not a square covariance matrix (Doesn't make sense)
        print('The covariance matrices are not square!!! ERROR!')
        exit()

    k = cov_0.shape[0]
    inv_1 = np.linalg.inv(cov_1)
    diff_means = mean_1 - mean_0

    return .5 * (np.trace(np.dot(inv_1, cov_0)) + np.dot(diff_means.T, np.dot(inv_1, diff_means)) - k + np.log(np.linalg.det(cov_1) / np.linalg.det(cov_0)))


def multivariate_normal_diff_entropy(co_var_mat):
    # This one came from the Multivariate Normal page on Wikipedia
    print(co_var_mat.shape)
    exit()
    # this is the dimensionality of the matrix
    k = co_var_mat.shape[0]
    the_det = np.linalg.det(co_var_mat, 2)

    # same thing as .5 * np.log( ((2*np.pi*np.exp(1))**k) * np.linalg.det(co_var_mat) )
    return (k/2) + (k/2) * np.log(2*np.pi) + .5*np.log(the_det)


def main():
    # Import the objects we want
    pickle_in = open("NewSisterCellClass_Population.pickle", "rb")
    Population = pickle.load(pickle_in)
    pickle_in.close()

    pickle_in = open("NewSisterCellClass_Sister.pickle", "rb")
    Sister = pickle.load(pickle_in)
    pickle_in.close()

    pickle_in = open("NewSisterCellClass_Nonsister.pickle", "rb")
    Nonsister = pickle.load(pickle_in)
    pickle_in.close()

    pickle_in = open("NewSisterCellClass_Control.pickle", "rb")
    Control = pickle.load(pickle_in)
    pickle_in.close()

    # renaming
    mom = Population.mother_dfs[0].copy()
    daug = Population.daughter_dfs[0].copy()

    # because it is a false cycle and has a negative growth rate which is impossible
    mom = mom.drop(index=1479)
    daug = daug.drop(index=1478)

    all_vars = ['generationtime', 'length_birth', 'length_final', 'growth_rate', 'fold_growth', 'division_ratio']

    # generationtime and growth rate are log normal
    mom_global = subtract_global_averages(df=mom, columns_names=Population._variable_names, log_vars=['generationtime', 'growth_rate', 'length_birth'])
    mom_trap = subtract_trap_averages(df=mom, columns_names=Population._variable_names, log_vars=['generationtime', 'growth_rate', 'length_birth'])
    mom_traj = subtract_traj_averages(df=mom, columns_names=Population._variable_names, log_vars=['generationtime', 'growth_rate', 'length_birth'])

    daug_global = subtract_global_averages(df=daug, columns_names=Population._variable_names, log_vars=['generationtime', 'growth_rate', 'length_birth'])
    daug_trap = subtract_trap_averages(df=daug, columns_names=Population._variable_names, log_vars=['generationtime', 'growth_rate', 'length_birth'])
    daug_traj = subtract_traj_averages(df=daug, columns_names=Population._variable_names, log_vars=['generationtime', 'growth_rate', 'length_birth'])









if __name__ == '__main__':
    main()
