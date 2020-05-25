
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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns
import itertools
import NewSisterCellClass as ssc


""" input the already centered factor and target dataframe """
def linear_regression_framework(factor_df, target_df, fit_intercept):

    # Make regression matrix in a dataframe representation
    df_matrix = pd.DataFrame(index=target_df.columns, columns=factor_df.columns)

    # arrays where we will keep the information
    scores = dict()
    intercepts = dict()
    predictions = dict()

    # loop over all the regressions we'll be doing
    for t_v in target_df.columns:
        # get the fit
        reg = LinearRegression(fit_intercept=fit_intercept).fit(factor_df, target_df[t_v])
        # save the information
        scores.update({t_v: round(reg.score(factor_df, target_df[t_v]), 3)})
        intercepts.update({t_v: round(reg.intercept_, 3)})
        predictions.update({t_v: reg.predict(factor_df)})

        coefficients = [round(coef, 3) for coef in reg.coef_]
        # put it in the matrix
        df_matrix.loc[t_v] = coefficients

    return df_matrix, scores, intercepts, predictions


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

    factors = Population.mother_dfs[0].copy()
    factors['length_birth'] = np.log(factors['length_birth']) - factors['log_trap_avg_length_birth']
    factors['growth_rate'] = factors['growth_rate'] - factors['trap_avg_growth_rate']
    factors['generationtime'] = factors['generationtime'] - factors['trap_avg_generationtime']
    factors['fold_growth'] = factors['fold_growth'] - factors['trap_avg_fold_growth']
    factors['division_ratio'] = np.log(factors['division_ratio']) - factors['log_trap_avg_division_ratio']
    factors['constant'] = np.mean(factors['trap_avg_fold_growth'] + factors['log_trap_avg_division_ratio'])

    targets = Population.daughter_dfs[0].copy()
    targets['length_birth'] = np.log(targets['length_birth']) - targets['log_trap_avg_length_birth']
    targets['growth_rate'] = targets['growth_rate'] - targets['trap_avg_growth_rate']
    targets['generationtime'] = targets['generationtime'] - targets['trap_avg_generationtime']
    factors['fold_growth'] = factors['fold_growth'] - factors['trap_avg_fold_growth']
    targets['division_ratio'] = np.log(targets['division_ratio']) - targets['log_trap_avg_division_ratio']

    print(stats.pearsonr(targets['length_birth'], factors['length_birth']+factors['fold_growth']+factors['division_ratio']+factors['constant']))
    print(stats.pearsonr(targets['length_birth'], factors['length_birth'] + factors['fold_growth']))

    print(stats.pearsonr(factors['trap_avg_generationtime']*factors['trap_avg_growth_rate'], factors['log_trap_avg_division_ratio'])[0])
    print(stats.pearsonr(factors['trap_avg_fold_growth'], factors['log_trap_avg_division_ratio'])[0])

    plt.hist(factors['trap_avg_generationtime']*factors['trap_avg_growth_rate']+factors['log_trap_avg_division_ratio'])
    plt.show()
    plt.close()
    plt.hist(factors['trap_avg_fold_growth']+factors['log_trap_avg_division_ratio'])
    plt.show()
    plt.close()

    df_matrix, scores, intercepts, predictions = linear_regression_framework(factor_df=factors[['fold_growth', 'length_birth', 'division_ratio', 'constant']],
                                                                target_df=targets[['fold_growth', 'length_birth', 'division_ratio']],
                                                                fit_intercept=False)

    print(df_matrix)
    print(scores)
    print()
    print('length birth:', stats.pearsonr(targets['length_birth'], predictions['length_birth'])[0])
    print('fold growth:', stats.pearsonr(targets['fold_growth'], predictions['fold_growth'])[0])
    print('division ratio:', stats.pearsonr(targets['division_ratio'], predictions['division_ratio'])[0])


if __name__ == '__main__':
    main()
