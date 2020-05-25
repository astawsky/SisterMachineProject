
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
import pymc3 as pm


# Make a new prediction from the test set and compare to actual value
def test_model(trace, test_observation, factors, target):
    # Print out the test observation data
    print('Test Observation:')
    print(test_observation[factors])
    var_dict = {}
    for variable in trace.varnames:
        var_dict[variable] = trace[variable]

    # Results into a dataframe
    var_weights = pd.DataFrame(var_dict)

    # Standard deviation of the likelihood
    sd_value = var_weights['sd'].mean()

    # Actual Value
    actual = test_observation[target]

    # Add in intercept term
    test_observation['Intercept'] = np.mean(var_dict['Intercept'])
    test_observation = test_observation.drop(target)

    # Align weights and test observation
    var_weights = var_weights[factors]

    # Means for all the weights
    var_means = var_weights.mean(axis=0)

    # Location of mean for observation
    mean_loc = np.dot(var_means, test_observation[factors])

    # Estimates of grade
    estimates = np.random.normal(loc=mean_loc, scale=sd_value,
                                 size=1000)

    # Plot all the estimates
    plt.figure()
    sns.distplot(estimates, hist=True, kde=True, bins=19,
                 hist_kws={'edgecolor': 'k', 'color': 'darkblue'},
                 kde_kws={'linewidth': 4},
                 label='Estimated Dist.')
    # Plot the actual grade
    plt.vlines(x=actual, ymin=0, ymax=5,
               linestyles='--', colors='red',
               label='True Grade',
               linewidth=2.5)

    # Plot the mean estimate
    plt.vlines(x=mean_loc, ymin=0, ymax=5,
               linestyles='-', colors='orange',
               label='Mean Estimate',
               linewidth=2.5)

    plt.legend(loc=1)
    plt.title('Density Plot for Test Observation')
    plt.xlabel('Grade')
    plt.ylabel('Density')

    # Prediction information
    print('True Grade = %d' % actual)
    print('Average Estimate = %0.4f' % mean_loc)
    print('5%% Estimate = %0.4f    95%% Estimate = %0.4f' % (np.percentile(estimates, 5),
                                                             np.percentile(estimates, 95)))


def subtract_traj_averages(df, columns_names):
    df_new = pd.DataFrame(columns=columns_names)
    for col in columns_names:
        df_new[col] = df[col] - df['traj_avg_' + col]

    return df_new


def subtract_trap_averages(df, columns_names):
    df_new = pd.DataFrame(columns=columns_names)
    for col in columns_names:
        df_new[col] = df[col] - df['trap_avg_'+col]
    
    return df_new


def subtract_global_averages(df, columns_names):
    df_new = pd.DataFrame(columns=columns_names)
    for col in columns_names:
        df_new[col] = df[col] - np.mean(df[col])

    return df_new


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

    # print(mom.columns)
    
    mom_global = subtract_global_averages(df=mom, columns_names=Population._variable_names)
    mom_trap = subtract_trap_averages(df=mom, columns_names=Population._variable_names)
    mom_traj = subtract_traj_averages(df=mom, columns_names=Population._variable_names)

    daug_global = subtract_global_averages(df=daug, columns_names=Population._variable_names)
    daug_trap = subtract_trap_averages(df=daug, columns_names=Population._variable_names)
    daug_traj = subtract_traj_averages(df=daug, columns_names=Population._variable_names)

    count = 0
    for mother, daughter in zip([mom_global, mom_trap, mom_traj], [daug_global, daug_trap, daug_traj]):
        print(count)
        X_train, X_test, y_train, y_test = train_test_split(mom, daughter, test_size=0.33)

        # Context for the model
        with pm.Model() as normal_model:

            data = pd.concat([X_train.rename(columns=dict(zip(X_train.columns, [col+'_m' for col in X_train.columns]))),
                              y_train.rename(columns=dict(zip(y_train.columns, [col+'_d' for col in y_train.columns])))], axis=1)

            tests = pd.concat([X_test.rename(columns=dict(zip(X_test.columns, [col+'_m' for col in X_test.columns]))),
                              y_test.rename(columns=dict(zip(y_test.columns, [col+'_d' for col in y_test.columns])))], axis=1)

            factors = [col+'_m' for col in y_train.columns]
            target = 'generationtime_d'

            # The prior for the data likelihood is a Normal Distribution
            family = pm.glm.families.Normal()

            # Creating the model requires a formula and data (and optionally a family)
            pm.GLM.from_formula(formula=str(target+' ~ '+' + '.join(factors)), data=data, family=family)

            # Perform Markov Chain Monte Carlo sampling letting PyMC3 choose the algorithm
            normal_trace = pm.sample(draws=500, step=None, init='auto', n_init=200000, start=None, trace=None, chain_idx=0, chains=None, cores=1,
                                     tune=500, progressbar=True, model=None, random_seed=None, discard_tuned_samples=True, compute_convergence_checks=True)

            pm.traceplot(normal_trace)
            plt.show()

            pm.plot_posterior(normal_trace)
            plt.show()

            print(pm.summary(normal_trace))

            for num in range(len(tests)):
                # print(normal_trace.varnames)
                # print(normal_trace['Intercept'])
                # print(np.mean(normal_trace['Intercept']))
                test_model(normal_trace, tests.iloc[num], factors=factors, target=target)
                plt.show()
                plt.close()



        count=count+1
    
    
    exit()

    # copy them so we don't mess up the original mother and daughter dfs
    mom_global = mom.copy()
    mom_trap = mom.copy()
    mom_traj = mom.copy()
    daug_global = daug.copy()
    daug_trap = daug.copy()
    daug_traj = daug.copy()


    # SECOND WAY! Subtracting by the mean of the log of the original
    # log the length and center the fold growth and length for all three average-specific m/d dataframes
    print('2. Subtracting by the mean of the log of the original')
    mom_global['length_birth'] = np.log(mom_global['length_birth']) - np.mean(np.log(mom_global['length_birth']))
    daug_global['length_birth'] = np.log(daug_global['length_birth']) - np.mean(np.log(daug_global['length_birth']))
    mom_global['fold_growth'] = mom_global['fold_growth'] - np.mean(mom_global['fold_growth'])
    daug_global['fold_growth'] = daug_global['fold_growth'] - np.mean(daug_global['fold_growth'])

    mom_trap['length_birth'] = np.log(mom_trap['length_birth']) - mom_trap['log_trap_avg_length_birth']
    daug_trap['length_birth'] = np.log(daug_trap['length_birth']) - daug_trap['log_trap_avg_length_birth']
    mom_trap['fold_growth'] = mom_trap['fold_growth'] - mom_trap['trap_avg_fold_growth']
    daug_trap['fold_growth'] = daug_trap['fold_growth'] - daug_trap['trap_avg_fold_growth']

    mom_traj['length_birth'] = np.log(mom_traj['length_birth']) - mom_traj['log_traj_avg_length_birth']
    daug_traj['length_birth'] = np.log(daug_traj['length_birth']) - daug_traj['log_traj_avg_length_birth']
    mom_traj['fold_growth'] = mom_traj['fold_growth'] - mom_traj['traj_avg_fold_growth']
    daug_traj['fold_growth'] = daug_traj['fold_growth'] - daug_traj['traj_avg_fold_growth']



if __name__ == '__main__':
    main()
