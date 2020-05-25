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


def subtract_traj_averages(df, columns_names, log_vars=['generationtime', 'growth_rate', 'length_birth']):
    df_new = pd.DataFrame(columns=columns_names)
    for col in columns_names:
        if col in log_vars:
            df_new[col] = df[col] - df['log_traj_avg_' + col]
        else:
            df_new[col] = df[col] - df['traj_avg_' + col]

    return df_new


def subtract_trap_averages(df, columns_names, log_vars=['generationtime', 'growth_rate', 'length_birth']):
    df_new = pd.DataFrame(columns=columns_names)
    for col in columns_names:
        if col in log_vars:
            df_new[col] = df[col] - df['log_trap_avg_' + col]
        else:
            df_new[col] = df[col] - df['trap_avg_' + col]

    return df_new


def subtract_global_averages(df, columns_names):
    df_new = pd.DataFrame(columns=columns_names)
    for col in columns_names:
        df_new[col] = df[col] - np.mean(df[col])

    return df_new


def conditional_probability_multinormal(daughter, mother):
    # daughter and mother are dataframes that contain the targets and factors respectively

    # choose a specific instance of each of the daughters variables
    for variable in daughter.columns:
        from_model = []
        var_from_model = []
        from_data = []
        prob_from_model = []
        for sample in range(len(daughter)):
            # print('variable', variable)
            # print(daughter[variable].mean())
            # print([np.cov(daughter[variable], mother[col])[1, 0] for col in mother.columns])

            dependancy = np.array([np.cov(daughter[variable], mother[col])[1, 0] for col in mother.columns])
            # print(dependancy)
            # print(mother.cov())
            cond_mean = daughter[variable].mean() + np.dot(dependancy, np.dot(np.linalg.inv(np.array(mother.cov())), (mother[variable].iloc[sample]-mother.mean())))
            cond_variance = daughter[variable].var() - np.dot(dependancy, np.dot(np.linalg.inv(np.array(mother.cov())), dependancy.T))
            cond_std = np.sqrt(cond_variance)
            error = [1]
            # probability1 = (1/cond_std) * (1/np.sqrt((2*np.pi))) * np.exp( -(1/2) * ((daughter[variable].iloc[sample]-cond_mean) / cond_std))
            # probability = (1/(cond_std*np.sqrt(2*np.pi)))*\
            #               np.exp((-.5)*(((daughter[variable].iloc[sample]-cond_mean)/cond_std)**2))
            # probability2 = stats.norm(cond_mean, cond_std).pdf(daughter[variable].iloc[sample])
            # print(probability, cond_mean, cond_variance)
            from_model.append(cond_mean)
            from_data.append(daughter[variable].iloc[sample])
            var_from_model.append(cond_variance)
            # prob_from_model.append(probability)
            # print(probability)
            # print(probability1)
            # print(cond_mean)
            # print(probability2)
            # print('------')
            # print((-.5)*(((daughter[variable].iloc[sample]-cond_mean)/cond_std)**2))
            # print(1/(cond_std*np.sqrt(2*np.pi)))
            # exit()
            # simulation = np.random.normal(loc=cond_mean, scale=np.sqrt(cond_variance), size=500)
            # plt.hist(simulation)
            # plt.axvline(x=daughter[variable].iloc[sample], color='orange')
            # plt.axvline(x=cond_mean, color='black')
            # plt.xlim([-1, 1])
            # plt.show()
            # plt.close()
            # exit()
        # print(np.mean(prob_from_model))
        # print(np.std(prob_from_model))
        # sns.distplot(prob_from_model, label=r'${}\pm{}$'.format(round(np.mean(prob_from_model)[0], 2), round(np.std(prob_from_model)[0], 2)))
        # sns.regplot(from_model, from_data, label='{}'.format(round(stats.pearsonr(from_model, from_data)[0], 3)), line_kws={'color': 'orange'})
        # g = sns.jointplot(from_model, from_data, kind="kde")
        g = sns.kdeplot(from_model, from_data, shade=False)
        sns.regplot(from_model, from_data, ax=g, scatter=False, label='{}'.format(round(stats.pearsonr(from_model, from_data)[0], 3)), line_kws={'color': 'orange'})
        # sns.jointplot(from_model, from_data, kind="reg", label='{}'.format(round(stats.pearsonr(from_model, from_data)[0], 3)), line_kws={'color': 'orange'})
        plt.legend()
        plt.title(variable)
        plt.show()
        plt.close()


def conditional_joint_prob_data(x1_mean, x2_mean, ccov_11, ccov_12, ccov_21, ccov_22, a):
    new_mean = x1_mean + np.dot(ccov_12, np.dot(np.linalg.inv(ccov_22), np.array(a - x2_mean)))
    new_variance = ccov_11 - np.dot(ccov_12, np.dot(np.linalg.inv(ccov_22), ccov_21))

    return new_mean, new_variance


def conditional_joint_probability_multinormal(x1, x2, a):
    a = a.rename(index=dict(zip(a.index, [ind+'_x2' for ind in a.index])))

    if isinstance(x1, pd.Series):
        x1 = x1.rename(str(x1.name) + '_x1')

        if isinstance(x2, pd.Series):
            x2.name = str(x2.name) + '_x2'
            # daughter and mother are dataframes that contain the targets and factors respectively
            cross_covariance_12 = np.array(pd.concat([x1, x2], axis=1).cov())
            cross_covariance_21 = np.array(pd.concat([x1, x2], axis=1).cov())
            new_mean = x1.mean() + np.dot(cross_covariance_12, np.dot(np.linalg.inv(np.array(x2.var())), np.array(a - x2.mean())))
            new_variance = x1.var() - np.dot(cross_covariance_12, np.dot(np.linalg.inv(np.array(x2.var())), cross_covariance_21))
        else:
            x2 = x2.rename(columns=dict(zip(x2.columns, [col + '_x2' for col in x2.columns])))
            # daughter and mother are dataframes that contain the targets and factors respectively
            cross_covariance_12 = np.array(pd.concat([x1, x2], axis=1).cov()[x2.columns].loc[x1.name])
            cross_covariance_21 = np.array(pd.concat([x1, x2], axis=1).cov()[x1.name].loc[x2.columns])
            # print(pd.concat([x1, x2], axis=1).cov())
            # print(np.array(a - x2.mean()))
            # print(np.linalg.inv(np.array(x2.cov())))
            # print(x2.cov())
            new_mean = x1.mean() + np.dot(cross_covariance_12, np.dot(np.linalg.inv(np.array(x2.cov())), np.array(a - x2.mean())))
            new_variance = x1.var() - np.dot(cross_covariance_12, np.dot(np.linalg.inv(np.array(x2.cov())), cross_covariance_21))
    else:
        x1 = x1.rename(columns=dict(zip(x1.columns, [col + '_x1' for col in x1.columns])))
        if isinstance(x2, pd.Series):
            x2.name = str(x2.name) + '_x2'
            # daughter and mother are dataframes that contain the targets and factors respectively
            cross_covariance_12 = np.array(pd.concat([x1, x2], axis=1).cov()[x2.name].loc[x1.columns])
            cross_covariance_21 = np.array(pd.concat([x1, x2], axis=1).cov()[x1.columns].loc[x2.name])
            new_mean = x1.mean() + np.dot(cross_covariance_12, np.dot(np.linalg.inv(np.array(x2.var())), np.array(a - x2.mean())))
            new_variance = x1.cov() - np.dot(cross_covariance_12, np.dot(np.linalg.inv(np.array(x2.var())), cross_covariance_21))
        else:
            x2 = x2.rename(columns=dict(zip(x2.columns, [col + '_x2' for col in x2.columns])))
            # daughter and mother are dataframes that contain the targets and factors respectively
            cross_covariance_12 = np.array(pd.concat([x1, x2], axis=1).cov()[x2.columns].loc[x1.columns])
            cross_covariance_21 = np.array(pd.concat([x1, x2], axis=1).cov()[x1.columns].loc[x2.columns])

            keep_count = 0
            if not np.isfinite(cross_covariance_21).all():
                print('_21', cross_covariance_21)
                keep_count += 1
            if not np.isfinite(cross_covariance_12).all() > 0:
                print('_12', cross_covariance_12)
                keep_count += 1
            if x1.cov().isnull().values.sum() > 0:
                print('_11')
                with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
                    print(x1.cov())
                keep_count += 1
            if x2.cov().isnull().values.sum() > 0:
                print('_22')
                with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
                    print(x2.cov())
                keep_count += 1
            if not np.isfinite(np.array(a - x2.mean())).all():
                print('a', a)
                print('x2.mean()', x2.mean())
                print('np.array(a - x2.mean()):')
                print(np.array(a - x2.mean()))
                keep_count += 1
            if not np.isfinite(np.linalg.inv(np.array(x2.cov()))).all():
                print('np.linalg.inv(np.array(x2.cov())):')
                print(np.linalg.inv(np.array(x2.cov())))
                keep_count += 1
            if not np.isfinite(np.dot(np.linalg.inv(np.array(x2.cov())), np.array(a - x2.mean()))).all():
                print('np.dot(np.linalg.inv(np.array(x2.cov())), np.array(a - x2.mean())):')
                print(np.dot(np.linalg.inv(np.array(x2.cov())), np.array(a - x2.mean())))
                keep_count += 1

            if keep_count > 0:
                exit()

            # print('_12', cross_covariance_12)
            # print('_11', x1.cov())
            # print('_22', x2.cov())
            # exit()
            # print(np.array(a - x2.mean()).shape)
            # print(np.linalg.inv(np.array(x2.cov())).shape)
            # print(np.dot(np.linalg.inv(np.array(x2.cov())), np.array(a - x2.mean())))
            new_mean = np.array(x1.mean()) + np.dot(cross_covariance_12, np.dot(np.linalg.inv(np.array(x2.cov())), np.array(a - x2.mean())))
            new_variance = np.array(x1.cov()) - np.dot(cross_covariance_12, np.dot(np.linalg.inv(np.array(x2.cov())), cross_covariance_21))
            # exit()


    # if isinstance(x1, pd.Series) and isinstance(x2, pd.DataFrame):
    #     # daughter and mother are dataframes that contain the targets and factors respectively
    #     cross_covariance_12 = np.array(pd.concat([x1, x2], axis=1).cov()[x2.columns].loc[x1.name])
    #     cross_covariance_21 = np.array(pd.concat([x1, x2], axis=1).cov()[x1.name].loc[x2.columns])
    #     print(pd.concat([x1, x2], axis=1).cov())
    #     print(np.array(a - x2.mean()))
    #     print(np.linalg.inv(np.array(x2.cov())))
    #     print(x2.cov())
    #     new_mean = x1.mean() + np.dot(cross_covariance_12, np.dot(np.linalg.inv(np.array(x2.cov())), np.array(a - x2.mean())))
    #     new_variance = x1.var() - np.dot(cross_covariance_12, np.dot(np.linalg.inv(np.array(x2.cov())), cross_covariance_21))
    #     print(1)
    # elif isinstance(x2, pd.Series) and isinstance(x1, pd.DataFrame):
    #     # daughter and mother are dataframes that contain the targets and factors respectively
    #     cross_covariance_12 = np.array(pd.concat([x1, x2], axis=1).cov()[x2.name].loc[x1.columns])
    #     cross_covariance_21 = np.array(pd.concat([x1, x2], axis=1).cov()[x1.columns].loc[x2.name])
    #     new_mean = x1.mean() + np.dot(cross_covariance_12, np.dot(np.linalg.inv(np.array(x2.var())), np.array(a - x2.mean())))
    #     new_variance = x1.cov() - np.dot(cross_covariance_12, np.dot(np.linalg.inv(np.array(x2.var())), cross_covariance_21))
    #     print(2)
    # elif isinstance(x1, pd.Series) and isinstance(x2, pd.Series):
    #     # daughter and mother are dataframes that contain the targets and factors respectively
    #     cross_covariance_12 = np.array(pd.concat([x1, x2], axis=1).cov())
    #     cross_covariance_21 = np.array(pd.concat([x1, x2], axis=1).cov())
    #     new_mean = x1.mean() + np.dot(cross_covariance_12, np.dot(np.linalg.inv(np.array(x2.var())), np.array(a - x2.mean())))
    #     new_variance = x1.var() - np.dot(cross_covariance_12, np.dot(np.linalg.inv(np.array(x2.var())), cross_covariance_21))
    #     print(3)
    # elif isinstance(x1, pd.DataFrame) and isinstance(x2, pd.DataFrame):
    #     # daughter and mother are dataframes that contain the targets and factors respectively
    #     cross_covariance_12 = np.array(pd.concat([x1, x2], axis=1).cov()[x2.columns].loc[x1.columns])
    #     cross_covariance_21 = np.array(pd.concat([x1, x2], axis=1).cov()[x1.columns].loc[x2.columns])
    #     new_mean = x1.mean() + np.dot(cross_covariance_12, np.dot(np.linalg.inv(np.array(x2.cov())), np.array(a - x2.mean())))
    #     new_variance = x1.cov() - np.dot(cross_covariance_12, np.dot(np.linalg.inv(np.array(x2.cov())), cross_covariance_21))
    #     print(4)

    return new_mean, new_variance


def between_predictions_with_sample_covariance():
    pass


def between_predictions_with_data_covariance(df, daug, targets=['generationtime', 'length_birth', 'length_final', 'growth_rate', 'fold_growth', 'division_ratio'],
                               factors=None,  how_many_samples=1000,
                               all_variable_names=['generationtime', 'length_birth', 'length_final', 'growth_rate', 'fold_growth', 'division_ratio'],
                               graph_reg_plots=[]):

    # if we don't choose a specific set of factors for each target, then we will use the rest of the variables in the same cell to make the prediction
    # (in this case there will be 5 variable factors).
    if factors == None:
        factors = dict(zip([str(target) for target in np.array(targets)], [[var for var in all_variable_names if var != tar] for tar in targets]))
        print('factors in general:\n', factors)

    # for each target we use factors to find the distribution of the targets conditional on
    # the values of the factors we are using. We save these distributions for each target in the means and variances dictionaries.
    # Then we take the mean of this distribution as the "prediction" of the model and compare that to the data value of the target for
    # how_many_samples number of times. A metric of interest is the pearson coefficient between the prediction and the actual value.
    means = dict()
    variances = dict()
    data = dict()
    predictions = dict()
    for target in np.array(targets):
        means.update({target: np.array([])})
        variances.update({target: np.array([])})
        data.update({target: np.array([])})
        predictions.update({target: np.array([])})
        print('target:\n', target)
        print('factors:\n', factors[str(target)])
        for num_of_samples in range(how_many_samples):
            # new_mean, new_variance = conditional_joint_prob_data(x1_mean=daug[target], x2_mean, ccov_11, ccov_12, ccov_21, ccov_22, a)

            new_mean, new_variance = conditional_joint_probability_multinormal(x1=df[target],
                                                                               x2=df[factors[str(target)]],
                                                                               a=df[factors[str(target)]].iloc[num_of_samples])

            means[target] = np.append(means[target], new_mean)
            variances[target] = np.append(variances[target], new_variance)
            data[target] = np.append(data[target], df[target].iloc[num_of_samples])
            predictions[target] = np.append(predictions[target], new_mean)

        if target in graph_reg_plots:
            sns.regplot(data[target], predictions[target], line_kws={'color': 'orange'})
            plt.title(target)
            plt.xlabel('data')
            plt.ylabel('prediction')
            plt.show()
            plt.close()
        print('coefficient', round(stats.pearsonr(data[target], predictions[target])[0], 3))
        print('r squared', round(sklearn.metrics.r2_score(data[target], predictions[target]), 3))
        print('-----')

    # This is to see if the prediction of the growth rate and the prediction of the generationtime give the same correlation to the data fold growth
    # as the predicted fold growth
    print('another')
    print('coefficient',
          round(stats.pearsonr(data['fold_growth'], np.array(predictions['generationtime']) * np.array(predictions['growth_rate']))[0], 3))
    print('r squared', sklearn.metrics.r2_score(data['fold_growth'], np.array(predictions['generationtime']) * np.array(predictions['growth_rate'])))
    print('-----')

    return data, predictions, means, variances


def new_conditional(mom, daug, targets=['generationtime', 'length_birth', 'length_final', 'growth_rate', 'fold_growth', 'division_ratio'],
                               factors=None,  how_many_samples=1000,
                               all_variable_names=['generationtime', 'length_birth', 'length_final', 'growth_rate', 'fold_growth', 'division_ratio'],
                               graph=False):

    # if we don't choose a specific set of mother variables to use to predict the daughter variable,
    # then we use all 6 mother variables for each daughter variable
    # if factors == None:
    #     factors = dict(zip([str(target) for target in np.array(targets)], [all_variable_names+['same_length_birth'] for tar in targets]))#
    #     print('factors:\n', factors)

    # for each daughter variable (target) we use mother variables (factors) to find the distribution of the daughter variables conditional on
    # the values of the mother variables we are using. We save these distributions for each daughter target in the means and variances dictionaries.
    # Then we take the mean of this distribution as the "prediction" of the model and compare that to the data value of the daughter variable for
    # how_many_samples number of times. A metric of interest is the pearson coefficient between the prediction and the actual value.
    means = pd.DataFrame(columns=targets)
    variances = []
    data = pd.DataFrame(columns=targets)

    # print(mom[targets])
    # print(daug['length_birth'].rename('same_length_birth'))
    # exit()

    # new_facts = mom[targets].copy()
    # new_facts['same_length_birth'] = daug['length_birth'].rename('same_length_birth')
    # new_facts['same_fold_growth'] = daug['fold_growth'].rename('same_fold_growth')

    # print(new_facts[all_variable_names + ['same_length_birth']])
    # print(daug['length_birth'])
    # print(new_facts[all_variable_names+['same_length_birth']].iloc[990])
    # print(daug['length_birth'].iloc[990])
    # exit()

    for num_of_samples in range(how_many_samples):
        # new_mean, new_variance = conditional_joint_probability_multinormal(x1=daug[targets],
        #                                                                    x2=new_facts[all_variable_names+['same_length_birth']+['same_fold_growth']],
        #                                                                    a=new_facts[all_variable_names+['same_length_birth']+['same_fold_growth']].iloc[num_of_samples])

        new_mean, new_variance = conditional_joint_probability_multinormal(x1=daug[targets],
                                                                           x2=mom[factors],
                                                                           a=mom[factors].iloc[num_of_samples])

        # print(new_mean)
        # print(new_variance)

        means = means.append(pd.Series(new_mean, index=targets), ignore_index=True)
        variances.append(new_variance)
        # variances = variances.append(pd.Series(new_variance, index=targets), ignore_index=True)
        data = data.append(daug[targets].iloc[num_of_samples], ignore_index=True)

    # print(means)
    # print(means.isnull().sum())

    for target in targets:
        print('target:\n', target)
        print('factors:\n', factors)
        # print('coefficient', round(stats.pearsonr(data[target], means[target])[0], 3))
        # print('r squared', round(sklearn.metrics.r2_score(data[target], means[target]), 3))
        print(round(sklearn.metrics.r2_score(data[target], means[target]), 3), round(stats.pearsonr(data[target], means[target])[0], 3))
        print('-----')
        if graph:
            sns.regplot(data[target], means[target], line_kws={'color': 'orange'})
            plt.xlabel('data')
            plt.ylabel('prediction')
            plt.title(target)
            plt.show()
            plt.close()

    return data, means, variances


def same_cell_prediction(df, targets=['generationtime', 'length_birth', 'length_final', 'growth_rate', 'fold_growth', 'division_ratio'],
                               factors=None,  how_many_samples=1000,
                               all_variable_names=['generationtime', 'length_birth', 'length_final', 'growth_rate', 'fold_growth', 'division_ratio']):

    # this has to be done because the final size and the division ration are not determined at the beginning and therefore cannot be a factor in the
    # decision making of the generationtime, growth_rate, length_birth, or fold_growth... I think
    all_variable_names.remove('division_ratio')
    all_variable_names.remove('length_final')

    # if we don't choose a specific set of factors for each target, then we will use the rest of the variables in the same cell to make the prediction
    # (in this case there will be 5 variable factors).
    if factors == None:
        factors = dict(zip([str(target) for target in np.array(targets)], [[var for var in all_variable_names if var != tar] for tar in targets]))
        print('factors in general:\n', factors)

    # for each target we use factors to find the distribution of the targets conditional on
    # the values of the factors we are using. We save these distributions for each target in the means and variances dictionaries.
    # Then we take the mean of this distribution as the "prediction" of the model and compare that to the data value of the target for
    # how_many_samples number of times. A metric of interest is the pearson coefficient between the prediction and the actual value.
    means = dict()
    variances = dict()
    data = dict()
    predictions = dict()
    for target in np.array(targets):
        means.update({target: np.array([])})
        variances.update({target: np.array([])})
        data.update({target: np.array([])})
        predictions.update({target: np.array([])})
        print('target:\n', target)
        print('factors:\n', factors[str(target)])
        for num_of_samples in range(how_many_samples):
            new_mean, new_variance = conditional_joint_probability_multinormal(x1=df[target],
                                                                               x2=df[factors[str(target)]],
                                                                               a=df[factors[str(target)]].iloc[num_of_samples])

            means[target] = np.append(means[target], new_mean)
            variances[target] = np.append(variances[target], new_variance)
            data[target] = np.append(data[target], df[target].iloc[num_of_samples])
            predictions[target] = np.append(predictions[target], new_mean)

        sns.regplot(data[target], predictions[target], line_kws={'color': 'orange'})
        plt.title(target)
        plt.xlabel('data')
        plt.ylabel('prediction')
        plt.show()
        plt.close()
        print('coefficient', round(stats.pearsonr(data[target], predictions[target])[0], 3))
        print('r squared', round(sklearn.metrics.r2_score(data[target], predictions[target]), 3))
        print('-----')

    # This is to see if the prediction of the growth rate and the prediction of the generationtime give the same correlation to the data fold growth
    # as the predicted fold growth
    print('another')
    print('coefficient',
          round(stats.pearsonr(data['fold_growth'], np.array(predictions['generationtime']) * np.array(predictions['growth_rate']))[0], 3))
    print('r squared', sklearn.metrics.r2_score(data['fold_growth'], np.array(predictions['generationtime']) * np.array(predictions['growth_rate'])))
    print('-----')

    return data, predictions, means, variances


def mother_to_predict_daughter(mom, daug, targets=['generationtime', 'length_birth', 'length_final', 'growth_rate', 'fold_growth', 'division_ratio'],
                               factors=None,  how_many_samples=1000,
                               all_variable_names=['generationtime', 'length_birth', 'length_final', 'growth_rate', 'fold_growth', 'division_ratio'],
                               graph_reg_plots=[]):

    # if we don't choose a specific set of mother variables to use to predict the daughter variable,
    # then we use all 6 mother variables for each daughter variable
    if factors == None:
        factors = dict(zip([str(target) for target in np.array(targets)], [all_variable_names for tar in targets]))
        print('factors:\n', factors)

    # for each daughter variable (target) we use mother variables (factors) to find the distribution of the daughter variables conditional on
    # the values of the mother variables we are using. We save these distributions for each daughter target in the means and variances dictionaries.
    # Then we take the mean of this distribution as the "prediction" of the model and compare that to the data value of the daughter variable for
    # how_many_samples number of times. A metric of interest is the pearson coefficient between the prediction and the actual value.
    means = dict()
    variances = dict()
    data = dict()
    predictions = dict()
    for target in np.array(targets):
        means.update({target: np.array([])})
        variances.update({target: np.array([])})
        data.update({target: np.array([])})
        predictions.update({target: np.array([])})
        print('target:\n', target)
        print('factors:\n', factors[str(target)])
        for num_of_samples in range(how_many_samples):
            new_mean, new_variance = conditional_joint_probability_multinormal(x1=daug[target],
                                                                               x2=mom[factors[str(target)]],
                                                                               a=mom[factors[str(target)]].iloc[num_of_samples])

            means[target] = np.append(means[target], new_mean)
            variances[target] = np.append(variances[target], new_variance)
            data[target] = np.append(data[target], daug[target].iloc[num_of_samples])
            predictions[target] = np.append(predictions[target], new_mean)

        if target in graph_reg_plots:
            sns.regplot(data[target], predictions[target], line_kws={'color': 'orange'})
            plt.xlabel('data')
            plt.ylabel('prediction')
            plt.title(target)
            plt.show()
            plt.close()
        print('coefficient', round(stats.pearsonr(data[target], predictions[target])[0], 3))
        print('r squared', round(sklearn.metrics.r2_score(data[target], predictions[target]), 3))
        print('-----')

    # This is to see if the prediction of the growth rate and the prediction of the generationtime give the same correlation to the data fold growth
    # as the predicted fold growth
    # print('another')
    # print('coefficient', round(stats.pearsonr(data['fold_growth'], np.array(predictions['generationtime']) * np.array(predictions['growth_rate']))[0], 3))
    # print('r squared', sklearn.metrics.r2_score(data['fold_growth'], np.array(predictions['generationtime']) * np.array(predictions['growth_rate'])))
    # print('-----')

    return data, predictions, means, variances


def mother_daughter_and_same_with_real_values(mom, daug, targets=['generationtime', 'length_birth', 'length_final', 'growth_rate', 'fold_growth', 'division_ratio'],
                               md_factors=None,  same_factors = None, how_many_samples=1000,
                               all_variable_names=['generationtime', 'length_birth', 'length_final', 'growth_rate', 'fold_growth', 'division_ratio'],
                             graph_reg_plots=[]):

    # if we don't choose a specific set of mother variables to use to predict the daughter variable,
    # then we use all 6 mother variables for each daughter variable
    if md_factors == None:
        md_factors = dict(zip([str(target) for target in np.array(targets)], [all_variable_names for tar in targets]))

    # this has to be done because the final size and the division ration are not determined at the beginning and therefore cannot be a factor in the
    # decision making of the generationtime, growth_rate, length_birth, or fold_growth... I think
    all_variable_names1 = all_variable_names.copy()
    all_variable_names1.remove('division_ratio')
    all_variable_names1.remove('length_final')
    all_variable_names1.remove('fold_growth')

    # if we don't choose a specific set of factors for each target, then we will use the rest of the variables in the same cell to make the prediction
    # (in this case there will be 5 variable factors).
    if same_factors == None:
        same_factors = dict(zip([str(target) for target in np.array(targets)], [[var for var in all_variable_names1 if var != tar] for tar in targets]))

    # for each daughter variable (target) we use mother variables (factors) to find the distribution of the daughter variables conditional on
    # the values of the mother variables we are using. We save these distributions for each daughter target in the means and variances dictionaries.
    # Then we take the mean of this distribution as the "prediction" of the model and compare that to the data value of the daughter variable for
    # how_many_samples number of times. A metric of interest is the pearson coefficient between the prediction and the actual value.
    means = dict()
    variances = dict()
    data = dict()
    predictions = dict()
    for target in np.array(targets):
        means.update({target: np.array([])})
        variances.update({target: np.array([])})
        data.update({target: np.array([])})
        predictions.update({target: np.array([])})
        print('target:\n', target)
        print('md_factors:\n', md_factors[str(target)])
        print('same_factors:\n', same_factors[str(target)])
        md_col_names = ['inter_' + col for col in md_factors[str(target)]]
        same_col_names = ['same_'+col for col in same_factors[str(target)]]

        # Now instead of ONLY using the mother or ONLY using the same-cell data we will use a mixture of both.
        mixed_gaussian = pd.concat([mom[md_factors[str(target)]].rename(columns=dict(zip(md_factors[str(target)], md_col_names))),
                                    daug[same_factors[str(target)]].rename(columns=dict(zip(same_factors[str(target)], same_col_names)))], axis=1)
        for num_of_samples in range(how_many_samples):

            new_mean, new_variance = conditional_joint_probability_multinormal(x1=daug[target],
                                                                               x2=mixed_gaussian[md_col_names+same_col_names],
                                                                               a=mixed_gaussian.iloc[num_of_samples])

            means[target] = np.append(means[target], new_mean)
            variances[target] = np.append(variances[target], new_variance)
            data[target] = np.append(data[target], daug[target].iloc[num_of_samples])
            predictions[target] = np.append(predictions[target], new_mean)

        if target in graph_reg_plots:
            sns.regplot(data[target], predictions[target], line_kws={'color': 'orange'})
            plt.xlabel('data')
            plt.ylabel('prediction')
            plt.title(target)
            plt.show()
            plt.close()
        # print('Average coefficient of variation {:.3}'.format(np.mean(np.array(variances[target]) / np.abs(np.array(means[target])))))
        print('coefficient', round(stats.pearsonr(data[target], predictions[target])[0], 3))
        print('r squared', round(sklearn.metrics.r2_score(data[target], predictions[target]), 3))
        print('-----')

    # This is to see if the prediction of the growth rate and the prediction of the generationtime give the same correlation to the data fold growth
    # as the predicted fold growth
    if 'another' in graph_reg_plots:
        print('another')
        print('coefficient',
              round(stats.pearsonr(predictions['fold_growth'], np.array(predictions['generationtime']) * np.array(predictions['growth_rate']))[0], 3))
        print('r squared', sklearn.metrics.r2_score(predictions['fold_growth'], np.array(predictions['generationtime']) * np.array(predictions['growth_rate'])))
        sns.regplot(predictions['fold_growth'], np.array(predictions['generationtime']) * np.array(predictions['growth_rate']), line_kws={'color': 'orange'})
        plt.xlabel('prediction fold growth')
        plt.ylabel('prediction generationtime * growth_rate')
        plt.title('another')
        plt.show()
        plt.close()
        print('-----')

    return data, predictions, means, variances


def mother_daughter_and_same_with_predicted_values(mom, daug, targets=['generationtime', 'length_birth', 'length_final', 'growth_rate', 'fold_growth', 'division_ratio'],
                               md_factors=None,  same_factors = None, how_many_samples_same=1000, how_many_samples_md=1000,
                               all_variable_names=['generationtime', 'length_birth', 'length_final', 'growth_rate', 'fold_growth', 'division_ratio'],
                             graph_reg_plots=[]):

    # if we don't choose a specific set of mother variables to use to predict the daughter variable,
    # then we use all 6 mother variables for each daughter variable
    if md_factors == None:
        md_factors = dict(zip([str(target) for target in np.array(targets)], [all_variable_names for tar in targets]))

    # this has to be done because the final size and the division ration are not determined at the beginning and therefore cannot be a factor in the
    # decision making of the generationtime, growth_rate, length_birth, or fold_growth... I think
    all_variable_names1 = all_variable_names.copy()
    all_variable_names1.remove('division_ratio')
    all_variable_names1.remove('length_final')
    all_variable_names1.remove('fold_growth')

    # if we don't choose a specific set of factors for each target, then we will use the rest of the variables in the same cell to make the prediction
    # (in this case there will be 5 variable factors).
    if same_factors == None:
        same_factors = dict(zip([str(target) for target in np.array(targets)], [[var for var in all_variable_names1 if var != tar] for tar in targets]))

    # predict the daughter and use this prediction as the condition
    same_factor_data, same_factor_predictions, same_factor_means, same_factor_variances = mother_to_predict_daughter(mom, daug, targets=same_factors,
                                   factors=md_factors, how_many_samples=how_many_samples_same, all_variable_names=all_variable_names)
    same_factor_predictions = pd.DataFrame(same_factor_predictions)

    # for each daughter variable (target) we use mother variables (factors) to find the distribution of the daughter variables conditional on
    # the values of the mother variables we are using. We save these distributions for each daughter target in the means and variances dictionaries.
    # Then we take the mean of this distribution as the "prediction" of the model and compare that to the data value of the daughter variable for
    # how_many_samples number of times. A metric of interest is the pearson coefficient between the prediction and the actual value.
    means = dict()
    variances = dict()
    data = dict()
    predictions = dict()

    for target in np.array(targets):
        means.update({target: np.array([])})
        variances.update({target: np.array([])})
        data.update({target: np.array([])})
        predictions.update({target: np.array([])})
        print('target:\n', target)
        print('md_factors:\n', md_factors[str(target)])
        print('same_factors:\n', same_factors[str(target)])
        md_col_names = ['inter_' + col for col in md_factors[str(target)]]
        same_col_names = ['same_'+col for col in same_factors[str(target)]]

        # Now instead of ONLY using the mother or ONLY using the same-cell data we will use a mixture of both.
        mixed_gaussian = pd.concat([mom[md_factors[str(target)]].rename(columns=dict(zip(md_factors[str(target)], md_col_names))),
                                    same_factor_predictions[same_factors[str(target)]].rename(columns=dict(zip(same_factors[str(target)], same_col_names)))], axis=1) # daug[same_factors[str(target)]].rename(columns=dict(zip(same_factors[str(target)], same_col_names)))

        for num_of_samples_md in range(how_many_samples_md):

            new_mean, new_variance = conditional_joint_probability_multinormal(x1=daug[target],
                                                                               x2=mixed_gaussian[md_col_names+same_col_names],
                                                                               a=mixed_gaussian.iloc[num_of_samples_md])

            means[target] = np.append(means[target], new_mean)
            variances[target] = np.append(variances[target], new_variance)
            data[target] = np.append(data[target], daug[target].iloc[num_of_samples_md])
            predictions[target] = np.append(predictions[target], new_mean)

        if target in graph_reg_plots:
            sns.regplot(data[target], predictions[target], line_kws={'color': 'orange'})
            plt.xlabel('data')
            plt.ylabel('prediction')
            plt.title(target)
            plt.show()
            plt.close()
        # print('Average coefficient of variation {:.3}'.format(np.mean(np.array(variances[target]) / np.abs(np.array(means[target])))))
        print('coefficient', round(stats.pearsonr(data[target], predictions[target])[0], 3))
        print('r squared', round(sklearn.metrics.r2_score(data[target], predictions[target]), 3))
        print('-----')

    # This is to see if the prediction of the growth rate and the prediction of the generationtime give the same correlation to the data fold growth
    # as the predicted fold growth
    if 'another' in graph_reg_plots:
        print('another')
        print('coefficient',
              round(stats.pearsonr(predictions['fold_growth'], np.array(predictions['generationtime']) * np.array(predictions['growth_rate']))[0], 3))
        print('r squared', sklearn.metrics.r2_score(predictions['fold_growth'], np.array(predictions['generationtime']) * np.array(predictions['growth_rate'])))
        sns.regplot(predictions['fold_growth'], np.array(predictions['generationtime']) * np.array(predictions['growth_rate']), line_kws={'color': 'orange'})
        plt.xlabel('prediction fold growth')
        plt.ylabel('prediction generationtime * growth_rate')
        plt.title('another')
        plt.show()
        plt.close()
        print('-----')

    return data, predictions, means, variances


def mean_and_var_of_normal_product_dist(mean_1, mean_2, std_1, std_2, p_coeff):
    # generationtime is 1 and growth rate is 2
    product_mean = mean_1*mean_2+p_coeff*std_1*std_2
    product_var = mean_1**2*std_2**2 + mean_2**2*std_1**2+std_1**2*std_2**2+2*p_coeff*std_1*std_2*mean_1*mean_2+p_coeff**2*std_1**2*std_2**2

    return product_mean, product_var


def plot_it(coeff, corr, high, low, relations, **kwargs):
    ax = plt.gca()
    ax.axhline(y=0, color='black', ls='--')
    yerr = np.array([np.abs(np.array(low - coeff)), np.abs(np.array(high - coeff))])
    ax.errorbar(x=[str(ind) for ind in range(1, len(relations)+1)], y=coeff, yerr=yerr, marker='.', capsize=1.7, barsabove=True)
    relations = relations.reset_index(drop=True)
    # making sure we are plotting the points in the corresponding order
    for separation in range(len(relations)):
        if relations.loc[separation] != separation:
            print('Error '+str(separation)+' separation is not in '+str(separation)+' index')


def pearson_corr_with_confidence_intervals(vec1, vec2):
    # Get the Pearson Correlation
    r = stats.pearsonr(vec1, vec2)[0]
    # Get the confidence interval:
    # Use the Fisher transformation to get z
    z = np.arctanh(r)
    # sigma value is the standard error
    sigma = (1/((len(vec1)-3)**0.5))
    # get a 95% confidence interval
    cint = z + np.array([-1, 1]) * sigma * stats.norm.ppf((1 + 0.95) / 2)
    # get the interval
    low = np.tanh(cint)[0]
    high = np.tanh(cint)[1]

    return r, low, high


def inter_generational_correlations(mom, daug, Population):
    # collect all possible permutations of two variables which is the same as all possible correlations one can do with these variables
    corrs = [p for p in itertools.product(Population._variable_names, repeat=2)]
    corrs_sym = [p for p in itertools.product(Population._variable_symbols.loc['without subscript'], repeat=2)]
    y_labels = [r'$\rho(${}$_m, ${}$_d)$'.format(corr[0], corr[1]) for corr in corrs_sym]

    all_corrs = pd.DataFrame(columns=['correlation coefficient', 'low', 'high', 'corr', 'relation'])

    for separation in range(len(mom)):
        for corr, label in zip(corrs, y_labels):
            r, low, high = pearson_corr_with_confidence_intervals(mom[separation][corr[0]],
                                                                  daug[separation][corr[1]])
            all_corrs = all_corrs.append(dict(zip(['correlation coefficient', 'low', 'high', 'corr', 'relation'],
                                                  [r, low, high, label, separation])), ignore_index=True)

    # sanity check to make sure the confidence intervals make sense
    for index in range(len(all_corrs)):
        if all_corrs['correlation coefficient'].iloc[index] < all_corrs['low'].iloc[index]:
            print('index', index)
            print(all_corrs['correlation coefficient'].iloc[index], 'is lower than', all_corrs['low'].iloc[index])
            print(all_corrs.iloc[index])
        if all_corrs['correlation coefficient'].iloc[index] > all_corrs['high'].iloc[index]:
            print('index', index)
            print(all_corrs['correlation coefficient'].iloc[index], 'is greater than', all_corrs['high'].iloc[index])
            print(all_corrs.iloc[index])

    # set the background style of the plots
    sns.set_style('whitegrid')
    # plot the correlations
    g = sns.FacetGrid(all_corrs, col="corr", col_wrap=6)
    g.map(plot_it, 'correlation coefficient', 'corr', 'high', 'low', 'relation')
    # make sure we have comprehensive y labels where we need to and no x_labels anywhere except for the tick marks
    for row, ylabel in zip(range(0, 37, 6), Population._variable_symbols.loc['without subscript']):
        g.axes[row].set_ylabel(ylabel + r'$_m$')
    for ind, xlabel in zip(range(30, 37), Population._variable_symbols.loc['without subscript']):
        g.axes[ind].set_xlabel(xlabel + r'$_d$')
    for col, label in zip(range(36), y_labels):
        g.axes[col].set_title('')
        g.axes[col].annotate(label, xy=(.1, .9), xycoords=g.axes[col].transAxes, weight='extra bold', size='xx-large')
        # g.axes[col].set_xlabel('')
    # save the figure we just made
    plt.show()
    # plt.savefig('Inter-Generational Correlations, new and more', dpi=300)


def someplacetostoreoldthings():
    values = []
    samples = np.random.choice(mom['generationtime'] * mom['growth_rate'], 3000, replace=True)

    # x1, x2 are the arrays that carry the distributions
    # x is the x-value of the density function to give the y-value
    # N is the number of terms we want to keep
    x1 = mom['generationtime']
    x2 = mom['growth_rate']
    x1 = np.random.normal(x1.mean(), x1.std(), 1000)
    x2 = np.random.normal(x2.mean(), x2.std(), 1000)
    x1 = np.random.normal(3, 2.1, 1000)
    x2 = np.random.normal(1, 1, 1000)
    samples = x1 * x2
    exes = np.linspace(min(samples), max(samples), 1000)

    # we will need this for the distribution
    mean_1, std_1 = np.mean(x1), np.std(x1)
    mean_2, std_2 = np.mean(x2), np.std(x2)
    p_corr = stats.pearsonr(x1, x2)[0]

    print(mean_1, mean_2, std_1, std_2, p_corr)

    for sample in exes:
        value = checking_product_distribution(mean_1, std_1, mean_2, std_2, p_corr, x=sample, N=16)
        values.append(value)

    print(len(values), len(samples))
    print(np.mean(samples), np.mean(values))
    print(np.std(samples), np.std(values))

    ax = sns.distplot(samples, kde=True, kde_kws={"color": "g", "lw": 3, "label": "Samples from data"},
                      hist_kws={"histtype": "step", "linewidth": 3, "alpha": .5, "color": "g"}, norm_hist=True)
    plt.plot(exes, values)
    # sns.distplot(values, kde=True, ax=ax, kde_kws={"color": "blue", "lw": 3, "label": "Product Distribution"}, hist_kws={"histtype": "step", "linewidth": 3, "alpha": .5, "color": "blue"}, norm_hist=True)
    plt.show()
    plt.close()
    #
    # ax = sns.distplot(samples, kde=True, kde_kws={"color": "g", "lw": 3, "label": "Samples from data"},
    #                   hist_kws={"histtype": "step", "linewidth": 3, "alpha": .5, "color": "g"})
    # plt.show()
    # plt.close()
    exit()


def synthetic_mom_and_daughter_dataframes(mom, daug, number_of_traces=None, number_of_generations=6):
    # Since this is based on the data we must input the mother and daughter dataframes from


    all_vars = ['generationtime', 'length_birth', 'length_final', 'growth_rate', 'fold_growth', 'division_ratio']

    data, predictions, means, variances = mother_daughter_and_same(mom, daug, targets=all_vars,
                                                                   md_factors={
                                                                       'generationtime': all_vars, 'length_birth': [var for var in all_vars if
                                                                                                                    var not in ['length_final',
                                                                                                                                'division_ratio']],
                                                                       'length_final': all_vars, 'growth_rate': all_vars, 'fold_growth': all_vars,
                                                                       'division_ratio': all_vars
                                                                   }, same_factors={
            'generationtime': ['length_birth', 'growth_rate'], 'length_birth': ['generationtime', 'growth_rate'], 'length_final': ['length_birth'],
            'growth_rate': ['length_birth'], 'fold_growth': ['length_birth'], 'division_ratio': ['length_birth', 'fold_growth']
        }, how_many_samples=1000, all_variable_names=all_vars, graph_reg_plots=[])


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


    # generationtime and growth rate are log normal
    mom['generationtime'] = np.log(mom['generationtime'])
    mom['length_birth'] = np.log(mom['length_birth'])
    mom['length_final'] = np.log(mom['length_final'])
    mom['growth_rate'] = np.log(mom['growth_rate'])
    mom['fold_growth'] = np.log(mom['fold_growth'])
    mom['division_ratio'] = np.log(mom['division_ratio'])

    daug['generationtime'] = np.log(daug['generationtime'])
    daug['length_birth'] = np.log(daug['length_birth'])
    daug['length_final'] = np.log(daug['length_final'])
    daug['growth_rate'] = np.log(daug['growth_rate'])
    daug['fold_growth'] = np.log(daug['fold_growth'])
    daug['division_ratio'] = np.log(daug['division_ratio'])

    all_vars = ['generationtime', 'length_birth', 'length_final', 'growth_rate', 'fold_growth', 'division_ratio']

    mom_global = subtract_global_averages(df=mom, columns_names=Population._variable_names)
    mom_trap = subtract_trap_averages(df=mom, columns_names=Population._variable_names, log_vars=all_vars)
    mom_traj = subtract_traj_averages(df=mom, columns_names=Population._variable_names, log_vars=all_vars)

    daug_global = subtract_global_averages(df=daug, columns_names=Population._variable_names)
    daug_trap = subtract_trap_averages(df=daug, columns_names=Population._variable_names, log_vars=all_vars)
    daug_traj = subtract_traj_averages(df=daug, columns_names=Population._variable_names, log_vars=all_vars)




    golden_number = 3000
    mom_trap = mom_trap.copy().iloc[:golden_number]
    daug_trap = daug_trap.copy().iloc[:golden_number]

    # data, predictions, means, variances = mother_to_predict_daughter(mom=mom_trap, daug=daug_trap, how_many_samples=1000, targets=['length_birth'],
    #                                                            factors={'length_birth': ['length_birth', 'fold_growth']})
    # exit()

    # data, predictions, means, variances = same_cell_prediction(df=mom, how_many_samples=1000, targets=['fold_growth'],
    #                                                            factors={'fold_growth': ['length_birth']})
    #
    # exit()

    # data, predictions, means, variances = same_cell_prediction(df=mom_trap, how_many_samples=1000)

    # data, predictions, means, variances = mother_to_predict_daughter(mom=mom_trap, daug=daug_trap, how_many_samples=1000)
    # exit()

    X_train, X_test, y_train, y_test = train_test_split(mom_global, daug_global, test_size=0.33, random_state=42, shuffle=True)
    data, means, variances = new_conditional(
        X_train, y_train, targets=all_vars, factors=all_vars, how_many_samples=1000, all_variable_names=all_vars, graph=False)

    X_train, X_test, y_train, y_test = train_test_split(mom_trap, daug_trap, test_size=0.33, random_state=42, shuffle=True)
    data, means, variances = new_conditional(
        X_train, y_train, targets=all_vars, factors=all_vars, how_many_samples=1000, all_variable_names=all_vars, graph=False)

    X_train, X_test, y_train, y_test = train_test_split(mom_traj, daug_traj, test_size=0.33, random_state=42, shuffle=True)
    data, means, variances = new_conditional(
        X_train, y_train, targets=all_vars, factors=all_vars, how_many_samples=1000, all_variable_names=all_vars, graph=False)

    exit()

    second_X_train = X_train.copy()
    second_X_train['division_ratio'] = 1 - X_train['division_ratio']
    _, second_means, second_variances = new_conditional(
        second_X_train, y_train, targets=all_vars, factors=all_vars, how_many_samples=500, all_variable_names=all_vars, graph=False)

    # Population.plot_relationship_correlations(df_1=X_train.iloc[:len(means)], df_2=y_train.iloc[:len(means)], df_1_variables=Population._variable_names,
    #                                           df_2_variables=Population._variable_names,
    #                                           x_labels=Population._variable_symbols.loc['_n, normalized lengths'],
    #                                           y_labels=Population._variable_symbols.loc['_{n+1}, normalized lengths'])

    new_m = X_train[Population._variable_names].iloc[:len(means)].reset_index(drop=True)

    # sample form many
    samples_A = pd.DataFrame(columns=Population._variable_names)
    samples_B = pd.DataFrame(columns=Population._variable_names)
    for ind in range(len(means)):
        x = np.random.multivariate_normal(means.iloc[ind], variances[ind])
        y = np.random.multivariate_normal(second_means.iloc[ind], second_variances[ind])
        # print(pd.Series(x.flatten(), index=Population._variable_names))
        samples_A = samples_A.append(pd.Series(x.flatten(), index=Population._variable_names), ignore_index=True)
        samples_B = samples_B.append(pd.Series(y.flatten(), index=Population._variable_names), ignore_index=True)

    # print(samples)

    Population.plot_relationship_correlations(df_1=samples_B[Population._variable_names], df_2=samples_A[Population._variable_names],
                                              df_1_variables=Population._variable_names,
                                              df_2_variables=Population._variable_names,
                                              x_labels=Population._variable_symbols.loc['_B, normalized lengths'],
                                              y_labels=Population._variable_symbols.loc['_A, normalized lengths'])

    # Population.plot_relationship_correlations(df_1=samples_B[Population._variable_names], df_2=samples_A[Population._variable_names], df_1_variables=Population._variable_names,
    #                                           df_2_variables=Population._variable_names,
    #                                           x_labels=Population._variable_symbols.loc['_n, normalized lengths'],
    #                                           y_labels=Population._variable_symbols.loc['_{n+1}, normalized lengths'])

    # Population.plot_same_cell_correlations(df=samples, variables=samples.columns,
    #                                        labels=Population._variable_symbols.loc['without subscript, normalized lengths'])

    exit()

    Population.plot_relationship_correlations(df_1=new_m, df_2=means[Population._variable_names], df_1_variables=Population._variable_names,
                                              df_2_variables=Population._variable_names,
                                              x_labels=Population._variable_symbols.loc['_n, normalized lengths'],
                                              y_labels=Population._variable_symbols.loc['_{n+1}, normalized lengths'])

    Population.plot_same_cell_correlations(df=data, variables=data.columns,
                                           labels=Population._variable_symbols.loc['without subscript, normalized lengths'])

    Population.plot_same_cell_correlations(df=means, variables=means.columns,
                                           labels=Population._variable_symbols.loc['without subscript, normalized lengths'])

    exit()

    for l in range(len(means['generationtime'])):

        product_mean, product_var = mean_and_var_of_normal_product_dist(
            mean_1=means['generationtime'][l], mean_2=means['growth_rate'][l], std_1=np.sqrt(variances['generationtime'][l]),
            std_2=np.sqrt(variances['growth_rate'][l]), p_coeff=stats.pearsonr(mom['generationtime'], mom['growth_rate'])[0])

        product = np.random.normal(loc=product_mean, scale=np.sqrt(product_var), size=200)
        fold_growth = np.random.normal(loc=means['fold_growth'][l], scale=np.sqrt(variances['fold_growth'][l]), size=200)

        sns.kdeplot(product, shade=True, color='r')
        sns.kdeplot(fold_growth, shade=True, color='b')
        plt.show()
        plt.close()
    print(means)

    exit()

    data, predictions, means, variances = mother_daughter_and_same_with_real_values(mom, daug, targets=all_vars,
                                                                                    md_factors={
                                                                                        'generationtime': all_vars,
                                                                                        'length_birth': [var for var in all_vars if
                                                                                                         var not in ['length_final',
                                                                                                                     'division_ratio']],
                                                                                        'length_final': all_vars, 'growth_rate': all_vars,
                                                                                        'fold_growth': all_vars, 'division_ratio': all_vars
                                                                                    }, same_factors={
            'generationtime': ['length_birth'], 'length_birth': [], 'length_final': ['length_birth'],
            'growth_rate': ['length_birth'], 'fold_growth': ['length_birth'], 'division_ratio': ['length_birth']
        }, how_many_samples=500, all_variable_names=all_vars, graph_reg_plots=[])

    exit()

    data, predictions, means, variances = mother_daughter_and_same_with_real_values(mom, daug, targets=all_vars,
                             md_factors={
                                 'generationtime': all_vars, 'length_birth': [var for var in all_vars if var not in ['length_final', 'division_ratio']],
                                 'length_final': all_vars, 'growth_rate': all_vars, 'fold_growth': all_vars, 'division_ratio': all_vars
                             }, same_factors={
            'generationtime': ['length_birth'], 'length_birth': [], 'length_final': ['length_birth'],
            'growth_rate': ['length_birth'], 'fold_growth': ['length_birth'], 'division_ratio': ['length_birth']
        }, how_many_samples=500, all_variable_names=all_vars, graph_reg_plots=[])



    exit()

    # conditional_probability_multinormal(daug_trap, mom_trap)

    # First we do the length birth for the daughter using all the others, this has a R^2 of .926, ie. its also perfect.
    data = dict()
    predictions = dict()
    data.update({'length_birth': np.array([])})
    predictions.update({'length_birth': np.array([])})
    for num_of_samples in range(golden_number):
        new_mean, new_variance = conditional_joint_probability_multinormal(x1=daug_trap['length_birth'],
                                                                           x2=mom_trap[Population._variable_names],
                                                                           a=mom_trap[Population._variable_names].iloc[num_of_samples])

        data['length_birth'] = np.append(data['length_birth'], daug_trap['length_birth'].iloc[num_of_samples])
        predictions['length_birth'] = np.append(predictions['length_birth'], new_mean)
        # print(num_of_samples)

    # joint_results.append()
    # print(data['length_birth'], predictions['length_birth'])
    sns.regplot(data['length_birth'], predictions['length_birth'], line_kws={'color': 'orange'})
    plt.show()
    plt.close()
    print('length_birth')
    print('coefficient', round(stats.pearsonr(data['length_birth'], predictions['length_birth'])[0], 3))
    print('r squared', sklearn.metrics.r2_score(data['length_birth'], predictions['length_birth']))
    print('-----')


    mom_trap_and_daug_length = pd.concat([mom_trap.copy(), pd.Series(data['length_birth'].copy(), name='length_birth_daug')], axis=1) # predictions['length_birth']
    daug_without_length = daug_trap.copy().drop(columns=['length_birth'])

    # Now, having the daughter size, we predict the others
    targets = ['generationtime', 'length_final', 'growth_rate', 'fold_growth', 'division_ratio']
    factors = dict(zip([str(target) for target in np.array(targets)],
            [['generationtime', 'length_birth', 'length_final', 'growth_rate', 'fold_growth', 'division_ratio', 'length_birth_daug'] for target in np.array(targets)]))


    # print(mom_trap_and_daug_length)
    # print(daug_without_length)



    joint_results = pd.DataFrame(columns=['daughter_variables', 'pearson correlation coefficient', 'R squared'])
    data = dict()
    predictions = dict()
    for target in np.array(targets):
        data.update({target: np.array([])})
        predictions.update({target: np.array([])})
        for num_of_samples in range(golden_number):
            new_mean, new_variance = conditional_joint_probability_multinormal(x1=daug_without_length[target],
                                                                               x2=mom_trap_and_daug_length[factors[str(target)]],
                                                                               a=mom_trap_and_daug_length[factors[str(target)]].iloc[num_of_samples])

            data[target] = np.append(data[target], daug_without_length[target].iloc[num_of_samples])
            predictions[target] = np.append(predictions[target], new_mean)
            # print(num_of_samples)

        # joint_results.append()
        # print(data[target], predictions[target])
        sns.regplot(data[target], predictions[target], line_kws={'color': 'orange'})
        plt.show()
        plt.close()
        print(target)
        # print('coefficient', round(stats.pearsonr(data[target], predictions[target])[0], 3))
        # print('r squared', sklearn.metrics.r2_score(data[target], predictions[target]))
        # print('-----')

        print('coefficient', round(stats.pearsonr(data[target], predictions[target])[0], 3))
        print('r squared', sklearn.metrics.r2_score(data[target], predictions[target]))
        print('-----')
        # print(target)
        # print('coefficient', round(stats.pearsonr(data[target], predictions[target])[0], 3))
        # print('r squared', sklearn.metrics.r2_score(data[target], predictions[target]))
        # print('-----')

    # print(data['fold_growth'], np.array(predictions['generationtime']) * np.array(predictions['growth_rate']))
    print('another')
    print('coefficient', round(stats.pearsonr(data['fold_growth'], np.array(predictions['generationtime']) * np.array(predictions['growth_rate']))[0], 3))
    print('r squared', sklearn.metrics.r2_score(data['fold_growth'], np.array(predictions['generationtime']) * np.array(predictions['growth_rate'])))
    print('-----')

    exit()

    new_mean, new_variance = conditional_joint_probability_multinormal(mom_trap[['generationtime', 'growth_rate']],
                                                                       mom_trap[['fold_growth', 'division_ratio']],
                                                                       mom_trap[['fold_growth', 'division_ratio']].iloc[0])
    exit()

    count = 0
    for mother, daughter in zip([mom_global, mom_trap, mom_traj], [daug_global, daug_trap, daug_traj]):
        print(count)
        X_train, X_test, y_train, y_test = train_test_split(mom, daughter, test_size=0.33)

        # Context for the model
        with pm.Model() as normal_model:

            data = pd.concat([X_train.rename(columns=dict(zip(X_train.columns, [col + '_m' for col in X_train.columns]))),
                              y_train.rename(columns=dict(zip(y_train.columns, [col + '_d' for col in y_train.columns])))], axis=1)

            tests = pd.concat([X_test.rename(columns=dict(zip(X_test.columns, [col + '_m' for col in X_test.columns]))),
                               y_test.rename(columns=dict(zip(y_test.columns, [col + '_d' for col in y_test.columns])))], axis=1)

            factors = [col + '_m' for col in y_train.columns]
            target = 'generationtime_d'

            # The prior for the data likelihood is a Normal Distribution
            family = pm.glm.families.Normal()

            # Creating the model requires a formula and data (and optionally a family)
            pm.GLM.from_formula(formula=str(target + ' ~ ' + ' + '.join(factors)), data=data, family=family)

            # Perform Markov Chain Monte Carlo sampling letting PyMC3 choose the algorithm
            normal_trace = pm.sample(draws=500, step=None, init='auto', n_init=200000, start=None, trace=None, chain_idx=0, chains=None, cores=1,
                                     tune=500, progressbar=True, model=None, random_seed=None, discard_tuned_samples=True,
                                     compute_convergence_checks=True)

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

        count = count + 1

if __name__ == '__main__':
    main()
