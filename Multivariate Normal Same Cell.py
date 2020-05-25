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
            cond_mean = daughter[variable].mean() + np.dot(dependancy, np.dot(np.linalg.inv(np.array(mother.cov())),
                                                                              (mother[variable].iloc[sample] - mother.mean())))
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
        sns.regplot(from_model, from_data, ax=g, scatter=False, label='{}'.format(round(stats.pearsonr(from_model, from_data)[0], 3)),
                    line_kws={'color': 'orange'})
        # sns.jointplot(from_model, from_data, kind="reg", label='{}'.format(round(stats.pearsonr(from_model, from_data)[0], 3)), line_kws={'color': 'orange'})
        plt.legend()
        plt.title(variable)
        plt.show()
        plt.close()


def conditional_joint_probability_multinormal(x1, x2, a):
    a = a.rename(index=dict(zip(a.index, [ind + '_x2' for ind in a.index])))

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
            # print(cross_covariance_21.shape)
            # print(cross_covariance_12.shape)
            # print(np.array(a - x2.mean()).shape)
            # print(np.linalg.inv(np.array(x2.cov())).shape)
            # print(np.dot(np.linalg.inv(np.array(x2.cov())), np.array(a - x2.mean())))
            new_mean = x1.mean() + np.dot(cross_covariance_12, np.dot(np.linalg.inv(np.array(x2.cov())), np.array(a - x2.mean())))
            new_variance = x1.cov() - np.dot(cross_covariance_12, np.dot(np.linalg.inv(np.array(x2.cov())), cross_covariance_21))
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

    # conditional_probability_multinormal(daug_trap, mom_trap)

    golden_number = 3000
    mom_trap = mom_trap.copy().iloc[:golden_number]
    daug_trap = daug_trap.copy().iloc[:golden_number]

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
    print('length_birth')
    print('coefficient', round(stats.pearsonr(data['length_birth'], predictions['length_birth'])[0], 3))
    print('r squared', sklearn.metrics.r2_score(data['length_birth'], predictions['length_birth']))
    print('-----')

    mom_trap_and_daug_length = pd.concat([mom_trap.copy(), pd.Series(predictions['length_birth'], name='length_birth_daug')], axis=1)
    daug_without_length = daug_trap.copy().drop(columns=['length_birth'])

    # Now, having the daughter size, we predict the others
    targets = ['generationtime', 'length_final', 'growth_rate', 'fold_growth', 'division_ratio']
    factors = dict(zip([str(target) for target in np.array(targets)],
                       [['generationtime', 'length_birth', 'length_final', 'growth_rate', 'fold_growth', 'division_ratio', 'length_birth_daug'] for
                        target in np.array(targets)]))

    # print(mom_trap_and_daug_length)
    # print(daug_without_length)

    joint_results = pd.DataFrame(columns=['daughter_variables', 'pearson correlation coefficient', 'R squared'])
    data = dict()
    predictions = dict()
    for target in np.array(targets):
        data.update({target: np.array([])})
        predictions.update({target: np.array([])})
        for num_of_samples in range(golden_number):
            print(mom_trap_and_daug_length[factors[str(target)]])
            print(predictions['length_birth'])
            exit()
            new_mean, new_variance = conditional_joint_probability_multinormal(x1=daug_without_length[target],
                                                                               x2=mom_trap_and_daug_length[factors[str(target)]],
                                                                               a=mom_trap_and_daug_length[factors[str(target)]].iloc[num_of_samples])

            data[target] = np.append(data[target], daug_without_length[target].iloc[num_of_samples])
            predictions[target] = np.append(predictions[target], new_mean)
            # print(num_of_samples)

        # joint_results.append()
        # print(data[target], predictions[target])
        print(target)
        print('coefficient', round(stats.pearsonr(data[target], predictions[target])[0], 3))
        print('r squared', sklearn.metrics.r2_score(data[target], predictions[target]))
        print('-----')

    # print(data['fold_growth'], np.array(predictions['generationtime']) * np.array(predictions['growth_rate']))
    print('another')
    print('coefficient',
          round(stats.pearsonr(data['fold_growth'], np.array(predictions['generationtime']) * np.array(predictions['growth_rate']))[0], 3))
    print('r squared', sklearn.metrics.r2_score(data['fold_growth'], np.array(predictions['generationtime']) * np.array(predictions['growth_rate'])))
    print('-----')

    exit()

    # targets = ['fold_growth', 'length_birth']
    # targets = Population._variable_names
    targets = ['generationtime', 'growth_rate', 'fold_growth']
    # targets = list(itertools.combinations(Population._variable_names, 2))
    # factors = dict(zip(targets, [['length_birth', 'division_ratio'], ['length_birth', 'division_ratio']]))
    factors = dict(zip([str(target) for target in np.array(targets)], [Population._variable_names for tar in targets]))
    # factors = ['fold_growth', 'length_birth']

    joint_results = pd.DataFrame(columns=['daughter_variables', 'pearson correlation coefficient', 'R squared'])
    data = dict()
    predictions = dict()
    for target in np.array(targets):
        data.update({target: np.array([])})
        predictions.update({target: np.array([])})
        print(data)
        for num_of_samples in range(800):
            new_mean, new_variance = conditional_joint_probability_multinormal(x1=daug_trap[target],
                                                                               x2=mom_trap[factors[str(target)]],
                                                                               a=mom_trap[factors[str(target)]].iloc[num_of_samples])

            data[target] = np.append(data[target], daug_trap[target].iloc[num_of_samples])
            predictions[target] = np.append(predictions[target], new_mean)
            # print(num_of_samples)

        # joint_results.append()
        # print(data[target], predictions[target])
        print(target)
        print('coefficient', round(stats.pearsonr(data[target], predictions[target])[0], 3))
        print('r squared', sklearn.metrics.r2_score(data[target], predictions[target]))
        print('-----')

    print(data['fold_growth'], np.array(predictions['generationtime']) * np.array(predictions['growth_rate']))
    print('another')
    print('coefficient', round(stats.pearsonr(data[target], predictions[target])[0], 3))
    print('r squared', sklearn.metrics.r2_score(data[target], predictions[target]))
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
