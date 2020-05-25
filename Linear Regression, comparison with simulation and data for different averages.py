
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


def print_full_dataframe(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df)


""" input the already centered factor and target dataframe """
def linear_regression_framework(factor_df, target_df, fit_intercept):

    # Make regression matrix in a dataframe representation
    df_matrix = pd.DataFrame(index=target_df.columns, columns=factor_df.columns)

    # arrays where we will keep the information
    scores = dict()
    intercepts = dict()

    # loop over all the regressions we'll be doing
    for t_v in target_df.columns:
        # get the fit
        reg = LinearRegression(fit_intercept=fit_intercept).fit(factor_df, target_df[t_v])
        # save the information
        scores.update({t_v: round(reg.score(factor_df, target_df[t_v]), 3)})
        intercepts.update({t_v: round(reg.intercept_, 3)})

        coefficients = [round(coef, 3) for coef in reg.coef_]
        # put it in the matrix
        df_matrix.loc[t_v] = coefficients

    return df_matrix, scores, intercepts


""" input the already centered factor and target dataframe """
def linear_regression_framework_reg_beta(factor_df, target_df, fit_intercept):

    # Make regression matrix in a dataframe representation
    df_matrix = pd.DataFrame(index=target_df.columns, columns=factor_df.columns)

    # arrays where we will keep the information
    scores = dict()
    intercepts = dict()

    # loop over all the regressions we'll be doing
    for t_v in target_df.columns:
        # get the fit
        reg = LinearRegression(fit_intercept=fit_intercept).fit(factor_df, target_df[t_v])
        # save the information
        scores.update({t_v: round(reg.score(factor_df, target_df[t_v]), 3)})
        intercepts.update({t_v: round(reg.intercept_, 3)})

        # coefficients = [round(coef, 3) for coef in reg.coef_]
        # put it in the matrix
        df_matrix.loc[t_v] = reg.coef_

    return df_matrix, scores, intercepts


'''
P and S are the coefficients of noise terms, y and tau
the number of traces we will manufacture
how many cycles we will record per trace
'''
def fold_growth_and_length_birth_intergenerational_simulation(beta, P, S, number_of_traces, cycles_per_trace, column_names,
                                                     length_mean, length_std, phi_mean, phi_std):

    # the model matrix with chosen growth rate and beta
    model_matrix = np.array([[-beta, -beta], [1, 1]])

    # the random initial variables from which the 50 traces will stem from
    mean_of_length = length_mean
    std_of_length = length_std

    initial_normalized_lengths = np.random.normal(loc=mean_of_length, scale=std_of_length, size=number_of_traces)
    # notice that now we are subtracting the global phi average
    initial_deviation_of_phi_from_mean = np.random.normal(loc=phi_mean,
                                                          scale=phi_std,
                                                          size=number_of_traces)

    # Going to hold all the synthetic trajectories
    all_traces_df = pd.DataFrame(columns=column_names)

    # loop over the num
    for trace_num in range(number_of_traces):

        # using the previous generation to predict the next, previous_generation[0] = delta_phi_n and previous_generation[1] = y_n
        previous_generation = np.array([initial_deviation_of_phi_from_mean[trace_num], initial_normalized_lengths[trace_num]])

        # add it to the all_traces dataframe
        all_traces_df = all_traces_df.append(dict(zip(column_names, previous_generation)), ignore_index=True)


        for cycle_num in range(cycles_per_trace-1):

            # standard normal noise with coefficients named P, S
            noise = np.array([P * np.random.normal(0, 1, 1), S * np.random.normal(0, 1, 1)])

            # print('noise\n', noise)

            # Get the next generation which will become the previous generation so that's what I name it, recursive programming
            previous_generation = np.dot(model_matrix, previous_generation.T) + noise.T

            # this is because, for some reason, the array from the previous action gives the vector inside another, useless array and putting [0]
            # takes it out of this useless array and we get the vector
            previous_generation = previous_generation[0]

            # add it to the all_traces dataframe
            all_traces_df = all_traces_df.append(dict(zip(column_names, previous_generation)), ignore_index=True)

    # mother and daughter dataframes
    daughter_df = pd.DataFrame(columns=all_traces_df.columns)
    mother_df = pd.DataFrame(columns=all_traces_df.columns)

    # add the mother and daughter values to the dataframe, note that the daughter dataframe starts form index 1 while mother from index 0
    for mother_index in range(1, len(all_traces_df)+1):
        if np.mod(mother_index, cycles_per_trace) != 0:
            mother_df = mother_df.append(all_traces_df.loc[mother_index-1])
            daughter_df = daughter_df.append(all_traces_df.loc[mother_index])

    return all_traces_df, mother_df, daughter_df


'''
P and S are the coefficients of noise terms, y and tau
the number of traces we will manufacture
how many cycles we will record per trace
'''
def fold_growth_and_length_birth_intergenerational_simulation_input(model_matrix, P, S, number_of_traces, cycles_per_trace, column_names,
                                                     length_mean, length_std, phi_mean, phi_std):

    # the model matrix with chosen growth rate and beta
    # model_matrix = np.array([[-beta_phi, -beta_length], [1, 1]])

    # the random initial variables from which the 50 traces will stem from
    mean_of_length = length_mean
    std_of_length = length_std

    initial_normalized_lengths = np.random.normal(loc=mean_of_length, scale=std_of_length, size=number_of_traces)
    # notice that now we are subtracting the global phi average
    initial_deviation_of_phi_from_mean = np.random.normal(loc=phi_mean,
                                                          scale=phi_std,
                                                          size=number_of_traces)

    # Going to hold all the synthetic trajectories
    all_traces_df = pd.DataFrame(columns=column_names)

    # loop over the num
    for trace_num in range(number_of_traces):

        # using the previous generation to predict the next, previous_generation[0] = delta_phi_n and previous_generation[1] = y_n
        previous_generation = np.array([initial_deviation_of_phi_from_mean[trace_num], initial_normalized_lengths[trace_num]])

        # add it to the all_traces dataframe
        all_traces_df = all_traces_df.append(dict(zip(column_names, previous_generation)), ignore_index=True)


        for cycle_num in range(cycles_per_trace-1):

            # standard normal noise with coefficients named P, S
            noise = np.array([P * np.random.normal(0, 1, 1), S * np.random.normal(0, 1, 1)])

            # print('noise\n', noise)

            # Get the next generation which will become the previous generation so that's what I name it, recursive programming
            previous_generation = np.dot(model_matrix, previous_generation.T) + noise.T

            # this is because, for some reason, the array from the previous action gives the vector inside another, useless array and putting [0]
            # takes it out of this useless array and we get the vector
            previous_generation = previous_generation[0]

            # add it to the all_traces dataframe
            all_traces_df = all_traces_df.append(dict(zip(column_names, previous_generation)), ignore_index=True)

    # mother and daughter dataframes
    daughter_df = pd.DataFrame(columns=all_traces_df.columns)
    mother_df = pd.DataFrame(columns=all_traces_df.columns)

    # add the mother and daughter values to the dataframe, note that the daughter dataframe starts form index 1 while mother from index 0
    for mother_index in range(1, len(all_traces_df)+1):
        if np.mod(mother_index, cycles_per_trace) != 0:
            mother_df = mother_df.append(all_traces_df.loc[mother_index-1])
            daughter_df = daughter_df.append(all_traces_df.loc[mother_index])

    return all_traces_df, mother_df, daughter_df


def fold_growth_and_length_birth_intergenerational_model_parameters_analytics(df):
    # using a dataframe that contains only the length_birth and the fold_growth
    covar_matrix = df.cov()

    length_var_data = covar_matrix['length_birth']['length_birth']

    phi_var_data = covar_matrix['fold_growth']['fold_growth']

    mixed_var_data = covar_matrix['length_birth']['fold_growth']

    print('var(length)=', round(length_var_data, 3))
    print('var(fold_growth)=', round(phi_var_data, 3))
    print('covariance of length_birth and fold_growth (not normalized)=', round(mixed_var_data, 3))

    optimal_beta = - mixed_var_data / (length_var_data + phi_var_data + 2*mixed_var_data)

    optimal_p_squared = phi_var_data - (mixed_var_data ** 2) / (length_var_data + phi_var_data + 2 * mixed_var_data)

    optimal_s_squared = - 2 * mixed_var_data - phi_var_data

    return optimal_beta, optimal_p_squared, optimal_s_squared


def generationtime_and_length_birth_intergenerational_model_parameters_analytics(df):
    # using a dataframe that contains only the length_birth and the generationtime
    covar_matrix = df.cov()

    print_full_dataframe(covar_matrix)

    length_var_data = covar_matrix['length_birth']['length_birth']

    phi_var_data = covar_matrix['generationtime']['generationtime']

    mixed_var_data = covar_matrix['length_birth']['generationtime']

    print('var(length)=', round(length_var_data, 3))
    print('var(generationtime)=', round(phi_var_data, 3))
    print('covariance of length_birth and generationtime (not normalized)=', round(mixed_var_data, 3))

    optimal_beta = - mixed_var_data / (length_var_data + phi_var_data + 2*mixed_var_data)

    optimal_p_squared = phi_var_data - (mixed_var_data ** 2) / (length_var_data + phi_var_data + 2 * mixed_var_data)

    optimal_s_squared = - 2 * mixed_var_data - phi_var_data

    return optimal_beta, optimal_p_squared, optimal_s_squared


def run_schema():
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


    # change the length births to log length births, including their averages to the log averages
    mom['length_birth'] = np.log(mom['length_birth'])
    daug['length_birth'] = np.log(daug['length_birth'])

    # copy them so we don't mess up the original mother and daughter dfs
    mom_global = mom.copy()
    mom_trap = mom.copy()
    mom_traj = mom.copy()
    daug_global = daug.copy()
    daug_trap = daug.copy()
    daug_traj = daug.copy()
    
    # log the length and center the fold growth and length for all three average-specific m/d dataframes
    mom_global['length_birth'] = mom_global['length_birth'] - np.mean(mom_global['length_birth'])
    daug_global['length_birth'] = daug_global['length_birth'] - np.mean(daug_global['length_birth'])
    mom_global['fold_growth'] = mom_global['fold_growth'] - np.mean(mom_global['fold_growth'])
    daug_global['fold_growth'] = daug_global['fold_growth'] - np.mean(daug_global['fold_growth'])

    mom_trap['length_birth'] = mom_trap['length_birth'] - mom_trap['log_trap_avg_length_birth']
    daug_trap['length_birth'] = daug_trap['length_birth'] - daug_trap['log_trap_avg_length_birth']
    mom_trap['fold_growth'] = mom_trap['fold_growth'] - mom_trap['trap_avg_fold_growth']
    daug_trap['fold_growth'] = daug_trap['fold_growth'] - daug_trap['trap_avg_fold_growth']

    mom_traj['length_birth'] = mom_traj['length_birth'] - mom_traj['log_traj_avg_length_birth']
    daug_traj['length_birth'] = daug_traj['length_birth'] - daug_traj['log_traj_avg_length_birth']
    mom_traj['fold_growth'] = mom_traj['fold_growth'] - mom_traj['traj_avg_fold_growth']
    daug_traj['fold_growth'] = daug_traj['fold_growth'] - daug_traj['traj_avg_fold_growth']

    # run it for the pure combinations
    for m_df, d_df in zip([mom_global, mom_trap, mom_traj],
                          [daug_global, daug_trap, daug_traj]):

        # get the linear regression matrix and score
        # mat, scores, intercepts = linear_regression_framework(factor_df=m_df[['length_birth', 'fold_growth']], target_df=d_df[['length_birth', 'fold_growth']], fit_intercept=False)
        mat, scores, intercepts = linear_regression_framework_reg_beta(factor_df=m_df[['length_birth', 'fold_growth']],
                                                              target_df=d_df[['length_birth', 'fold_growth']], fit_intercept=False)

        print('regression matrix\n', mat)
        print(np.array([mat.loc['fold_growth'], mat.loc['length_birth']]))
        print('scores\n', scores)

        reg_beta = (-mat['length_birth'].loc['fold_growth']-mat['fold_growth'].loc['fold_growth'])/2

        # get the model parameters from centered data around a certain average
        optimal_beta, optimal_p_squared, optimal_s_squared = fold_growth_and_length_birth_intergenerational_model_parameters_analytics(df=m_df[['length_birth', 'fold_growth']])

        print('optimal beta=', round(optimal_beta, 3))
        print('reg beta=', round(reg_beta, 3))
        print('optimal p squared=', round(optimal_p_squared, 3))
        print('optimal s squared=', round(optimal_s_squared, 3))

        # run the simulation
        optimal_all_traces_df, optimal_mother_df, optimal_daughter_df = fold_growth_and_length_birth_intergenerational_simulation(beta=optimal_beta,
            P=np.sqrt(optimal_p_squared), S=np.sqrt(optimal_s_squared), number_of_traces=1, cycles_per_trace=3000, column_names=['fold_growth', 'length_birth'],
            length_mean=np.mean(mom_global['length_birth']), length_std=np.std(mom_global['length_birth']), phi_mean=np.mean(mom_global['fold_growth']), phi_std=np.std(mom_global['fold_growth']))

        # run the simulation
        reg_all_traces_df, reg_mother_df, reg_daughter_df = fold_growth_and_length_birth_intergenerational_simulation(beta=reg_beta,
                                                                                                          P=np.sqrt(optimal_p_squared),
                                                                                                          S=np.sqrt(optimal_s_squared),
                                                                                                          number_of_traces=1, cycles_per_trace=3000,
                                                                                                          column_names=['fold_growth',
                                                                                                                        'length_birth'],
                                                                                                          length_mean=np.mean(
                                                                                                              mom_global['length_birth']),
                                                                                                          length_std=np.std(
                                                                                                              mom_global['length_birth']),
                                                                                                          phi_mean=np.mean(mom_global['fold_growth']),
                                                                                                          phi_std=np.std(mom_global['fold_growth']))

        # run the simulation
        comp_reg_all_traces_df, comp_reg_mother_df, comp_reg_daughter_df = fold_growth_and_length_birth_intergenerational_simulation_input(model_matrix=np.array([mat.loc['fold_growth'], mat.loc['length_birth']]),
                                                                                                                      P=np.sqrt(optimal_p_squared),
                                                                                                                      S=np.sqrt(optimal_s_squared),
                                                                                                                      number_of_traces=1,
                                                                                                                      cycles_per_trace=3000,
                                                                                                                      column_names=['fold_growth',
                                                                                                                                    'length_birth'],
                                                                                                                      length_mean=np.mean(
                                                                                                                          mom_global['length_birth']),
                                                                                                                      length_std=np.std(
                                                                                                                          mom_global['length_birth']),
                                                                                                                      phi_mean=np.mean(
                                                                                                                          mom_global['fold_growth']),
                                                                                                                      phi_std=np.std(
                                                                                                                          mom_global['fold_growth']))

        # get the pearson coefficients and put them in a matrix in the same order as the model co-variance matrix
        data = np.array([[round(stats.pearsonr(m_df['length_birth'], d_df['length_birth'])[0], 3), round(stats.pearsonr(m_df['length_birth'], d_df['fold_growth'])[0], 3)],
                  [round(stats.pearsonr(m_df['fold_growth'], d_df['length_birth'])[0], 3), round(stats.pearsonr(m_df['fold_growth'], d_df['fold_growth'])[0], 3)]])
        print('data COV \n', data)

        # same as above but with the simulated data
        reg_model = np.array([[round(stats.pearsonr(reg_mother_df['length_birth'], reg_daughter_df['length_birth'])[0], 3),
                          round(stats.pearsonr(reg_mother_df['length_birth'], reg_daughter_df['fold_growth'])[0], 3)],
                         [round(stats.pearsonr(reg_mother_df['fold_growth'], reg_daughter_df['length_birth'])[0], 3),
                          round(stats.pearsonr(reg_mother_df['fold_growth'], reg_daughter_df['fold_growth'])[0], 3)]])
        
        # same as above but with the simulated data
        comp_reg_model = np.array([[round(stats.pearsonr(comp_reg_mother_df['length_birth'], comp_reg_daughter_df['length_birth'])[0], 3),
                               round(stats.pearsonr(comp_reg_mother_df['length_birth'], comp_reg_daughter_df['fold_growth'])[0], 3)],
                              [round(stats.pearsonr(comp_reg_mother_df['fold_growth'], comp_reg_daughter_df['length_birth'])[0], 3),
                               round(stats.pearsonr(comp_reg_mother_df['fold_growth'], comp_reg_daughter_df['fold_growth'])[0], 3)]])

        # same as above but with the simulated data
        optimal_model = np.array([[round(stats.pearsonr(optimal_mother_df['length_birth'], optimal_daughter_df['length_birth'])[0], 3),
                           round(stats.pearsonr(optimal_mother_df['length_birth'], optimal_daughter_df['fold_growth'])[0], 3)],
                          [round(stats.pearsonr(optimal_mother_df['fold_growth'], optimal_daughter_df['length_birth'])[0], 3),
                           round(stats.pearsonr(optimal_mother_df['fold_growth'], optimal_daughter_df['fold_growth'])[0], 3)]])
        print('reg_model COV \n', reg_model)
        print('comp_reg_model COV \n', comp_reg_model)
        print('optimal_model COV \n', optimal_model)
        print('-----------')


def run_schema_with_generationtime():
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

    # change the length births to log length births, including their averages to the log averages
    mom['length_birth'] = np.log(mom['length_birth'])
    daug['length_birth'] = np.log(daug['length_birth'])

    # copy them so we don't mess up the original mother and daughter dfs
    mom_global = mom.copy()
    mom_trap = mom.copy()
    mom_traj = mom.copy()
    daug_global = daug.copy()
    daug_trap = daug.copy()
    daug_traj = daug.copy()

    # log the length and center the fold growth and length
    mom_global['length_birth'] = mom_global['length_birth'] - np.mean(mom_global['length_birth'])
    daug_global['length_birth'] = daug_global['length_birth'] - np.mean(daug_global['length_birth'])
    mom_global['generationtime'] = mom_global['generationtime'] - np.mean(mom_global['generationtime'])
    daug_global['generationtime'] = daug_global['generationtime'] - np.mean(daug_global['generationtime'])

    mom_trap['length_birth'] = mom_trap['length_birth'] - mom_trap['log_trap_avg_length_birth']
    daug_trap['length_birth'] = daug_trap['length_birth'] - daug_trap['log_trap_avg_length_birth']
    mom_trap['generationtime'] = mom_trap['generationtime'] - mom_trap['trap_avg_generationtime']
    daug_trap['generationtime'] = daug_trap['generationtime'] - daug_trap['trap_avg_generationtime']

    mom_traj['length_birth'] = mom_traj['length_birth'] - mom_traj['log_traj_avg_length_birth']
    daug_traj['length_birth'] = daug_traj['length_birth'] - daug_traj['log_traj_avg_length_birth']
    mom_traj['generationtime'] = mom_traj['generationtime'] - mom_traj['traj_avg_generationtime']
    daug_traj['generationtime'] = daug_traj['generationtime'] - daug_traj['traj_avg_generationtime']

    # run it for the pure combinations
    for m_df, d_df in zip([mom_global, mom_trap, mom_traj],
                          [daug_global, daug_trap, daug_traj]):
        # get the linear regression matrix and score
        mat, scores, intercepts = linear_regression_framework(factor_df=m_df[['length_birth', 'generationtime']],
                                                              target_df=d_df[['length_birth', 'generationtime']], fit_intercept=False)

        print('regression matrix\n', mat)
        print('scores\n', scores)

        # get the model parameters from centered data around a certain average
        optimal_beta, optimal_p_squared, optimal_s_squared = generationtime_and_length_birth_intergenerational_model_parameters_analytics(
            df=m_df[['length_birth', 'generationtime']])

        print('optimal beta=', round(optimal_beta, 3))
        print('optimal p squared=', round(optimal_p_squared, 3))
        print('optimal s squared=', round(optimal_s_squared, 3))

        # run the simulation
        all_traces_df, mother_df, daughter_df = fold_growth_and_length_birth_intergenerational_simulation(beta=optimal_beta,
                                                                                                          P=np.sqrt(optimal_p_squared),
                                                                                                          S=np.sqrt(optimal_s_squared),
                                                                                                          number_of_traces=1, cycles_per_trace=3000,
                                                                                                          column_names=['generationtime',
                                                                                                                        'length_birth'],
                                                                                                          length_mean=np.mean(
                                                                                                              mom_global['length_birth']),
                                                                                                          length_std=np.std(
                                                                                                              mom_global['length_birth']),
                                                                                                          phi_mean=np.mean(mom_global['generationtime']),
                                                                                                          phi_std=np.std(mom_global['generationtime']))

        # get the pearson coefficients and put them in a matrix in the same order as the model co-variance matrix
        data = np.array([[round(stats.pearsonr(m_df['length_birth'], d_df['length_birth'])[0], 3),
                          round(stats.pearsonr(m_df['length_birth'], d_df['generationtime'])[0], 3)],
                         [round(stats.pearsonr(m_df['generationtime'], d_df['length_birth'])[0], 3),
                          round(stats.pearsonr(m_df['generationtime'], d_df['generationtime'])[0], 3)]])
        print(data)

        # same as above but with the simulated data
        model = np.array([[round(stats.pearsonr(mother_df['length_birth'], daughter_df['length_birth'])[0], 3),
                           round(stats.pearsonr(mother_df['length_birth'], daughter_df['generationtime'])[0], 3)],
                          [round(stats.pearsonr(mother_df['generationtime'], daughter_df['length_birth'])[0], 3),
                           round(stats.pearsonr(mother_df['generationtime'], daughter_df['generationtime'])[0], 3)]])
        print(model)
        print('-----------')




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





    # the mother
    mom = Population.mother_dfs[0].copy()
    daug = Population.daughter_dfs[0].copy()

    pop_stats = Population.pop_stats.copy()
    pop_stats['length_birth'].loc['mean'] = np.log(pop_stats['length_birth'].loc['mean'])

    mom['length_birth'] = np.log(mom['length_birth'])
    daug['length_birth'] = np.log(daug['length_birth'])
    mom['trap_avg_length_birth'] = np.log(mom['trap_avg_length_birth'])
    daug['trap_avg_length_birth'] = np.log(daug['trap_avg_length_birth'])
    mom['traj_avg_length_birth'] = np.log(mom['traj_avg_length_birth'])
    daug['traj_avg_length_birth'] = np.log(daug['traj_avg_length_birth'])
    #
    # # for the analysis in the black notebook
    # mom['division_ratio'] = np.log(mom['division_ratio'])
    # daug['division_ratio'] = np.log(daug['division_ratio'])
    # mom['trap_avg_division_ratio'] = np.log(mom['trap_avg_division_ratio'])
    # daug['trap_avg_division_ratio'] = np.log(daug['trap_avg_division_ratio'])
    # mom['traj_avg_division_ratio'] = np.log(mom['traj_avg_division_ratio'])
    # daug['traj_avg_division_ratio'] = np.log(daug['traj_avg_division_ratio'])

    mom_global = mom.copy()
    mom_trap = mom.copy()
    mom_traj = mom.copy()
    daug_global = daug.copy()
    daug_trap = daug.copy()
    daug_traj = daug.copy()

    # Global
    for var in Population._variable_names:
        mom_global[var] = mom_global[var] - pop_stats[var].loc['mean']
        daug_global[var] = daug_global[var] - pop_stats[var].loc['mean']

    # Trap
    for var in Population._variable_names:
        mom_trap[var] = mom_trap[var] - mom_trap['trap_avg_' + var]
        daug_trap[var] = daug_trap[var] - daug_trap['trap_avg_' + var]

    # Traj
    for var in Population._variable_names:
        mom_traj[var] = mom_traj[var] - mom_trap['traj_avg_' + var]
        daug_traj[var] = daug_traj[var] - daug_trap['traj_avg_' + var]



    # trap_1 = - mom_trap['length_birth'] + daug_trap['length_birth']
    # trap_2 = mom_trap['fold_growth'] + mom_trap['division_ratio'] + mom_trap['trap_avg_fold_growth'] + mom_trap['trap_avg_division_ratio']
    #
    # global_1 = - mom_global['length_birth'] + daug_global['length_birth']
    # global_2 = mom_global['fold_growth'] + mom_global['division_ratio'] + pop_stats['fold_growth'].loc['mean'] + pop_stats['division_ratio'].loc['mean']
    #
    # global_3 = - mom_global['length_birth'] + daug_global['length_birth']
    # global_4 = mom_global['fold_growth'] + mom_global['division_ratio']
    #
    # sns.regplot(x=global_1, y=global_2, label=round(stats.pearsonr(global_1, global_2)[0], 3))
    # plt.legend()
    # plt.show()
    # plt.close()
    #
    # sns.regplot(x=global_3, y=global_4, label=round(stats.pearsonr(global_3, global_4)[0], 3))
    # plt.legend()
    # plt.show()
    # plt.close()
    #
    # sns.regplot(x=trap_1, y=trap_2, label=round(stats.pearsonr(trap_1, trap_2)[0], 3))
    # plt.legend()
    # plt.show()
    # plt.close()

    exit()

    primary_vars = ['generationtime', 'length_birth', 'growth_rate', 'fold_growth']
        
    # the linear regression of mother daughter
    for m_df, d_df in zip([mom_global[primary_vars], mom_trap[primary_vars], mom_traj[primary_vars]],
                          [daug_global[primary_vars], daug_trap[primary_vars], daug_traj[primary_vars]]):
        # get the linear regression
        mat, scores, intercepts = linear_regression_framework(factor_df=m_df, target_df=d_df, fit_intercept=False)

        print('regression matrix\n')
        print_full_dataframe(mat)
        print('scores\n', scores)
        print('\n -------------------------- \n')

    # the linear regression of same-cell
    for m_df, d_df in zip([mom_global[primary_vars], mom_trap[primary_vars], mom_traj[primary_vars]],
                          [daug_global[primary_vars], daug_trap[primary_vars], daug_traj[primary_vars]]):
        # get the linear regression
        mat, scores, intercepts = linear_regression_framework(factor_df=m_df, target_df=d_df, fit_intercept=False)

        print('regression matrix\n')
        print_full_dataframe(mat)
        print('scores\n', scores)
        print('\n -------------------------- \n')
        
    
    
    
    exit()

    mother_df = Population.mother_dfs[0].copy()
    
    traj_avg_mother_df = pd.DataFrame(columns=['length_birth', 'fold_growth'])
    traj_avg_mother_df['length_birth'] = np.log(mother_df['length_birth'] / mother_df['traj_avg_length_birth'])
    traj_avg_mother_df['fold_growth'] = mother_df['fold_growth'] - mother_df['traj_avg_fold_growth']
    
    trap_avg_mother_df = pd.DataFrame(columns=['length_birth', 'fold_growth'])
    trap_avg_mother_df['length_birth'] = np.log(mother_df['length_birth'] / mother_df['trap_avg_length_birth'])
    trap_avg_mother_df['fold_growth'] = mother_df['fold_growth'] - mother_df['trap_avg_fold_growth']
    
    global_avg_mother_df = pd.DataFrame(columns=['length_birth', 'fold_growth'])
    global_avg_mother_df['length_birth'] = np.log(mother_df['length_birth'] / Population.pop_stats['length_birth'].loc['mean'])
    global_avg_mother_df['fold_growth'] = mother_df['fold_growth'] - Population.pop_stats['fold_growth'].loc['mean']

    # the daughter

    daughter_df = Population.daughter_dfs[0].copy()

    traj_avg_daughter_df = pd.DataFrame(columns=['length_birth', 'fold_growth'])
    traj_avg_daughter_df['length_birth'] = np.log(daughter_df['length_birth'] / daughter_df['traj_avg_length_birth'])
    traj_avg_daughter_df['fold_growth'] = daughter_df['fold_growth'] - daughter_df['traj_avg_fold_growth']

    trap_avg_daughter_df = pd.DataFrame(columns=['length_birth', 'fold_growth'])
    trap_avg_daughter_df['length_birth'] = np.log(daughter_df['length_birth'] / daughter_df['trap_avg_length_birth'])
    trap_avg_daughter_df['fold_growth'] = daughter_df['fold_growth'] - daughter_df['trap_avg_fold_growth']

    global_avg_daughter_df = pd.DataFrame(columns=['length_birth', 'fold_growth'])
    global_avg_daughter_df['length_birth'] = np.log(daughter_df['length_birth'] / Population.pop_stats['length_birth'].loc['mean'])
    global_avg_daughter_df['fold_growth'] = daughter_df['fold_growth'] - Population.pop_stats['fold_growth'].loc['mean']

    # for var in ['trap_avg_length_birth', 'trap_avg_fold_growth', 'traj_avg_length_birth', 'traj_avg_fold_growth']:

    # # get the histograms of the averages for the simulation
    # for var in ['trap_avg_length_birth', 'trap_avg_fold_growth', 'traj_avg_length_birth', 'traj_avg_fold_growth']:
    #     sns.set_style('darkgrid')
    #     plt.hist(x=mother_df[var], weights=1/np.sum(mother_df[var])*np.ones_like(mother_df[var]), label=r'${:.2e} \pm {:.2e}$'.format(np.mean(mother_df[var]), np.std(mother_df[var])))
    #     plt.legend()
    #     plt.xlabel(var)
    #     plt.ylabel('pdf')
    #     # plt.show()
    #     plt.savefig(r'/Users/alestawsky/PycharmProjects/untitled/Histograms/'+var+'.png', dpi=300)
    #     plt.close()
    #
    # combs = np.array(list(itertools.combinations(['trap_avg_length_birth', 'trap_avg_fold_growth', 'traj_avg_length_birth', 'traj_avg_fold_growth'], 2)))
    #
    # for var1, var2 in zip(combs[:, 0], combs[:, 1]):
    #     sns.regplot(x=var1, y=var2, data=mother_df, label=r'$\rho={:.2e}$'.format(stats.pearsonr(mother_df[var1], mother_df[var2])[0]))
    #     plt.legend()
    #     plt.xlabel(var1)
    #     plt.ylabel(var2)
    #     plt.savefig(r'/Users/alestawsky/PycharmProjects/untitled/Correlation Regression Plots for trap and traj averages/' + var1+' '+var2 + '.png', dpi=300)
    #     # plt.show()
    #     plt.close()
    #
    # exit()

    regression_table = pd.DataFrame(columns=['Global', 'Trap', 'Traj'], index=[r'$R_{\delta y}^2$', r'$R_{\delta \phi}^2$', r'$R_{\delta y}^2$'])

    # run it for the pure combinations
    for m_df, d_df in zip([global_avg_mother_df, trap_avg_mother_df, traj_avg_mother_df],
                          [global_avg_daughter_df, trap_avg_daughter_df, traj_avg_daughter_df]):
        # get the linear regression
        mat, scores, intercepts = linear_regression_framework(factor_df=m_df, target_df=d_df, fit_intercept=False)

        print('regression matrix\n', mat)
        print('scores\n', scores)

        # get the model parameters from data
        optimal_beta, optimal_p_squared, optimal_s_squared = fold_growth_and_length_birth_intergenerational_model_parameters_analytics(df=m_df)

        print('optimal beta=', optimal_beta)
        print('optimal p squared=', optimal_p_squared)
        print('optimal s squared=', optimal_s_squared)

        # fold_growth_and_length_birth_intergenerational_simulation(beta=optimal_beta, P=np.sqrt(optimal_p_squared), S=np.sqrt(optimal_s_squared),
        #                                                           number_of_traces=1, cycles_per_trace=5000, column_names=,
        #                                                           mean_and_std_of_initial_trace_vars)



if __name__ == '__main__':
    # main()
    run_schema()
    # run_schema_with_generationtime()
