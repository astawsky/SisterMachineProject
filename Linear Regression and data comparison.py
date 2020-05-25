
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

""" Gets the model parameters from the same-cell centered data """
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


'''
Runs the simulation with a specific length/phi mean and std, beta, P, S ets... 
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


def mother_daughter_correlation_comparison_with_lukas(beta, P_squared, S_squared):
    var_y = (P_squared+S_squared*(-(beta**2)+2*beta+1))/(beta*(2-beta))
    var_phi = (2*P_squared+beta*S_squared)/(2-beta)
    phi_m_phi_d = round(beta*((-P_squared+S_squared*(1-beta))/(2*P_squared+beta*S_squared)), 3)
    y_m_y_d = round((P_squared*(1-beta)+S_squared*(-(beta**2)+beta+1)) / (P_squared+S_squared*(-(beta**2)+2*beta+1)), 3)
    # lukas_y_m_y_d = round((P_squared*(beta-1)+S_squared)/(beta*P_squared+2*S_squared), 3)
    phi_m_y_d = round(((P_squared*beta+S_squared*(beta*(beta-1)))/(beta*(2-beta)))/np.sqrt((var_y*var_phi)), 3)
    y_m_phi_d = round(((P_squared*(beta*(beta-1))+S_squared*(beta*(beta**2-beta-1)))/(beta*(2-beta)))/np.sqrt((var_y*var_phi)), 3)

    print('analytic results')

    print(
        np.array(
            [[y_m_y_d, y_m_phi_d], [phi_m_y_d, phi_m_phi_d]]
        )
    )


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

    # # change the length births to log length births, including their averages to the log averages
    # mom['length_birth'] = np.log(mom['length_birth'])
    # daug['length_birth'] = np.log(daug['length_birth'])

    # copy them so we don't mess up the original mother and daughter dfs
    mom_global = mom.copy()
    mom_trap = mom.copy()
    mom_traj = mom.copy()
    daug_global = daug.copy()
    daug_trap = daug.copy()
    daug_traj = daug.copy()

    # # FIRST WAY! Dividing by the original mean
    # # log the length and center the fold growth and length for all three average-specific m/d dataframes
    # print('1. Dividing by the original mean')
    # mom_global['length_birth'] = np.log(mom_global['length_birth'] / np.mean(mom_global['length_birth']))
    # daug_global['length_birth'] = np.log(daug_global['length_birth'] / np.mean(daug_global['length_birth']))
    # mom_global['fold_growth'] = mom_global['fold_growth'] - np.mean(mom_global['fold_growth'])
    # daug_global['fold_growth'] = daug_global['fold_growth'] - np.mean(daug_global['fold_growth'])
    #
    # mom_trap['length_birth'] = np.log(mom_trap['length_birth'] / mom_trap['trap_avg_length_birth'])
    # daug_trap['length_birth'] = np.log(daug_trap['length_birth'] / daug_trap['trap_avg_length_birth'])
    # mom_trap['fold_growth'] = mom_trap['fold_growth'] - mom_trap['trap_avg_fold_growth']
    # daug_trap['fold_growth'] = daug_trap['fold_growth'] - daug_trap['trap_avg_fold_growth']
    #
    # mom_traj['length_birth'] = np.log(mom_traj['length_birth'] / mom_traj['traj_avg_length_birth'])
    # daug_traj['length_birth'] = np.log(daug_traj['length_birth'] / daug_traj['traj_avg_length_birth'])
    # mom_traj['fold_growth'] = mom_traj['fold_growth'] - mom_traj['traj_avg_fold_growth']
    # daug_traj['fold_growth'] = daug_traj['fold_growth'] - daug_traj['traj_avg_fold_growth']


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

    # run it for the different kinds of averages
    for m_df, d_df in zip([mom_global, mom_trap, mom_traj],
                          [daug_global, daug_trap, daug_traj]):

        # get the linear regression matrix and score
        mat, scores, intercepts = linear_regression_framework(factor_df=m_df[['length_birth', 'fold_growth']],
                                                                       target_df=d_df[['length_birth', 'fold_growth']], fit_intercept=False)

        print('regression matrix\n', mat)
        print('scores\n', scores)

        # get the model parameters from centered data around a certain average
        optimal_beta, optimal_p_squared, optimal_s_squared = fold_growth_and_length_birth_intergenerational_model_parameters_analytics(
            df=m_df[['length_birth', 'fold_growth']])

        # print what we got
        print('optimal beta=', round(optimal_beta, 3))
        print('optimal p squared=', round(optimal_p_squared, 3))
        print('optimal s squared=', round(optimal_s_squared, 3))

        # run the simulation, Try running it with many number of traces and different length/phi means/stds
        optimal_all_traces_df, optimal_mother_df, optimal_daughter_df = fold_growth_and_length_birth_intergenerational_simulation(
            beta=optimal_beta, P=np.sqrt(optimal_p_squared), S=np.sqrt(optimal_s_squared), number_of_traces=1, cycles_per_trace=3000,
              column_names=['fold_growth', 'length_birth'], length_mean=np.mean(mom_global['length_birth']),
              length_std=np.std(mom_global['length_birth']), phi_mean=np.mean(mom_global['fold_growth']),
              phi_std=np.std(mom_global['fold_growth']))

        # # get the pearson coefficients and put them in a matrix in the same order as the model co-variance matrix
        # data = pd.DataFrame(columns=[['length_birth', 'fold_growth']], index=[['length_birth', 'fold_growth']])
        # for col in ['length_birth', 'fold_growth']:
        #     for ind in ['length_birth', 'fold_growth']:
        #         data[col].loc[ind] = round(stats.pearsonr(m_df[col], d_df[ind])[0], 3)



        data = np.array([[round(stats.pearsonr(m_df['length_birth'], d_df['length_birth'])[0], 3),
                          round(stats.pearsonr(m_df['length_birth'], d_df['fold_growth'])[0], 3)],
                         [round(stats.pearsonr(m_df['fold_growth'], d_df['length_birth'])[0], 3),
                          round(stats.pearsonr(m_df['fold_growth'], d_df['fold_growth'])[0], 3)]])
        print('data COV \n', data)

        # get the pearson coefficients and put them in a matrix in the same order as the model co-variance matrix
        # optimal_model = pd.DataFrame(columns=[['length_birth', 'fold_growth']], index=[['length_birth', 'fold_growth']])
        # for col in ['length_birth', 'fold_growth']:
        #     for ind in ['length_birth', 'fold_growth']:
        #         left = optimal_mother_df[col]
        #         right = optimal_daughter_df[ind]
        #         optimal_model[col].loc[ind] = round(stats.pearsonr(left, right)[0], 3)

        # # same as above but with the simulated data
        optimal_model = np.array([[round(stats.pearsonr(optimal_mother_df['length_birth'], optimal_daughter_df['length_birth'])[0], 3),
                                   round(stats.pearsonr(optimal_mother_df['length_birth'], optimal_daughter_df['fold_growth'])[0], 3)],
                                  [round(stats.pearsonr(optimal_mother_df['fold_growth'], optimal_daughter_df['length_birth'])[0], 3),
                                   round(stats.pearsonr(optimal_mother_df['fold_growth'], optimal_daughter_df['fold_growth'])[0], 3)]])
        print('optimal_model COV \n', optimal_model)

        mother_daughter_correlation_comparison_with_lukas(optimal_beta, optimal_p_squared, optimal_s_squared)
        print('-----------')


if __name__ == '__main__':
    main()
