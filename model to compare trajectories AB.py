
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


def get_mother_daughter_dfs(df):
    df_size = len(df)

    mother_df = pd.DataFrame(columns=df.columns)
    daughter_df = pd.DataFrame(columns=df.columns)

    for mom_ind, daug_ind in zip(range(df_size-1), range(1, df_size)):
        mother_df = mother_df.append(df.iloc[mom_ind], ignore_index=True)
        daughter_df = daughter_df.append(df.iloc[daug_ind], ignore_index=True)

    return mother_df, daughter_df


def higher_up(list_of_keys, A_dict, B_dict, global_avg, average):
    # the covariance lists that contain the
    y_m_y_d, phi_m_phi_d, y_m_phi_d, phi_m_y_d = [], [], [], []

    for pair in list_of_keys:

        if average == 'Global':
            average_df_A = global_avg
            average_df_B = global_avg
        elif average == 'Trap':
            average_df_A = pd.concat([A_dict[pair[0]].copy(), B_dict[pair[1]].copy()], axis=0).mean()
            average_df_B = pd.concat([A_dict[pair[0]].copy(), B_dict[pair[1]].copy()], axis=0).mean()
        elif average == 'Traj':
            average_df_A = A_dict[pair[0]].mean()
            average_df_B = B_dict[pair[1]].mean()
        else:
            print('Error defining average in higher_up')
            exit()

        # subtract the desired mean
        trajA = A_dict[pair[0]].copy()

        # print(trajA)

        for col in average_df_A.columns:
            trajA[col] = trajA[col] - average_df_A[col].iloc[0]
        trajB = B_dict[pair[0]].copy()
        for col in average_df_B.columns:
            trajB[col] = trajB[col] - average_df_B[col].iloc[0]

        schema(trajA, trajB) #y_m_y_d, phi_m_phi_d, y_m_phi_d, phi_m_y_d = 

    return y_m_y_d, phi_m_phi_d, y_m_phi_d, phi_m_y_d


def schema(trajA, trajB):

    # Get the mother and daughter dfs to get their Co-variance matrices
    mother_A, daughter_A = get_mother_daughter_dfs(trajA)
    mother_B, daughter_B = get_mother_daughter_dfs(trajB)
    
    # limit them to the variables we care about
    mother_A = mother_A[['length_birth', 'fold_growth']]
    mother_B = mother_B[['length_birth', 'fold_growth']]
    daughter_A = daughter_A[['length_birth', 'fold_growth']]
    daughter_B = daughter_B[['length_birth', 'fold_growth']]
    
    # the MD covariance for trace A 
    A_mother_daughter_df = pd.DataFrame(columns=['length_birth_m', 'fold_growth_m'], index=['length_birth_d', 'fold_growth_d'])
    A_mother_daughter_df['length_birth_m'].loc['length_birth_d'] = round(stats.pearsonr(mother_A['length_birth'], daughter_A['length_birth'])[0], 3)
    A_mother_daughter_df['length_birth_m'].loc['fold_growth_d'] = round(stats.pearsonr(mother_A['length_birth'], daughter_A['fold_growth'])[0], 3)
    A_mother_daughter_df['fold_growth_m'].loc['length_birth_d'] = round(stats.pearsonr(mother_A['fold_growth'], daughter_A['length_birth'])[0], 3)
    A_mother_daughter_df['fold_growth_m'].loc['fold_growth_d'] = round(stats.pearsonr(mother_A['fold_growth'], daughter_A['fold_growth'])[0], 3)

    # the MD covariance for trace B
    B_mother_daughter_df = pd.DataFrame(columns=['length_birth_m', 'fold_growth_m'], index=['length_birth_d', 'fold_growth_d'])
    B_mother_daughter_df['length_birth_m'].loc['length_birth_d'] = round(stats.pearsonr(mother_B['length_birth'], daughter_B['length_birth'])[0], 3)
    B_mother_daughter_df['length_birth_m'].loc['fold_growth_d'] = round(stats.pearsonr(mother_B['length_birth'], daughter_B['fold_growth'])[0], 3)
    B_mother_daughter_df['fold_growth_m'].loc['length_birth_d'] = round(stats.pearsonr(mother_B['fold_growth'], daughter_B['length_birth'])[0], 3)
    B_mother_daughter_df['fold_growth_m'].loc['fold_growth_d'] = round(stats.pearsonr(mother_B['fold_growth'], daughter_B['fold_growth'])[0], 3)

    # optimal_model = np.array([[round(stats.pearsonr(mother_A['length_birth'], mother_A['length_birth'])[0], 3),
    #                            round(stats.pearsonr(mother_A['length_birth'], mother_A['fold_growth'])[0], 3)],
    #                           [round(stats.pearsonr(mother_A['fold_growth'], mother_A['length_birth'])[0], 3),
    #                            round(stats.pearsonr(mother_A['fold_growth'], mother_A['fold_growth'])[0], 3)]])
    #
    # A_mother_daughter = np.corrcoef(mother_A[['length_birth', 'fold_growth']], daughter_A[['length_birth', 'fold_growth']], rowvar=False)
    # B_mother_daughter = np.corrcoef(mother_B[['length_birth', 'fold_growth']], daughter_B[['length_birth', 'fold_growth']], rowvar=False)

    # get the model parameters from centered data around a certain average
    A_optimal_beta, A_optimal_p_squared, A_optimal_s_squared = fold_growth_and_length_birth_intergenerational_model_parameters_analytics(
        df=trajA[['length_birth', 'fold_growth']])

    print('A_optimal beta=', round(A_optimal_beta, 3))
    print('A_optimal p squared=', round(A_optimal_p_squared, 3))
    print('A_optimal s squared=', round(A_optimal_s_squared, 3))

    # run the simulation, NOTE WE ARE USING THE TRAP MEAN AND STD FOR PHI AND Y TO GET A BETTER APPROX, NOT THE GLOBAL LIKE BEFORE
    A_optimal_all_traces_df, A_optimal_mother_df, A_optimal_daughter_df = fold_growth_and_length_birth_intergenerational_simulation(
          beta=A_optimal_beta,
          P=np.sqrt(A_optimal_p_squared),
          S=np.sqrt(A_optimal_s_squared),
          number_of_traces=1,
          cycles_per_trace=3000,
          column_names=['fold_growth', 'length_birth'],
          length_mean=trajA.mean()['length_birth'],
          length_std=trajA.std()['length_birth'],
          phi_mean=trajA.mean()['fold_growth'],
          phi_std=trajA.std()['fold_growth'])

    # the simulation mother/daughter correlation
    # the MD covariance for simulated trace A 
    A_simulated_mother_daughter_df = pd.DataFrame(columns=['length_birth_m', 'fold_growth_m'], index=['length_birth_d', 'fold_growth_d'])
    A_simulated_mother_daughter_df['length_birth_m'].loc['length_birth_d'] = round(stats.pearsonr(A_optimal_mother_df['length_birth'], A_optimal_daughter_df['length_birth'])[0], 3)
    A_simulated_mother_daughter_df['length_birth_m'].loc['fold_growth_d'] = round(stats.pearsonr(A_optimal_mother_df['length_birth'], A_optimal_daughter_df['fold_growth'])[0], 3)
    A_simulated_mother_daughter_df['fold_growth_m'].loc['length_birth_d'] = round(stats.pearsonr(A_optimal_mother_df['fold_growth'], A_optimal_daughter_df['length_birth'])[0], 3)
    A_simulated_mother_daughter_df['fold_growth_m'].loc['fold_growth_d'] = round(stats.pearsonr(A_optimal_mother_df['fold_growth'], A_optimal_daughter_df['fold_growth'])[0], 3)

    # get the model parameters from centered data around a certain average
    B_optimal_beta, B_optimal_p_squared, B_optimal_s_squared = fold_growth_and_length_birth_intergenerational_model_parameters_analytics(
        df=trajB[['length_birth', 'fold_growth']])

    print('B_optimal beta=', round(B_optimal_beta, 3))
    print('B_optimal p squared=', round(B_optimal_p_squared, 3))
    print('B_optimal s squared=', round(B_optimal_s_squared, 3))

    # run the simulation, NOTE WE ARE USING THE TRAP MEAN AND STD FOR PHI AND Y TO GET A BETTER APPROX, NOT THE GLOBAL LIKE BEFORE
    B_optimal_all_traces_df, B_optimal_mother_df, B_optimal_daughter_df = fold_growth_and_length_birth_intergenerational_simulation(
        beta=B_optimal_beta,
        P=np.sqrt(
            B_optimal_p_squared),
        S=np.sqrt(
            B_optimal_s_squared),
        number_of_traces=1,
        cycles_per_trace=3000,
        column_names=[
            'fold_growth',
            'length_birth'],
        length_mean=
        trajB.mean()[
            'length_birth'],
        length_std=trajB.std()[
            'length_birth'],
        phi_mean=trajB.mean()[
            'fold_growth'],
        phi_std=trajB.std()[
            'fold_growth'])

    # the simulation mother/daughter correlation
    # the MD covariance for simulated trace B 
    B_simulated_mother_daughter_df = pd.DataFrame(columns=['length_birth_m', 'fold_growth_m'], index=['length_birth_d', 'fold_growth_d'])
    B_simulated_mother_daughter_df['length_birth_m'].loc['length_birth_d'] = round(
        stats.pearsonr(B_optimal_mother_df['length_birth'], B_optimal_daughter_df['length_birth'])[0], 3)
    B_simulated_mother_daughter_df['length_birth_m'].loc['fold_growth_d'] = round(
        stats.pearsonr(B_optimal_mother_df['length_birth'], B_optimal_daughter_df['fold_growth'])[0], 3)
    B_simulated_mother_daughter_df['fold_growth_m'].loc['length_birth_d'] = round(
        stats.pearsonr(B_optimal_mother_df['fold_growth'], B_optimal_daughter_df['length_birth'])[0], 3)
    B_simulated_mother_daughter_df['fold_growth_m'].loc['fold_growth_d'] = round(
        stats.pearsonr(B_optimal_mother_df['fold_growth'], B_optimal_daughter_df['fold_growth'])[0], 3)

    # Now we pair the A simulated MD COV to the B data MD COV for all the COV and output the arrays
    print(A_mother_daughter_df)
    print(A_simulated_mother_daughter_df)
    print(B_mother_daughter_df)
    print(B_simulated_mother_daughter_df)
    
    md_corrs.update({'A_data': , 'A_simulation': , 'B_data': , 'B_simulation': })


    # return





    # for m_df, d_df, traj in zip([mother_A, mother_B], [daughter_A, daughter_B], [trajA, trajB]):
    #
    #     # get the model parameters from centered data around a certain average
    #     optimal_beta, optimal_p_squared, optimal_s_squared = fold_growth_and_length_birth_intergenerational_model_parameters_analytics(
    #         df=traj[['length_birth', 'fold_growth']])
    #
    #     print('optimal beta=', round(optimal_beta, 3))
    #     print('optimal p squared=', round(optimal_p_squared, 3))
    #     print('optimal s squared=', round(optimal_s_squared, 3))
    #
    #     # run the simulation, NOTE WE ARE USING THE TRAP MEAN AND STD FOR PHI AND Y TO GET A BETTER APPROX, NOT THE GLOBAL LIKE BEFORE
    #     optimal_all_traces_df, optimal_mother_df, optimal_daughter_df = fold_growth_and_length_birth_intergenerational_simulation(beta=optimal_beta,
    #           P=np.sqrt(optimal_p_squared), S=np.sqrt(optimal_s_squared), number_of_traces=1, cycles_per_trace=3000,
    #           column_names=['fold_growth', 'length_birth'], length_mean=trajA.mean()['length_birth'],
    #           length_std=trajA.std()['length_birth'], phi_mean=trajA.mean()['fold_growth'], phi_std=trajA.std()['fold_growth'])
    #
    #     # the simulation mother/daughter correlation
    #     simulation_mother_daughter = np.corrcoef(optimal_mother_df, optimal_daughter_df)
    #
    #     print(simulation_mother_daughter)
        
        # STILL HAVENT COMPARED THEM!!!!!


    # A_same_cell = trajA.cov()


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

    # get the global averages
    log_length_birth_mean = np.log(Population.mother_dfs[0].copy()['length_birth']).mean()
    fold_growth_mean = Population.mother_dfs[0].copy()['fold_growth'].mean()

    global_avg = pd.DataFrame(data=dict({'length_birth': log_length_birth_mean, 'fold_growth': fold_growth_mean}), index=[0])

    print('global avg\n', global_avg)
    
    # minimum length both trajectories need to have in order to be considered in this analysis
    min_length = 20
    
    # for all the pairs greater than or equal to min_length
    long_enough_Sister_pairs = np.array([[keyA, keyB] for keyA, keyB in zip(Sister.A_dict.keys(), Sister.B_dict.keys()) if min(len(Sister.A_dict[keyA]), len(Sister.B_dict[keyB])) >= min_length])
    long_enough_Nonsister_pairs = np.array([[keyA, keyB] for keyA, keyB in zip(Nonsister.A_dict.keys(), Nonsister.B_dict.keys()) if
                                min(len(Nonsister.A_dict[keyA]), len(Nonsister.B_dict[keyB])) >= min_length])
    long_enough_Control_pairs = np.array([[keyA, keyB] for keyA, keyB in zip(Control.A_dict.keys(), Control.B_dict.keys()) if
                                min(len(Control.A_dict[keyA]), len(Control.B_dict[keyB])) >= min_length])

    # FYI
    print('Number of Sister pairs that have at least '+str(min_length)+' generations: '+str(len(long_enough_Sister_pairs)))
    print('Number of Nonsister pairs that have at least ' + str(min_length) + ' generations: ' + str(len(long_enough_Nonsister_pairs)))
    print('Number of Control pairs that have at least ' + str(min_length) + ' generations: ' + str(len(long_enough_Control_pairs)))

    # So we make sure we don't have more samples for a certain dataset and contaminate the pearson correlation with bias
    min_for_all_datasets = min(len(long_enough_Sister_pairs), len(long_enough_Nonsister_pairs), len(long_enough_Control_pairs))
    min_Sister_pairs = long_enough_Sister_pairs[np.random.choice(a=len(long_enough_Sister_pairs), size=min_for_all_datasets, replace=False)]
    min_Nonsister_pairs = long_enough_Nonsister_pairs[np.random.choice(a=len(long_enough_Nonsister_pairs), size=min_for_all_datasets, replace=False)]
    min_Control_pairs = long_enough_Control_pairs[np.random.choice(a=len(long_enough_Control_pairs), size=min_for_all_datasets, replace=False)]

    # higher_up(list_of_keys=long_enough_Sister_pairs, A_dict=Sister.A_dict, B_dict=Sister.B_dict, global_avg=global_avg, average='Global')

    for list_of_keys in [min_Sister_pairs, min_Nonsister_pairs, min_Control_pairs]:
        for avg in ['Global', 'Trap', 'Trace']:
            higher_up(list_of_keys=list_of_keys, A_dict=Sister.A_dict, B_dict=Sister.B_dict, global_avg=global_avg, average=avg)






if __name__ == '__main__':
    main()










