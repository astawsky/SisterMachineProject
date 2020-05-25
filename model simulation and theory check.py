import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression as LR
import matplotlib.pyplot as plt
from correlation_dataframe_and_heatmap import output_heatmap
from correlation_dataframe_and_heatmap import correlation_dataframe
import pickle
import scipy.stats as stats


'''
P and S are the coefficients of noise terms, y and tau
the number of traces we will manufacture
how many cycles we will record per trace
'''
def completely_synthetic_simulation(beta=.3, P=0.2, S=0.2, number_of_traces=1, cycles_per_trace=10000, column_names=['phi_m', 'length_birth_m']):

    # Import the Refined Data
    pickle_in = open("metastructdata.pickle", "rb")
    struct = pickle.load(pickle_in)
    pickle_in.close()

    # simple renaming
    m_d_dependance_units = struct.m_d_dependance_units

    # the model matrix with chosen growth rate and beta
    model_matrix = np.array([[-beta, -beta], [1, 1]])

    # the random initial variables from which the 50 traces will stem from
    mean_of_length = np.mean(m_d_dependance_units['length_birth_m'])
    std_of_length = np.std(m_d_dependance_units['length_birth_m'])

    # mean_of_length = np.mean(np.log(m_d_dependance_units['length_birth_m'] / np.mean(m_d_dependance_units['length_birth_m'])))
    # std_of_length = np.std(np.log(m_d_dependance_units['length_birth_m'] / np.mean(m_d_dependance_units['length_birth_m'])))
    initial_normalized_lengths = np.random.normal(loc=mean_of_length, scale=std_of_length, size=number_of_traces)
    # notice that now we are subtracting the global phi average
    initial_deviation_of_phi_from_mean = np.random.normal(loc=0, scale=np.std(m_d_dependance_units['phi_m']), size=number_of_traces)

    # Going to hold all the synthetic trajectories
    all_traces_df = pd.DataFrame(columns=column_names)

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


def get_the_parameters():
    pickle_name = 'metastructdata'
    # Import the Refined Data
    pickle_in = open(pickle_name + ".pickle", "rb")
    struct = pickle.load(pickle_in)
    pickle_in.close()

    m_d_dependance_units = struct.m_d_dependance_units

    covar_matrix = m_d_dependance_units[['length_birth_m', 'phi_m']].cov()

    length_var_data = covar_matrix['length_birth_m']['length_birth_m']

    phi_var_data = covar_matrix['phi_m']['phi_m']

    mixed_var_data = covar_matrix['length_birth_m']['phi_m']

    print('variance of length, variance of phi, variance of length*phi:', length_var_data, phi_var_data, mixed_var_data)

    optimal_beta = - mixed_var_data / (length_var_data + phi_var_data + 2*mixed_var_data)

    optimal_p_squared = phi_var_data - (mixed_var_data ** 2) / (length_var_data + phi_var_data + 2 * mixed_var_data)

    optimal_s_squared = - 2 * mixed_var_data - phi_var_data

    return length_var_data, phi_var_data, mixed_var_data, optimal_beta, optimal_p_squared, optimal_s_squared


def main():
    pickle_name = 'metastructdata'
    # Import the Refined Data
    pickle_in = open(pickle_name + ".pickle", "rb")
    struct = pickle.load(pickle_in)
    pickle_in.close()

    length_var_data, phi_var_data, mixed_var_data, optimal_beta, optimal_p_squared, optimal_s_squared = get_the_parameters()

    P = np.sqrt(optimal_p_squared)
    S = np.sqrt(optimal_s_squared)
    beta = optimal_beta

    print('P, S, beta:', P, S, beta)

    all_traces_df, mother_df, daughter_df = completely_synthetic_simulation(beta=beta, P=P, S=S, number_of_traces=1, cycles_per_trace=5000,
                                                                          column_names=['phi_m', 'length_birth_m'])

    daughters = daughter_df[['length_birth_m', 'phi_m']]
    daughters.columns = ['length_birth_d', 'phi_d']
    daughters = daughters.reset_index(drop='index')
    mothers = mother_df[['length_birth_m', 'phi_m']]

    together = pd.concat([mothers, daughters], axis=1)

    m_d_dependance_units = struct.m_d_dependance_units

    # print('simulation, same-cell covariance matrix (NOT NORMALIZED):\n', mother_df[['length_birth_m', 'phi_m']].cov())
    # print('data, same-cell covariance matrix (NOT NORMALIZED):\n', m_d_dependance_units[['length_birth_m', 'phi_m']].cov())
    print('simulation covariance matrix (NOT NORMALIZED):\n', together.cov())
    print('data covariance matrix (NOT NORMALIZED):\n', m_d_dependance_units[['length_birth_m', 'phi_m', 'length_birth_d',
                                                                                               'phi_d']].cov())

    exit()

    expression_length = (P**2*(1-beta)+S**2*(-beta**2+beta+1))/(beta*(2-beta))
    expression_phi = (beta*(-P**2+S**2*(beta*(1-beta))))/(2-beta)

    print('simulation of log-norm length \n', np.cov(mother_df['length_birth_m'], daughter_df['length_birth_m']))
    print('theoretical expression of log-norm length \n', expression_length)
    print('simulation of delta phi \n', np.cov(mother_df['phi_m'], daughter_df['phi_m']))
    print('theoretical expression of phi \n', expression_phi)
    exit()

    # print(mother_df)
    # print(daughter_df)
    

if __name__ == '__main__':
    main()
