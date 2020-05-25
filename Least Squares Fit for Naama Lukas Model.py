import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression as LR
import matplotlib.pyplot as plt
from correlation_dataframe_and_heatmap import output_heatmap
from correlation_dataframe_and_heatmap import correlation_dataframe
import pickle
import scipy.stats as stats
import scipy.optimize as optimize
from lmfit import minimize, Parameters


def normalize_the_covariance(matrix):

    # check if it is square and dimension bigger than just 1
    if len(matrix.shape) < 2 or matrix.shape[0] != matrix.shape[1]:
        IOError('Either not square or the dimension is smaller than 2')

    new_mat = np.ones_like(matrix)
    # normalize the matrix
    for row_num in np.arange(matrix.shape[0]):
        for col_num in np.arange(matrix.shape[1]):
            new_mat[row_num, col_num] = matrix[row_num, col_num] / np.sqrt(np.abs(matrix[row_num, row_num])*np.abs(matrix[col_num, col_num]))

    return new_mat


''' alpha, beta, a, b, P^2 and S^2 as specified in Lukas's formalism '''
def theoretic_correlations(alpha, beta, a, b, P, S):

    A_matrix = np.array([[-beta, -beta / alpha], [alpha, 1]])

    beta_coefficient = 1 / (-(beta ** 2) + 2 * beta)

    stationary_covariance = np.array([[(P ** 2) * (2 * beta) + (S ** 2) * ((beta ** 2) / (alpha ** 2)),
                                       (P ** 2) * (-beta * alpha) + (S ** 2) * (-beta / alpha)],
                                      [(P ** 2) * (-beta * alpha) + (S ** 2) * (-beta / alpha),
                                       (P ** 2) * (alpha ** 2) + (S ** 2) * (-(beta ** 2) + 2 * beta + 1)]])

    stationary_covariance = stationary_covariance * beta_coefficient

    transpose_power_b = np.linalg.matrix_power(A_matrix.T, b)

    the_two_matrices_on_the_right = np.matmul(stationary_covariance, transpose_power_b)

    A_power_a = np.linalg.matrix_power(A_matrix, a)

    correlation_matrix = np.matmul(A_power_a, the_two_matrices_on_the_right)

    covariance_matrix = np.ones_like(stationary_covariance)

    if a == 0 and b == 0: # same cell
        covariance_matrix = normalize_the_covariance(stationary_covariance)
        # covariance_matrix = stationary_covariance
    else: # mother/daughter
        for row in range(stationary_covariance.shape[0]):
            for col in range(stationary_covariance.shape[1]):
                covariance_matrix[row, col] = (correlation_matrix[row, col] / np.sqrt(np.abs(stationary_covariance[row, row]) *
                                                                                      np.abs(stationary_covariance[col, col])))

    return covariance_matrix


def completely_synthetic_simulation(growth_average, beta, c_1=0.2, c_2=0.2, number_of_traces=800, cycles_per_trace=50, column_names=
['generationtime', 'log-normalized_length_birth', 'growth_length', 'division_ratios__f_n', 'log-normalized_length_final', 'phi']):
    # Import the Refined Data
    pickle_in = open("metastructdata.pickle", "rb")
    struct = pickle.load(pickle_in)
    pickle_in.close()

    # simple renaming
    m_d_dependance_units = struct.m_d_dependance_units

    # the model matrix with chosen growth rate and beta
    model_matrix = np.array([[1, growth_average], [-beta / growth_average, -beta]])

    # the random initial variables from which the 50 traces will stem from
    mean_of_length = np.mean(np.log(m_d_dependance_units['length_birth_m'] / struct.x_avg))
    std_of_length = np.std(np.log(m_d_dependance_units['length_birth_m'] / struct.x_avg))
    initial_normalized_lengths = np.random.normal(loc=mean_of_length, scale=std_of_length, size=number_of_traces)
    # notice that now we are subtracting the global generationtime average
    initial_deviation_of_gentime_from_mean = np.random.normal(loc=0, scale=np.std(m_d_dependance_units['generationtime_m']), size=number_of_traces)

    # print(initial_normalized_lengths, initial_deviation_of_gentime_from_mean)

    # Going to hold all the synthetic trajectories
    all_traces_df = pd.DataFrame(columns=column_names)
    for trace_num in range(number_of_traces):

        # using the previous generation to predict the next, previous_generation[0] = y_n and previous_generation[1] = delta_tau_n
        previous_generation = np.array([initial_normalized_lengths[trace_num], initial_deviation_of_gentime_from_mean[trace_num]])

        # print('previous generation\n', previous_generation)

        for cycle_num in range(cycles_per_trace):
            # Convert delta_tau_n to tau_n so as to compare to the data, then the log normal length birth y_n, growth average, division ratio,
            # then the log normal final length based on the exponential model, finally the fold growth. Essentially looks like:
            # ['generationtime(added the mean)', 'log-normalized_length_birth', 'growth_length', 'division_ratios__f_n_m',
            # 'log-normalized_length_final', 'phi']
            tau_n = previous_generation[1] + np.mean(m_d_dependance_units['generationtime_m'])
            y_n = previous_generation[0]
            phi_n = growth_average * tau_n
            log_normal_final_length = previous_generation[0] + phi_n

            # Arrange it into a pandas series/array
            what_to_append = pd.Series([tau_n, y_n, growth_average, 0.5, log_normal_final_length, phi_n])
            # Define their order with the function parameter column_names
            all_traces_df = all_traces_df.append(dict(zip(column_names, what_to_append)), ignore_index=True)

            # print('what_to_append\n', what_to_append)

            # standard normal noise with coefficients named c_1, c_2
            noise = np.array([c_1 * np.random.normal(0, 1, 1), c_2 * np.random.normal(0, 1, 1)])

            # print('noise\n', noise)

            # Get the next generation which will become the previous generation so that's what I name it, recursive programming
            previous_generation = np.dot(model_matrix, previous_generation.T) + noise.T

            # this is because, for some reason, the array from the previous action gives the vector inside another, useless array and putting [0]
            # takes it out of this useless array and we get the vector
            previous_generation = previous_generation[0]

    # mother and daughter dataframes
    daughter_df = pd.DataFrame(columns=all_traces_df.columns)
    mother_df = pd.DataFrame(columns=all_traces_df.columns)

    # if the daughter index is a multiple of how many cycles per trace then they are not really mother-daughter and we can't add them to the
    # dataframes
    for mother_index in range(len(all_traces_df) - 1):
        if np.mod(mother_index + 1, cycles_per_trace) != 0:
            daughter_df = daughter_df.append(all_traces_df.loc[mother_index + 1])
            mother_df = mother_df.append(all_traces_df.loc[mother_index])

    return all_traces_df, mother_df, daughter_df


def check_the_theoretic_expressions(beta, P_squared, S_squared, number_of_traces, cycles_per_trace):
    pickle_name = 'metastructdata'
    # Import the Refined Data
    pickle_in = open(pickle_name + ".pickle", "rb")
    struct = pickle.load(pickle_in)
    pickle_in.close()

    m_d_dependance_units = struct.m_d_dependance_units

    # the algebraic expressions
    std_div_dphi = np.sqrt((2*P_squared+beta*S_squared)/(2-beta))
    std_div_y = np.sqrt((P_squared+S_squared*(-(beta**2)+2*beta+1))/(beta*(2-beta)))
    pearson_dphi_and_y = -(P_squared+S_squared) / np.sqrt((2*P_squared**2+(-2*(beta**2)+5*beta+2)*P_squared*S_squared+(-(
            beta**2)+2*beta+1)*S_squared**2)/beta)
    dphi_m_and_dphi_d = (beta*(-P_squared+S_squared*(1-beta)))/(2*P_squared+beta*S_squared)
    y_m_and_y_d = (P_squared*(1-beta)+S_squared*(-(beta**2)+beta+1))/(P_squared+S_squared*(-(beta**2)+2*beta+1))
    dphi_m_and_y_d = (P_squared*(beta-1)+S_squared*(beta**2-beta-1)) / np.sqrt((2*P_squared**2+(-2*(beta**2)+5*beta+2)*P_squared*S_squared+(-(
            beta**2)+2*beta+1)*S_squared**2)/beta)
    dphi_d_and_y_m = (P_squared+S_squared*(beta-1)) / np.sqrt((2*P_squared**2+(-2*(beta**2)+5*beta+2)*P_squared*S_squared+(-(
            beta**2)+2*beta+1)*S_squared**2)/beta)

    all_traces_df, mother_df, daughter_df = completely_synthetic_simulation(growth_average=1, beta=beta, c_1=np.sqrt(P_squared),
                                                                            c_2=np.sqrt(S_squared),
                                                                          number_of_traces=number_of_traces, cycles_per_trace=cycles_per_trace,
        column_names=['generationtime', 'log-normalized_length_birth', 'growth_length', 'division_ratios__f_n', 'log-normalized_length_final', 'phi'])

    same_cell_covariance_matrix = theoretic_correlations(alpha=1, beta=beta, a=0, b=0, P=np.sqrt(P_squared), S=np.sqrt(S_squared))
    m_d_covariance_matrix = theoretic_correlations(alpha=1, beta=beta, a=1, b=0, P=np.sqrt(P_squared), S=np.sqrt(S_squared))

    print('standard div phi: \n', std_div_dphi, np.std(mother_df['phi'] - np.mean(mother_df['phi'])))
    print('standard div log-normalized_length_birth: \n', std_div_y, np.std(mother_df['log-normalized_length_birth']))
    print('pearson_dphi_and_y: \n', pearson_dphi_and_y)
    print('same_cell_covariance_matrix: \n', same_cell_covariance_matrix)
    print('dphi_m_and_dphi_d, y_m_and_y_d, dphi_m_and_y_d, dphi_d_and_y_m: \n', dphi_m_and_dphi_d, y_m_and_y_d, dphi_m_and_y_d, dphi_d_and_y_m)
    print('m_d_covariance_matrix: \n', m_d_covariance_matrix)

    print('phi std: \n', np.std(m_d_dependance_units['phi_m'] - np.mean(m_d_dependance_units['phi_m'])))
    print('NLB std: \n', np.std(np.log(m_d_dependance_units['length_birth_m'] / np.mean(m_d_dependance_units['length_birth_m']))))
    print('r(phi, NLB): \n', stats.pearsonr(m_d_dependance_units['phi_m'] - np.mean(m_d_dependance_units['phi_m']),
                         np.log(m_d_dependance_units['length_birth_m'] / np.mean(m_d_dependance_units['length_birth_m'])))[0])
    print('r(phi_m, phi_d): \n', stats.pearsonr(m_d_dependance_units['phi_m'] - np.mean(m_d_dependance_units['phi_m']),
                                            m_d_dependance_units['phi_d'] - np.mean(m_d_dependance_units['phi_d']))[0])
    print('r(NLB_m, NLB_d): \n', stats.pearsonr(np.log(m_d_dependance_units['length_birth_m'] / np.mean(m_d_dependance_units['length_birth_m'])),
                                            np.log(m_d_dependance_units['length_birth_d'] / np.mean(m_d_dependance_units['length_birth_d'])))[0])
    print('r(phi_m, NLB_d): \n', stats.pearsonr(m_d_dependance_units['phi_m'] - np.mean(m_d_dependance_units['phi_m']),
                                            np.log(m_d_dependance_units['length_birth_d'] / np.mean(m_d_dependance_units['length_birth_d'])))[0])
    print('r(phi_d, NLB_m): \n', stats.pearsonr(m_d_dependance_units['phi_d'] - np.mean(m_d_dependance_units['phi_d']),
                                            np.log(m_d_dependance_units['length_birth_m'] / np.mean(m_d_dependance_units['length_birth_m'])))[0])


def loss_function(parameters):
    # artificial convinient but consistent order
    beta = parameters[0]
    P_squared = parameters[1]
    S_squared = parameters[2]
    # beta = parameters['beta']
    # P_squared = parameters['P_squared']
    # S_squared = parameters['S_squared']

    # the algebraic expressions
    std_div_dphi = np.sqrt((2 * P_squared + beta * S_squared) / (2 - beta))
    std_div_y = np.sqrt((P_squared + S_squared * (-(beta ** 2) + 2 * beta + 1)) / (beta * (2 - beta)))
    pearson_dphi_and_y = -(P_squared + S_squared) / np.sqrt((2 * P_squared ** 2 + (-2 * (beta ** 2) + 5 * beta + 2) * P_squared * S_squared + (-(
            beta ** 2) + 2 * beta + 1) * S_squared ** 2) / beta)
    dphi_m_and_dphi_d = (beta * (-P_squared + S_squared * (1 - beta))) / (2 * P_squared + beta * S_squared)
    y_m_and_y_d = (P_squared * (1 - beta) + S_squared * (-(beta ** 2) + beta + 1)) / (P_squared + S_squared * (-(beta ** 2) + 2 * beta + 1))
    dphi_m_and_y_d = (P_squared * (beta - 1) + S_squared * (beta ** 2 - beta - 1)) / np.sqrt(
        (2 * P_squared ** 2 + (-2 * (beta ** 2) + 5 * beta + 2) * P_squared * S_squared + (-(
                beta ** 2) + 2 * beta + 1) * S_squared ** 2) / beta)
    dphi_d_and_y_m = (P_squared + S_squared * (beta - 1)) / np.sqrt(
        (2 * P_squared ** 2 + (-2 * (beta ** 2) + 5 * beta + 2) * P_squared * S_squared + (-(
                beta ** 2) + 2 * beta + 1) * S_squared ** 2) / beta)

    expressions = [std_div_dphi, std_div_y, pearson_dphi_and_y, dphi_m_and_dphi_d, y_m_and_y_d, dphi_m_and_y_d, dphi_d_and_y_m]

    pickle_name = 'metastructdata'
    # Import the Refined Data
    pickle_in = open(pickle_name + ".pickle", "rb")
    struct = pickle.load(pickle_in)
    pickle_in.close()

    m_d_dependance_units = struct.m_d_dependance_units

    phi_std = np.std(m_d_dependance_units['phi_m'] - np.mean(m_d_dependance_units['phi_m']))
    NLB_std = np.std(np.log(m_d_dependance_units['length_birth_m'] / np.mean(m_d_dependance_units['length_birth_m'])))
    phi_NLB = stats.pearsonr(m_d_dependance_units['phi_m'] - np.mean(m_d_dependance_units['phi_m']),
                         np.log(m_d_dependance_units['length_birth_m'] / np.mean(m_d_dependance_units['length_birth_m'])))[0]
    phi_m_phi_d = stats.pearsonr(m_d_dependance_units['phi_m'] - np.mean(m_d_dependance_units['phi_m']),
                                            m_d_dependance_units['phi_d'] - np.mean(m_d_dependance_units['phi_d']))[0]
    NLB_m_NLB_d = stats.pearsonr(np.log(m_d_dependance_units['length_birth_m'] / np.mean(m_d_dependance_units['length_birth_m'])),
                                            np.log(m_d_dependance_units['length_birth_d'] / np.mean(m_d_dependance_units['length_birth_d'])))[0]
    phi_m_NLB_d = stats.pearsonr(m_d_dependance_units['phi_m'] - np.mean(m_d_dependance_units['phi_m']),
                                            np.log(m_d_dependance_units['length_birth_d'] / np.mean(m_d_dependance_units['length_birth_d'])))[0]
    phi_d_NLB_m = stats.pearsonr(m_d_dependance_units['phi_d'] - np.mean(m_d_dependance_units['phi_d']),
                                            np.log(m_d_dependance_units['length_birth_m'] / np.mean(m_d_dependance_units['length_birth_m'])))[0]

    data_analog = [phi_std, NLB_std, phi_NLB, phi_m_phi_d, NLB_m_NLB_d, phi_m_NLB_d, phi_d_NLB_m]

    return np.sum(np.array([(prediction - data)**2 for prediction, data in zip(expressions, data_analog)]))


def main():
    # pickle_name = 'metastructdata'
    # # Import the Refined Data
    # pickle_in = open(pickle_name + ".pickle", "rb")
    # struct = pickle.load(pickle_in)
    # pickle_in.close()

    beta, P_squared, S_squared = 0.25, 0.01, 0.01
    parameters = [beta, P_squared, S_squared]

    # params = Parameters()
    # params.add('beta', value=.25, min=0, max=2)
    # params.add('P_squared', value=.01, vary=True)
    # params.add('S_squared', value=.01, vary=True)

    check_the_theoretic_expressions(beta=beta, P_squared=P_squared, S_squared=S_squared, number_of_traces=2, cycles_per_trace=1000)

    # out = minimize(loss_function, params=params)
    #
    # print(out)

    res = optimize.least_squares(loss_function, parameters)

    print(res)


if __name__ == '__main__':
    main()
