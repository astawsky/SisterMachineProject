
from __future__ import print_function

import numpy as np

import matplotlib.pyplot as plt

import pickle

import scipy.stats as stats

from mpl_toolkits.mplot3d import Axes3D


def simulation():

    # set the dimension
    dim = 3

    # Creating the inheritance matrix
    norm_of_A = np.random.uniform(0, 1, size=1) # this will set one of the axis
    inheritance_matrix = np.random.rand(dim, dim)
    inheritance_matrix = (inheritance_matrix - np.ones_like(inheritance_matrix) * .5) * 2
    # print(np.linalg.norm(inheritance_matrix))
    inheritance_matrix = (norm_of_A / np.linalg.norm(inheritance_matrix)) * inheritance_matrix
    # print(inheritance_matrix)
    # print(np.linalg.norm(inheritance_matrix))

    # the other part of the axis
    x = np.random.uniform(0, 1, size=3)  # another one of the axis
    N = np.sqrt(x)[0]
    L = np.sqrt(x)[1]
    E = np.sqrt(x)[2]
    # print('N, L, E', x)

    # # # Now to calculate the value of the scaling index!

    # How many times we should repeat the simulation for each dimension
    num_of_reps = 1000

    # Create the covariance matrix for pair-correlated R.V.
    corr_cov = np.identity(2 * dim)
    for ind1 in range(2 * dim):
        for ind2 in range(2 * dim):
            if ind1 == ind2 - dim or ind2 == ind1 - dim or ind1 == ind2:
                corr_cov[ind1][ind2] = 1

    # Mean of the multivariate Gaussian
    mean = np.zeros(2 * dim)

    # Creating the different initial (mother(s)) internal states for the different dsets
    x_init_sis_A = np.random.multivariate_normal(np.zeros(dim), np.identity(dim), num_of_reps)
    x_init_sis_B = x_init_sis_A
    x_init_non_sis_A = np.random.multivariate_normal(np.zeros(dim), np.identity(dim), num_of_reps)
    x_init_non_sis_B = np.random.multivariate_normal(np.zeros(dim), np.identity(dim), num_of_reps)
    x_init_both_A = np.random.multivariate_normal(np.zeros(dim), np.identity(dim), num_of_reps)
    x_init_both_B = np.random.multivariate_normal(np.zeros(dim), np.identity(dim), num_of_reps)

    # Create arrays to store the hidden states of different generations for each pair in each dset
    sis_A_array = []
    sis_B_array = []
    non_sis_A_array = []
    non_sis_B_array = []
    both_A_array = []
    both_B_array = []

    # Create a random projection onto the observable space... Should this change with respect to what we are looking at (growth rate, gen time,
    # size, etc...)? Also, should this be uniform or gaussian as well?
    projection = np.random.uniform(0, 1, size=dim)

    # Where to truncate the C_inf
    trunc = 50

    # Create the Sister calculation
    sis_var_array = []
    for M in range(1, 21):
        # first matrix is 2N^2 * /sum_{n=1}^{M} (A^(n-1))(A.T^(n-1))
        first_matrix = np.sum(
            [np.matmul(np.linalg.matrix_power(inheritance_matrix, j - 1), np.linalg.matrix_power(inheritance_matrix.T,
                                                                                                 j - 1)) for j in range(1, M + 1)], 0)

        # second matrix is 2(L^2+N^2) * sum_j=0^n-2{(A^j)(A.T^j)}
        second_matrix = np.sum(
            [np.sum([np.matmul(np.linalg.matrix_power(inheritance_matrix, j), np.linalg.matrix_power(inheritance_matrix.T,
                                                                                                     j)) for j in range(n - 1) if n > 1],
                    0) for n in
             range(1, M + 1)], 0)
        sis_var = np.dot(projection.T, np.dot((2 * (N ** 2) * first_matrix + 2 * (L ** 2 + N ** 2) * second_matrix),
                                              projection))
        sis_var_array.append(sis_var)

    # Create the Non-Sister calculation
    non_sis_var_array = []
    for M in range(1, 21):
        inside_matrix = np.sum([np.matmul(np.linalg.matrix_power(inheritance_matrix, j), np.linalg.matrix_power(
            inheritance_matrix.T, j)) for j in range(trunc)], 0)

        first_matrix = np.sum(
            [np.matmul(np.linalg.matrix_power(inheritance_matrix, j), np.matmul(inside_matrix, np.linalg.matrix_power(inheritance_matrix.T,
                                                                                                                      j))) for j in
             range(1, M + 1)],
            0)
        second_matrix = np.sum(
            [np.sum([np.matmul(np.linalg.matrix_power(inheritance_matrix, j), np.linalg.matrix_power(inheritance_matrix.T,
                                                                                                     j)) for j in range(n)], 0)
             for n in range(1, M + 1)], 0)
        non_sis_var = np.dot(projection.T, np.dot((2 * (N ** 2 + L ** 2) * first_matrix + 2 * (L ** 2 + N ** 2) * second_matrix),
                                                  projection))
        non_sis_var_array.append(non_sis_var)

    # Create the Control calculation
    both_var_array = []
    for M in range(1, 21):
        inside_matrix = np.sum([np.matmul(np.linalg.matrix_power(inheritance_matrix, j), np.linalg.matrix_power(
            inheritance_matrix.T, j)) for j in range(trunc)], 0)

        first_matrix = np.sum(
            [np.matmul(np.linalg.matrix_power(inheritance_matrix, j), np.matmul(inside_matrix, np.linalg.matrix_power(inheritance_matrix.T,
                                                                                                                      j))) for j in
             range(1, M + 1)], 0)
        second_matrix = np.sum(
            [np.sum([np.matmul(np.linalg.matrix_power(inheritance_matrix, j), np.linalg.matrix_power(inheritance_matrix.T,
                                                                                                     j)) for j in range(n)], 0)
             for n in range(1, M + 1)], 0)
        both_var = np.dot(projection.T, np.dot((2 * (N ** 2 + L ** 2 + E ** 2) * first_matrix + 2 * (L ** 2 + E ** 2 + N ** 2) * second_matrix),
                                               projection))
        both_var_array.append(both_var)

    # step by step through generations
    for gen in range(20):

        # # # For Sisters dset

        # Noises, _A is the first pair, _B is the second pair
        noise_E = np.random.multivariate_normal(mean, corr_cov, num_of_reps)  # always correlated
        noise_E_A = noise_E[:, :dim]
        noise_E_B = noise_E[:, dim:]
        noise_N = np.random.multivariate_normal(mean, np.identity(2 * dim), num_of_reps)  # always uncorrelated
        noise_N_A = noise_N[:, :dim]
        noise_N_B = noise_N[:, dim:]
        # The genetic correlation between two sister cells that share the same mother, ie. only the first generation
        if gen == 0:
            noise_L = np.random.multivariate_normal(mean, corr_cov, num_of_reps)  # correlated on first one
            noise_L_A = noise_L[:, :dim]
            noise_L_B = noise_L[:, dim:]
        else:
            noise_L = np.random.multivariate_normal(mean, np.identity(2 * dim), num_of_reps)  # uncorrelated after first
            noise_L_A = noise_L[:, :dim]
            noise_L_B = noise_L[:, dim:]

        # Calculating the next generation hidden state for each pair
        x_sis_A = np.array(
            [np.dot(inheritance_matrix, x_init_sis_A[rep]) + E * noise_E_A[rep] + L * noise_L_A[rep] + N * noise_N_A[rep] for rep in
             range(num_of_reps)])
        x_sis_B = np.array(
            [np.dot(inheritance_matrix, x_init_sis_B[rep]) + E * noise_E_B[rep] + L * noise_L_B[rep] + N * noise_N_B[rep] for rep in
             range(num_of_reps)])
        sis_A_array.append(x_sis_A)
        sis_B_array.append(x_sis_B)

        # # # For Non-Sisters dset

        # Noises, _A is the first pair, _B is the second pair
        noise_E = np.random.multivariate_normal(mean, corr_cov, num_of_reps)  # always correlated
        noise_E_A = noise_E[:, :dim]
        noise_E_B = noise_E[:, dim:]
        noise_N = np.random.multivariate_normal(mean, np.identity(2 * dim), num_of_reps)  # always uncorrelated
        noise_N_A = noise_N[:, :dim]
        noise_N_B = noise_N[:, dim:]
        noise_L = np.random.multivariate_normal(mean, np.identity(2 * dim), num_of_reps)  # always uncorrelated
        noise_L_A = noise_L[:, :dim]
        noise_L_B = noise_L[:, dim:]

        # Calculating the next generation hidden state for each pair
        x_non_sis_A = np.array(
            [np.dot(inheritance_matrix, x_init_non_sis_A[rep]) + E * noise_E_A[rep] + L * noise_L_A[rep] + N * noise_N_A[rep] for rep in
             range(num_of_reps)])
        x_non_sis_B = np.array(
            [np.dot(inheritance_matrix, x_init_non_sis_B[rep]) + E * noise_E_B[rep] + L * noise_L_B[rep] + N * noise_N_B[rep] for rep in
             range(num_of_reps)])
        non_sis_A_array.append(x_non_sis_A)
        non_sis_B_array.append(x_non_sis_B)

        # # # For Control dset

        # Noises, _A is the first pair, _B is the second pair
        noise_E = np.random.multivariate_normal(mean, np.identity(2 * dim), num_of_reps)  # always uncorrelated
        noise_E_A = noise_E[:, :dim]
        noise_E_B = noise_E[:, dim:]
        noise_N = np.random.multivariate_normal(mean, np.identity(2 * dim), num_of_reps)  # always uncorrelated
        noise_N_A = noise_N[:, :dim]
        noise_N_B = noise_N[:, dim:]
        noise_L = np.random.multivariate_normal(mean, np.identity(2 * dim), num_of_reps)  # always uncorrelated
        noise_L_A = noise_L[:, :dim]
        noise_L_B = noise_L[:, dim:]

        # Calculating the next generation hidden state for each pair
        x_both_A = np.array(
            [np.dot(inheritance_matrix, x_init_both_A[rep]) + E * noise_E_A[rep] + L * noise_L_A[rep] + N * noise_N_A[rep] for rep in
             range(num_of_reps)])
        x_both_B = np.array(
            [np.dot(inheritance_matrix, x_init_both_B[rep]) + E * noise_E_B[rep] + L * noise_L_B[rep] + N * noise_N_B[rep] for rep in
             range(num_of_reps)])
        both_A_array.append(x_both_A)
        both_B_array.append(x_both_B)

    # Project to the observable space
    sis_observables = np.array([np.cumsum([np.dot(projection, sis_A_array[gen][ind][:]) - np.dot(projection, sis_B_array[gen][ind][:]) for gen in
                                           range(20)]) for ind in range(num_of_reps)])
    non_sis_observables = np.array([np.cumsum([np.dot(projection, non_sis_A_array[gen][ind][:]) - np.dot(projection, non_sis_B_array[gen][ind][
                                                                                                                     :]) for gen in range(20)])
                                    for ind in range(num_of_reps)])
    both_observables = np.array([np.cumsum([np.dot(projection, both_A_array[gen][ind][:]) - np.dot(projection, both_B_array[gen][ind][:]) for gen
                                            in range(20)]) for ind in range(num_of_reps)])

    return sis_observables, non_sis_observables, both_observables, sis_var_array, non_sis_var_array, both_var_array, np.linalg.norm(
        inheritance_matrix), N, L, E


def main():

    density_number = 1000

    # # place to save the scatter plots
    # saved_data = []
    # for f in range(density_number):
    #
    #     # run the simulation and get the needed things
    #     sis_observables, non_sis_observables, both_observables, sis_var_array, non_sis_var_array, both_var_array, A_norm, N, L, E = simulation()
    #
    #     # log the observables
    #     sis_scaling_index_array = np.array([np.log(np.var([sis_observables[ind][gen] for ind in range(len(sis_observables))])) for gen in range(20)])
    #     non_sis_scaling_index_array = np.array([np.log(np.var([non_sis_observables[ind][gen] for ind in range(len(non_sis_observables))])) for gen in range(20)])
    #     both_scaling_index_array = np.array([np.log(np.var([both_observables[ind][gen] for ind in range(len(both_observables))])) for gen in range(20)])
    #
    #     # get the scaling index slope
    #     slope_sis, intercept_sis, r_value_sis, p_value_sis, std_err_sis = stats.linregress(np.log(np.arange(1, len(sis_scaling_index_array)+1)),
    #                                                                          sis_scaling_index_array)
    #     slope_non_sis, intercept_non_sis, r_value_non_sis, p_value_non_sis, std_err_non_sis = stats.linregress(np.log(np.arange(1, len(non_sis_scaling_index_array) + 1)),
    #                                                                                       non_sis_scaling_index_array)
    #     slope_both, intercept_both, r_value_both, p_value_both, std_err_both = stats.linregress(np.log(np.arange(1, len(both_scaling_index_array) + 1)),
    #                                                                                       both_scaling_index_array)
    #
    #     # save it to 3D-scatter later
    #     saved_data.append(np.array([A_norm, N, L, E, slope_sis, intercept_sis, slope_non_sis, intercept_non_sis, slope_both, intercept_both,
    #                                 std_err_sis, std_err_non_sis, std_err_both, sis_observables, non_sis_observables, both_observables,
    #                                 sis_var_array, non_sis_var_array, both_var_array, sis_scaling_index_array, non_sis_scaling_index_array,
    #                                 both_scaling_index_array]))
    #
    #     print(f)
    #
    # ###### PICKLE IT NOW!!!!! #######
    #
    # pickle_out = open("Model_matrix_vs_noise_params_1000.pickle", "wb")
    # pickle.dump(saved_data, pickle_out)
    # pickle_out.close()
    #
    # print('pickle saved!')


    # Import the Refined Data
    pickle_in = open("Model_matrix_vs_noise_params_1000.pickle", "rb")
    saved_data = pickle.load(pickle_in)
    pickle_in.close()

    # print(len(saved_data))
    #
    # for sim in range(density_number):
    #     plt.plot(np.log(np.arange(1, len(saved_data[sim][-1]) + 1)), saved_data[sim][-1], label='Control, $\gamma=${}'.format(saved_data[sim][8]), color='g', marker='.')
    #     plt.plot(np.log(np.arange(1, len(saved_data[sim][-1]) + 1)), saved_data[sim][9]+np.log(np.arange(1, len(saved_data[sim][-1]) + 1))*saved_data[sim][8],
    #              label='Control (fit)', color='g', alpha=.5, linewidth='3')
    #     plt.plot(np.log(np.arange(1, len(saved_data[sim][-3]) + 1)), saved_data[sim][-3], label='Sister, $\gamma=${}'.format(saved_data[sim][4]),
    #              color='blue', marker='.')
    #     plt.plot(np.log(np.arange(1, len(saved_data[sim][-3]) + 1)), saved_data[sim][5] + np.log(np.arange(1, len(saved_data[sim][-3]) + 1)) * saved_data[sim][4],
    #              label='Sister (fit)', color='blue', alpha=.5, linewidth='3')
    #     plt.plot(np.log(np.arange(1, len(saved_data[sim][-2]) + 1)), saved_data[sim][-2], label='Non-Sister, $\gamma=${}'.format(saved_data[sim][6]), color='orange',
    #              marker='.')
    #     plt.plot(np.log(np.arange(1, len(saved_data[sim][-2]) + 1)), saved_data[sim][7] + np.log(np.arange(1, len(saved_data[sim][-2]) + 1)) * saved_data[sim][6],
    #              label='Non-Sister (fit)', color='orange', alpha=.5, linewidth='3')
    #     plt.legend()
    #     plt.show()

    A_norm = [saved_data[ind][0] for ind in range(density_number)]
    sum_norm = [saved_data[ind][1] ** 2 + saved_data[ind][2] ** 2 + saved_data[ind][3] ** 2 for ind in range(density_number)]
    sis_scaling_index = [saved_data[ind][4] for ind in range(density_number)]
    non_sis_scaling_index = [saved_data[ind][6] for ind in range(density_number)]
    both_scaling_index = [saved_data[ind][8] for ind in range(density_number)]

    # for sis
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(A_norm, sum_norm, sis_scaling_index, label='Sisters')
    ax.set_xlabel(r'$||A||$')
    ax.set_ylabel(r'$N^2+L^2+E^2$')
    ax.set_zlabel(r'$\gamma$ scaling index')
    plt.legend()
    plt.title('Norm, vs quadratic sum vs scaling index')
    plt.show()
    plt.close(fig)
    # for nonsis
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(A_norm, sum_norm, non_sis_scaling_index, label='Non-Sisters')
    ax.set_xlabel(r'$||A||$')
    ax.set_ylabel(r'$N^2+L^2+E^2$')
    ax.set_zlabel(r'$\gamma$ scaling index')
    plt.legend()
    plt.title('Norm, vs quadratic sum vs scaling index')
    plt.show()
    plt.close(fig)
    # for control
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(A_norm, sum_norm, both_scaling_index, label='Control')
    ax.set_xlabel(r'$||A||$')
    ax.set_ylabel(r'$N^2+L^2+E^2$')
    ax.set_zlabel(r'$\gamma$ scaling index')
    plt.legend()
    plt.title('Norm, vs quadratic sum vs scaling index')
    plt.show()
    plt.close(fig)




if __name__ == '__main__':
    main()
