
from __future__ import print_function

import numpy as np

import matplotlib.pyplot as plt

import pickle

import scipy.stats as stats


def main():

    # Import the Refined Data
    pickle_in = open("metastructdata.pickle", "rb")
    struct = pickle.load(pickle_in)
    pickle_in.close()

    # parameters
    # alpha_weights = np.array([.5, .3])
    # inheritance_matrix = np.array([[.4, -.6], [0, .5]])

    alpha_change = 1
    param_change = 6
    matrix_change = 1

    # How many dimensions it should have
    for dimen in range(4,5):
        print('dimension:', dimen)

        alpha_weights = (np.random.rand(dimen) - np.ones(dimen) * .5) * 2
        print('alphas:', alpha_weights)

        # Create the inheritance matrix... Should we keep an eye out for the norm and how that affects it?
        inheritance_matrix = np.random.rand(dimen, dimen)
        inheritance_matrix = (1 / (dimen-1)) * (inheritance_matrix - np.ones_like(inheritance_matrix) * .5) * 2
        print('inhertiance matrix:', inheritance_matrix)
        print('norm inhertiance matrix:', np.linalg.norm(inheritance_matrix))

        # Approximation of the initial covariance matrix over many experiments, made into a number... should the A's stay the same?
        y = (np.random.rand(2) - .5 * np.ones(2)) * 2
        initial_sigma_non_sis = y[0]
        initial_sigma_both = y[1]

        # how many times to change L, N, M parameters ----- Should I choose these as a degree of freedom each? A New for loop each...
        for new_param_set in range(param_change):
            # E^2+L^2+N^2 = 1
            # np.random.seed(6)
            x = (np.random.rand(3) - .5 * np.ones(3)) * 2
            E = x[0]
            L = x[1]
            N = x[2]
            print('E, L, N:', x)

            # # Approximation of the initial covariance matrix over many experiments, made into a number... should the A's stay the same?
            # y = (np.random.rand(2) - .5 * np.ones(2)) * 2
            # initial_sigma_non_sis = y[0]
            # initial_sigma_both = y[1]

            # how many times to change alphas
            for new_alpha_set in range(alpha_change):
                # alpha_weights = (np.random.rand(dimen)-np.ones(dimen)*.5)*2
                # print('alphas:', alpha_weights)

                # How many times to change the Inheritance Matrix A
                for rep in range(matrix_change):

                    # # Create the inheritance matrix... Should we keep an eye out for the norm and how that affects it?
                    # inheritance_matrix = np.random.rand(dimen, dimen)
                    # inheritance_matrix = (1/dimen)*(inheritance_matrix-np.ones_like(inheritance_matrix)*.5)*2
                    # print('inhertiance matrix:', inheritance_matrix)
                    # print('norm inhertiance matrix:', np.linalg.norm(inheritance_matrix))

                    # Create the Sister simulation
                    sis_var_array = []
                    for M in range(20):
                        # first matrix is 2N^2 * (A^(n-1))(A.T^(n-1))
                        first_matrix = np.sum(
                            [np.matmul(np.linalg.matrix_power(inheritance_matrix, j - 1), np.linalg.matrix_power(inheritance_matrix.T,
                                                                                                                 j - 1)) for j in range(1, M + 1)], 0)

                        # second matrix is 2(L^2+N^2) * sum_j=0^n-2{(A^j)(A.T^j)}
                        second_matrix = np.sum(
                            [np.sum([np.matmul(np.linalg.matrix_power(inheritance_matrix, j), np.linalg.matrix_power(inheritance_matrix.T,
                                                                                    j)) for j in range(n - 1)], 0) for n in range(1, M+1)], 0)
                        sis_var = np.dot(alpha_weights.T, np.dot((2 * (N ** 2) * first_matrix + 2 * (L ** 2 + N ** 2) * second_matrix),
                                                                 alpha_weights))
                        sis_var_array.append(sis_var)

                    # Create the Non-Sister simulation
                    non_sis_initial_var = initial_sigma_non_sis
                    non_sis_var_array = []
                    for M in range(20):
                        first_matrix = np.sum(
                            [np.matmul(np.linalg.matrix_power(inheritance_matrix, j), np.linalg.matrix_power(inheritance_matrix.T, j)) for j in
                             range(1, M + 1)], 0)
                        second_matrix = np.sum(
                            [np.sum([np.matmul(np.linalg.matrix_power(inheritance_matrix, j), np.linalg.matrix_power(inheritance_matrix.T,
                                                                                                                     j)) for j in range(n)], 0)
                             for n in range(1, M + 1)], 0)
                        non_sis_var = np.dot(alpha_weights.T, np.dot((non_sis_initial_var * first_matrix + 2 * (L ** 2 + N ** 2) * second_matrix),
                                             alpha_weights))
                        non_sis_var_array.append(non_sis_var)

                    # Create the Control simulation
                    both_initial_var = initial_sigma_both
                    both_var_array = []
                    for M in range(20):
                        first_matrix = np.sum(
                            [np.matmul(np.linalg.matrix_power(inheritance_matrix, j), np.linalg.matrix_power(inheritance_matrix.T, j)) for j in
                             range(1, M + 1)], 0)
                        second_matrix = np.sum(
                            [np.sum([np.matmul(np.linalg.matrix_power(inheritance_matrix, j), np.linalg.matrix_power(inheritance_matrix.T,
                                                                                                                     j)) for j in range(n)], 0)
                             for n in range(1, M + 1)], 0)
                        both_var = np.dot(alpha_weights.T,  np.dot((both_initial_var * first_matrix + 2 * (L ** 2 + E**2 + N ** 2) * second_matrix),
                                                                    alpha_weights))
                        both_var_array.append(both_var)

                    # Plot them
                    plt.plot(sis_var_array, marker='.', color='blue', label='Sisters')
                    plt.plot(non_sis_var_array, marker='.', color='orange', label='Non-Sisters')
                    plt.plot(both_var_array, marker='.', color='green', label='Control')
                    plt.legend()
                    plt.xlabel('Generation')
                    plt.ylabel('Accumulated variance of Simulated GenerationTime')
                    plt.show()



    '''
    # E^2+L^2+N^2 = 1
    np.random.seed(6)
    x = np.random.rand(3)
    E = x[0]
    L = x[1]
    N = x[2]

    sis_var_array = []
    for M in range(20):
        first_matrix = np.sum([np.matmul(np.linalg.matrix_power(inheritance_matrix, j-1), np.linalg.matrix_power(inheritance_matrix.T, j-1)) for j in
                               range(1,M+1)], 0)
        second_matrix = np.sum([np.sum([np.matmul(np.linalg.matrix_power(inheritance_matrix, j), np.linalg.matrix_power(inheritance_matrix.T,
                                                                                        j)) for j in range(n-1)], 0) for n in range(1, M+1)], 0)
        sis_var = 2*(N**2)*np.dot(alpha_weights.T, np.dot(first_matrix, alpha_weights))+2*(L**2+N**2)*np.dot(alpha_weights.T, np.dot(second_matrix,
                                                                                                                                    alpha_weights))
        sis_var_array.append(sis_var)

    non_sis_initial_var = np.var([struct.A_dict_non_sis[IDA]['generationtime'].loc[0]-struct.B_dict_non_sis[IDB]['generationtime'].loc[0] for IDA,
                            IDB in zip(struct.A_dict_non_sis.keys(), struct.B_dict_non_sis.keys())])

    non_sis_var_array = []
    for M in range(20):
        first_matrix = np.sum([np.matmul(np.linalg.matrix_power(inheritance_matrix, j), np.linalg.matrix_power(inheritance_matrix.T, j)) for j in
                               range(1, M+1)], 0)
        second_matrix = np.sum([np.sum([np.matmul(np.linalg.matrix_power(inheritance_matrix, j), np.linalg.matrix_power(inheritance_matrix.T,
                                                                                                                        j)) for j in range(n)], 0)
                                for n in range(1, M+1)], 0)
        non_sis_var = non_sis_initial_var*first_matrix+2*(L**2+N**2)*np.dot(alpha_weights.T, np.dot(second_matrix,alpha_weights))
        non_sis_var_array.append(non_sis_var)

    print(non_sis_var)
    plt.plot(sis_var_array)
    # plt.plot(non_sis_var_array)
    plt.show()

    # maxgen is the number of generations we do this numerical simulation for
    maxgen=20
    for gen in range(maxgen):
        # The different noise terms
        if gen == 0:
            lineage = np.random.normal(loc=0, scale=1, size=2)
        else:
            lineage = np.random.normal(loc=0, scale=1, size=2)
        enviro = np.random.normal(loc=0, scale=1, size=2)
        noise = np.random.normal(loc=0, scale=1, size=2)

        # A_pair =
        # B_pair =

        A_array = []
        B_array = []
    '''


if __name__ == '__main__':
    main()
