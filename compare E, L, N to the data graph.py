
from __future__ import print_function

import numpy as np

import matplotlib.pyplot as plt

import pickle

import scipy.stats as stats


def simulation():
    # How many times we should repeat the simulation for each dimension
    num_of_reps = 10000

    sis_observables_array = []
    non_sis_observables_array = []
    both_observables_array = []
    sis_theory_array = []
    non_sis_theory_array = []
    both_theory_array = []

    # Meta parameter to see how it changes with dimension of system
    for dim in range(2, 6):

        # Create the covariance matrix for pair-correlated R.V.
        corr_cov = np.identity(2 * dim)
        for ind1 in range(2 * dim):
            for ind2 in range(2 * dim):
                if ind1 == ind2 - dim or ind2 == ind1 - dim or ind1 == ind2:
                    corr_cov[ind1][ind2] = 1

        # Mean of the multivariate Gaussian
        mean = np.zeros(2 * dim)

        # # Creating the inheritance matrix
        # inheritance_matrix = np.random.rand(dim, dim)
        # inheritance_matrix = (inheritance_matrix - np.ones_like(inheritance_matrix) * .5) * 2
        # print(np.linalg.norm(inheritance_matrix))
        # inheritance_matrix = (1 / np.linalg.norm(inheritance_matrix)) * inheritance_matrix
        # print(inheritance_matrix)
        # print(np.linalg.norm(inheritance_matrix))

        # Creating the inheritance matrix 
        inheritance_matrix = np.random.rand(dim, dim)
        inheritance_matrix = (1 / (dim*5)) * (inheritance_matrix - np.ones_like(inheritance_matrix) * .5) * 2
        print(inheritance_matrix)
        print(np.linalg.norm(inheritance_matrix))

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

        # The point at which we truncate the C_inf matrix, ie. the covariance matrix of the difference of the mothers
        trunc = 500

        # Create the Sister calculation
        U_array = []
        V_array = []
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
            V_array.append(np.dot(projection.T, np.dot(first_matrix, projection)))
            U_array.append(np.dot(projection.T, np.dot(second_matrix, projection)))

        # Create the Non-Sister calculation
        K_array = []
        L_array = []
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
            K_array.append(np.dot(projection.T, np.dot(first_matrix, projection)))
            L_array.append(np.dot(projection.T, np.dot(second_matrix, projection)))

        # Trying to get the correct N, L, E for everything else given... but it's not working
        D_sis = 0.0173
        D_non_sis = 0.0252
        D_both = 0.126  # the one that I got with my random dataset
        point_of_eval = 1
        b = np.array([(D_sis*(point_of_eval+1)) / 2, (D_non_sis*(point_of_eval+1)) / (2 * np.array(np.array(K_array) + np.array(L_array))[point_of_eval]),
                      (D_both*(point_of_eval+1)) / (2 * np.array(np.array(K_array) + np.array(L_array))[point_of_eval])])
        a = np.array([[np.array(np.array(V_array) + np.array(U_array))[point_of_eval], U_array[point_of_eval], 0], [1, 1, 0], [1, 1, 1]])
        x = np.linalg.solve(a, b)  # N^2, L^2, E^2
        
        N = np.sqrt(x)[0]
        L = np.sqrt(x)[1]
        E = np.sqrt(x)[2]*100
        print('N, L, E', np.sqrt(x))

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
        
        sis_observables_array.append(sis_observables)
        non_sis_observables_array.append(non_sis_observables)
        both_observables_array.append(both_observables)
        sis_theory_array.append(sis_var_array)
        non_sis_theory_array.append(non_sis_var_array)
        both_theory_array.append(both_var_array)


    return sis_observables_array, non_sis_observables_array, both_observables_array, sis_theory_array, non_sis_theory_array, both_theory_array


def dataResults():
    # Import the Refined Data
    pickle_in = open("metastructdata.pickle", "rb")
    struct = pickle.load(pickle_in)
    pickle_in.close()

    # Where we will put all the variances for up to and then just generation
    sis_diff_array_var = []
    non_sis_diff_array_var = []
    both_diff_array_var = []

    # max_gen is the generation we want to look at, here we go up to 9 generations
    how_many = 20
    for max_gen in range(how_many):
        # # # NOW WE DO IT FOR ALL GENERATIONS UP TO "max_gen"

        # # Format the cycle parameters like in Lee's paper, and in the POWERPOINT
        # Sister

        gen_time_array = np.var(
            [np.sum(struct.A_dict_sis[keyA]['generationtime'].loc[:max_gen] - struct.B_dict_sis[keyB]['generationtime'].loc[:max_gen]) for keyA, keyB
             in zip(struct.A_dict_sis.keys(), struct.B_dict_sis.keys()) if
             min(len(struct.A_dict_sis[keyA]['generationtime']), len(struct.B_dict_sis[keyB]['generationtime'])) > max_gen])
        alpha_array = np.var(
            [np.sum(struct.A_dict_sis[keyA]['growth_length'].loc[:max_gen] - struct.B_dict_sis[keyB]['growth_length'].loc[:max_gen]) for keyA, keyB in
             zip(struct.A_dict_sis.keys(), struct.B_dict_sis.keys()) if
             min(len(struct.A_dict_sis[keyA]['generationtime']), len(struct.B_dict_sis[keyB]['generationtime'])) > max_gen])
        phi_array = np.var([np.sum(struct.A_dict_sis[keyA]['generationtime'].loc[:max_gen] * struct.A_dict_sis[keyA]['growth_length'].loc[:max_gen] -
                                   struct.B_dict_sis[keyB]['generationtime'].loc[:max_gen] * struct.B_dict_sis[keyB]['growth_length'].loc[:max_gen])
                            for keyA, keyB in zip(struct.A_dict_sis.keys(), struct.B_dict_sis.keys()) if
                            min(len(struct.A_dict_sis[keyA]['generationtime']), len(struct.B_dict_sis[keyB]['generationtime'])) > max_gen])
        div_ratios = np.var([np.sum(np.log(struct.A_dict_sis[keyA]['division_ratios__f_n'].loc[:max_gen]) - np.log(
            struct.B_dict_sis[keyB]['division_ratios__f_n'].loc[:max_gen])) for keyA, keyB in zip(struct.A_dict_sis.keys(), struct.B_dict_sis.keys())
                             if min(len(struct.A_dict_sis[keyA]['generationtime']), len(struct.B_dict_sis[keyB]['generationtime'])) > max_gen])
        birth_lengths = np.var([np.sum(np.log(struct.A_dict_sis[keyA]['length_birth'].loc[:max_gen] / struct.x_avg) - np.log(
            struct.B_dict_sis[keyB]['length_birth'].loc[:max_gen] / struct.x_avg)) for keyA, keyB in
                                zip(struct.A_dict_sis.keys(), struct.B_dict_sis.keys()) if
                                min(len(struct.A_dict_sis[keyA]['generationtime']), len(struct.B_dict_sis[keyB]['generationtime'])) > max_gen])

        sis_diff_array = [gen_time_array, alpha_array, phi_array, div_ratios, birth_lengths]
        sis_diff_array_var.append(np.array(sis_diff_array))

        # Non-Sister
        gen_time_array = np.var([np.sum(
            struct.A_dict_non_sis[keyA]['generationtime'].loc[:max_gen] - struct.B_dict_non_sis[keyB]['generationtime'].loc[
                                                                          :max_gen]) for keyA, keyB in
            zip(struct.A_dict_non_sis.keys(), struct.B_dict_non_sis.keys()) if
            min(len(struct.A_dict_non_sis[keyA]['generationtime']), len(struct.B_dict_non_sis[keyB]['generationtime'])) > max_gen])
        alpha_array = np.var([np.sum(
            struct.A_dict_non_sis[keyA]['growth_length'].loc[:max_gen] - struct.B_dict_non_sis[keyB]['growth_length'].loc[
                                                                         :max_gen]) for keyA, keyB in
            zip(struct.A_dict_non_sis.keys(), struct.B_dict_non_sis.keys()) if
            min(len(struct.A_dict_non_sis[keyA]['generationtime']), len(struct.B_dict_non_sis[keyB]['generationtime'])) > max_gen])
        phi_array = np.var([np.sum(
            struct.A_dict_non_sis[keyA]['generationtime'].loc[:max_gen] * struct.A_dict_non_sis[keyA]['growth_length'].loc[
                                                                          :max_gen] - struct.B_dict_non_sis[keyB][
                                                                                          'generationtime'].loc[:max_gen] *
            struct.B_dict_non_sis[keyB]['growth_length'].loc[:max_gen]) for keyA, keyB in
            zip(struct.A_dict_non_sis.keys(), struct.B_dict_non_sis.keys()) if
            min(len(struct.A_dict_non_sis[keyA]['generationtime']), len(struct.B_dict_non_sis[keyB]['generationtime'])) > max_gen])
        div_ratios = np.var([np.sum(
            np.log(struct.A_dict_non_sis[keyA]['division_ratios__f_n'].loc[:max_gen]) - np.log(
                struct.B_dict_non_sis[keyB]['division_ratios__f_n'].loc[:max_gen])) for keyA, keyB in
            zip(struct.A_dict_non_sis.keys(), struct.B_dict_non_sis.keys()) if
            min(len(struct.A_dict_non_sis[keyA]['generationtime']), len(struct.B_dict_non_sis[keyB]['generationtime'])) > max_gen])
        birth_lengths = np.var([np.sum(
            np.log(struct.A_dict_non_sis[keyA]['length_birth'].loc[:max_gen] / struct.x_avg) - np.log(
                struct.B_dict_non_sis[keyB]['length_birth'].loc[:max_gen] / struct.x_avg)) for keyA, keyB in
            zip(struct.A_dict_non_sis.keys(), struct.B_dict_non_sis.keys()) if
            min(len(struct.A_dict_non_sis[keyA]['generationtime']), len(struct.B_dict_non_sis[keyB]['generationtime'])) > max_gen])

        non_sis_diff_array = [gen_time_array, alpha_array, phi_array, div_ratios, birth_lengths]
        non_sis_diff_array_var.append(np.array(non_sis_diff_array))

        # Control
        gen_time_array = np.var([np.sum(
            struct.A_dict_both[keyA]['generationtime'].loc[:max_gen] - struct.B_dict_both[keyB]['generationtime'].loc[
                                                                       :max_gen]) for keyA, keyB in
            zip(struct.A_dict_both.keys(), struct.B_dict_both.keys()) if
            min(len(struct.A_dict_both[keyA]['generationtime']), len(struct.B_dict_both[keyB]['generationtime'])) > max_gen])
        alpha_array = np.var([np.sum(
            struct.A_dict_both[keyA]['growth_length'].loc[:max_gen] - struct.B_dict_both[keyB]['growth_length'].loc[
                                                                      :max_gen]) for keyA, keyB in
            zip(struct.A_dict_both.keys(), struct.B_dict_both.keys()) if
            min(len(struct.A_dict_both[keyA]['generationtime']), len(struct.B_dict_both[keyB]['generationtime'])) > max_gen])
        phi_array = np.var([np.sum(
            struct.A_dict_both[keyA]['generationtime'].loc[:max_gen] * struct.A_dict_both[keyA]['growth_length'].loc[
                                                                       :max_gen] - struct.B_dict_both[keyB][
                                                                                       'generationtime'].loc[:max_gen] *
            struct.B_dict_both[keyB]['growth_length'].loc[:max_gen]) for keyA, keyB in
            zip(struct.A_dict_both.keys(), struct.B_dict_both.keys()) if
            min(len(struct.A_dict_both[keyA]['generationtime']), len(struct.B_dict_both[keyB]['generationtime'])) > max_gen])
        div_ratios = np.var([np.sum(
            np.log(struct.A_dict_both[keyA]['division_ratios__f_n'].loc[:max_gen]) - np.log(
                struct.B_dict_both[keyB]['division_ratios__f_n'].loc[:max_gen])) for keyA, keyB in
            zip(struct.A_dict_both.keys(), struct.B_dict_both.keys()) if
            min(len(struct.A_dict_both[keyA]['generationtime']), len(struct.B_dict_both[keyB]['generationtime'])) > max_gen])
        birth_lengths = np.var([np.sum(
            np.log(struct.A_dict_both[keyA]['length_birth'].loc[:max_gen] / struct.x_avg) - np.log(
                struct.B_dict_both[keyB]['length_birth'].loc[:max_gen] / struct.x_avg)) for keyA, keyB in
            zip(struct.A_dict_both.keys(), struct.B_dict_both.keys()) if
            min(len(struct.A_dict_both[keyA]['generationtime']), len(struct.B_dict_both[keyB]['generationtime'])) > max_gen])

        both_diff_array = [gen_time_array, alpha_array, phi_array, div_ratios, birth_lengths]
        both_diff_array_var.append(np.array(both_diff_array))

    # Name of the cycle parameters for labels
    param_array = [r'Generation time ($T$)', r'Growth rate ($\alpha$)', r'Accumulation Exponent ($\phi$)', r'Log of Division Ratio ($\log(f)$)',
                   r'Normalized Birth Size ($\log(\frac{x}{x^*})$)']

    # separates up to and only options for data
    filename = [str(param_array[ind]) + ', ' + 'using up to said generation ' for ind in range(len(param_array))]
    title = filename

    return sis_diff_array_var, non_sis_diff_array_var, both_diff_array_var


def main():
    sis_observables_array, non_sis_observables_array, both_observables_array, sis_theory_array, non_sis_theory_array, both_theory_array = simulation()
    print('shape of simulation datas:', np.array(sis_observables_array).shape, np.array(non_sis_observables_array).shape,
          np.array(both_observables_array).shape)

    sis_diff_array_var, non_sis_diff_array_var, both_diff_array_var = dataResults()

    sis_diff_array_var = np.array([sis_diff_array_var[dex][0] for dex in range(len(sis_diff_array_var))])
    non_sis_diff_array_var = np.array([non_sis_diff_array_var[dex][0] for dex in range(len(non_sis_diff_array_var))])
    both_diff_array_var = np.array([both_diff_array_var[dex][0] for dex in range(len(both_diff_array_var))])

    # Create the labels for each data set
    label_sis = "Sister (Data)"  # GetLabels(sis_diff_array_var[ind], "Sister")
    label_non = "Non-Sister (Data)"  # GetLabels(non_sis_diff_array_var[ind], "Non-Sister")
    label_both = "Control (Data)"  # GetLabels(both_diff_array_var[ind], "Control")

    # Create the x-axis label
    xlabel = 'Generation number'
    ylabel = 'Accumulated variance of difference in generation time'

    sis_best_fit = np.array(
        [stats.linregress(x=range(len(sis_diff_array_var)), y=sis_diff_array_var)[0] * inp +
         stats.linregress(x=range(len(sis_diff_array_var)), y=sis_diff_array_var)[1] for inp in
         range(len(sis_diff_array_var))])
    non_best_fit = np.array([stats.linregress(x=range(len(non_sis_diff_array_var)), y=non_sis_diff_array_var)[0] * inp +
                             stats.linregress(x=range(len(non_sis_diff_array_var)), y=non_sis_diff_array_var)[1] for inp in
                             range(len(non_sis_diff_array_var))])
    both_best_fit = np.array([stats.linregress(x=range(len(both_diff_array_var)), y=both_diff_array_var)[0] * inp +
                              stats.linregress(x=range(len(both_diff_array_var)), y=both_diff_array_var)[1] for inp in
                              range(len(both_diff_array_var))])

    for dim in range(np.array(sis_observables_array).shape[0]):
        sis_var = [np.var([sis_observables_array[dim][ind][gen] for ind in range(10000)]) for gen in range(20)]
        non_sis_var = [np.var([non_sis_observables_array[dim][ind][gen] for ind in range(10000)]) for gen in range(20)]
        both_var = [np.var([both_observables_array[dim][ind][gen] for ind in range(10000)]) for gen in range(20)]

        x = np.arange(len(sis_diff_array_var))
        x = x[:, np.newaxis]
        a_sis, _, _, _ = np.linalg.lstsq(x, sis_diff_array_var - sis_diff_array_var[0], rcond=None)
        a_non, _, _, _ = np.linalg.lstsq(x, non_sis_diff_array_var - non_sis_diff_array_var[0], rcond=None)
        a_both, _, _, _ = np.linalg.lstsq(x, both_diff_array_var - both_diff_array_var[0], rcond=None)

        # plt.plot(sis_diff_array_var,
        #          label=label_sis + r' {:.2e}+{:.2e}*(gen. num.)'.format(stats.linregress(x=range(len(sis_diff_array_var)), y=sis_diff_array_var)[1], stats.linregress(
        #              x=range(len(sis_diff_array_var)), y=sis_diff_array_var)[0]), marker='.', color='b')
        # plt.plot(non_sis_diff_array_var,
        #          label=label_non + r' {:.2e}+{:.2e}*(gen. num.)'.format(stats.linregress(x=range(len(non_sis_diff_array_var)), y=non_sis_diff_array_var)[1], stats.linregress(
        #              x=range(len(non_sis_diff_array_var)), y=non_sis_diff_array_var)[0]), marker='.', color='orange')
        # plt.plot(both_diff_array_var, label=label_both + r' {:.2e}+{:.2e}*(gen. num.)'.format(stats.linregress(x=range(len(both_diff_array_var)), y=both_diff_array_var)[1],
        #                                                                             stats.linregress(x=range(len(both_diff_array_var)), y=both_diff_array_var)[0]),
        #          marker='.', color='green')
        # plt.plot(sis_best_fit, alpha=.5, linewidth='3')
        # plt.plot(non_best_fit, alpha=.5, linewidth='3')
        # plt.plot(both_best_fit, alpha=.5, linewidth='3')

        plt.plot(sis_var, label='sis (fitted), {:.2e}+{:.2e}*(gen. num.)'.format(sis_var[0], (sis_var[-1]-sis_var[0])/20), marker='v')
        plt.plot(non_sis_var, label='NS (fitted), {:.2e}+{:.2e}*(gen. num.)'.format(non_sis_var[0], (non_sis_var[-1]-non_sis_var[0])/20), marker='^')
        plt.plot(both_var, label='control (fitted), {:.2e}+{:.2e}*(gen. num.)'.format(both_var[0], (both_var[-1]-both_var[0])/20), marker='^')
        # plt.plot(sis_diff_array_var,
        #          label=label_sis + r' {:.2e}+{:.2e}*(gen. num.)'.format(sis_diff_array_var[0], a_sis[0]), marker='.', color='b')
        # plt.plot(non_sis_diff_array_var,
        #          label=label_non + r' {:.2e}+{:.2e}*(gen. num.)'.format(non_sis_diff_array_var[0], a_non[0]), marker='.', color='orange')
        # plt.plot(both_diff_array_var, label=label_both + r' {:.2e}+{:.2e}*(gen. num.)'.format(both_diff_array_var[0], a_both[0]),
        #          marker='.', color='green')
        # plt.plot(x, sis_diff_array_var[0] + a_sis*x, alpha=.5, linewidth='3')
        # plt.plot(x, non_sis_diff_array_var[0] + a_non*x, alpha=.5, linewidth='3')
        # plt.plot(x, both_diff_array_var[0] + a_both*x, alpha=.5, linewidth='3')
        # plt.plot(sis_theory_array[dim], alpha=.5, linewidth='3', linestyle='-.')
        # plt.plot(non_sis_theory_array[dim], alpha=.5, linewidth='3', linestyle='-.')
        # plt.plot(both_theory_array[dim], alpha=.5, linewidth='3', linestyle='-.')
        plt.legend()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(str(2+dim)+' dimensions')
        plt.show()
        # plt.savefig(filename + '.png', dpi=300)
        # plt.close()
    

if __name__ == "__main__":
    main()
