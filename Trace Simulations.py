
from __future__ import print_function

import numpy as np

import matplotlib.pyplot as plt


def main():

    # How many times we should repeat the simulation for each dimension
    num_of_reps = 10000

    # Meta parameter to see how it changes with dimension of system
    for dim in range(2, 10):

        # Create the covariance matrix for pair-correlated R.V.
        corr_cov = np.identity(2 * dim)
        for ind1 in range(2 * dim):
            for ind2 in range(2 * dim):
                if ind1 == ind2 - dim or ind2 == ind1 - dim or ind1 == ind2:
                    corr_cov[ind1][ind2] = 1

        print('corr_cov:', corr_cov)

        # Mean of the multivariate Gaussian
        mean = np.zeros(2*dim)
        
        # Creating the inheritance matrix
        inheritance_matrix = np.random.rand(dim, dim)
        inheritance_matrix = (1 / (dim)) * (inheritance_matrix - np.ones_like(inheritance_matrix) * .5) * 2
        print(inheritance_matrix)
        print(np.linalg.norm(inheritance_matrix))

        # # Creating the inheritance matrix 1
        # inheritance_matrix1 = np.random.rand(dim, dim)
        # inheritance_matrix1 = (1 / (dim*5)) * (inheritance_matrix1 - np.ones_like(inheritance_matrix1) * .5) * 2
        # print(inheritance_matrix1)
        # print(np.linalg.norm(inheritance_matrix1))

        # Creating the different initial (mother(s)) internal states for the different dsets
        x_init_sis_A = np.random.multivariate_normal(np.zeros(dim), np.identity(dim), num_of_reps)
        x_init_sis_B = x_init_sis_A
        x_init_non_sis_A = np.random.multivariate_normal(np.zeros(dim), np.identity(dim), num_of_reps)
        x_init_non_sis_B = np.random.multivariate_normal(np.zeros(dim), np.identity(dim), num_of_reps)
        x_init_both_A = np.random.multivariate_normal(np.zeros(dim), np.identity(dim), num_of_reps)
        x_init_both_B = np.random.multivariate_normal(np.zeros(dim), np.identity(dim), num_of_reps)
        
        # Select the influence parameters E,L,N
        E = 5
        L = 5
        N = 5
        
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

        # step by step through generations
        for gen in range(20):

            # # # For Sisters dset
            
            # Noises, _A is the first pair, _B is the second pair
            noise_E = np.random.multivariate_normal(mean, corr_cov, num_of_reps) # always correlated
            noise_E_A = noise_E[:,:dim]
            noise_E_B = noise_E[:,dim:]
            noise_N = np.random.multivariate_normal(mean, np.identity(2*dim), num_of_reps) # always uncorrelated
            noise_N_A = noise_N[:,:dim]
            noise_N_B = noise_N[:,dim:]
            # The genetic correlation between two sister cells that share the same mother, ie. only the first generation
            if gen == 0:
                noise_L = np.random.multivariate_normal(mean, corr_cov, num_of_reps) # correlated on first one
                noise_L_A = noise_L[:,:dim]
                noise_L_B = noise_L[:,dim:]
            else:
                noise_L = np.random.multivariate_normal(mean, np.identity(2*dim), num_of_reps) # uncorrelated after first
                noise_L_A = noise_L[:,:dim]
                noise_L_B = noise_L[:,dim:]
            
            # Calculating the next generation hidden state for each pair
            x_sis_A = np.array([np.dot(inheritance_matrix, x_init_sis_A[rep]) + E*noise_E_A[rep] + L*noise_L_A[rep] + N*noise_N_A[rep] for rep in 
                                range(num_of_reps)])
            x_sis_B = np.array(
                [np.dot(inheritance_matrix, x_init_sis_B[rep]) + E * noise_E_B[rep] + L * noise_L_B[rep] + N * noise_N_B[rep] for rep in
                 range(num_of_reps)])
            sis_A_array.append(x_sis_A)
            sis_B_array.append(x_sis_B)

            # # # For Non-Sisters dset

            # Noises, _A is the first pair, _B is the second pair
            noise_E = np.random.multivariate_normal(mean, corr_cov, num_of_reps)  # always correlated
            noise_E_A = noise_E[:,:dim]
            noise_E_B = noise_E[:,dim:]
            noise_N = np.random.multivariate_normal(mean, np.identity(2 * dim), num_of_reps)  # always uncorrelated
            noise_N_A = noise_N[:,:dim]
            noise_N_B = noise_N[:,dim:]
            noise_L = np.random.multivariate_normal(mean, np.identity(2 * dim), num_of_reps)  # always uncorrelated
            noise_L_A = noise_L[:,:dim]
            noise_L_B = noise_L[:,dim:]

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
            noise_E_A = noise_E[:,:dim]
            noise_E_B = noise_E[:,dim:]
            noise_N = np.random.multivariate_normal(mean, np.identity(2 * dim), num_of_reps)  # always uncorrelated
            noise_N_A = noise_N[:,:dim]
            noise_N_B = noise_N[:,dim:]
            noise_L = np.random.multivariate_normal(mean, np.identity(2 * dim), num_of_reps)  # always uncorrelated
            noise_L_A = noise_L[:,:dim]
            noise_L_B = noise_L[:,dim:]

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
        sis_observables = np.array([np.cumsum([np.dot(projection, sis_A_array[gen][ind][:])-np.dot(projection, sis_B_array[gen][ind][:]) for gen in
                                               range(20)]) for ind in range(num_of_reps)])
        non_sis_observables = np.array([np.cumsum([np.dot(projection, non_sis_A_array[gen][ind][:])-np.dot(projection, non_sis_B_array[gen][ind][
                                                    :]) for gen in range(20)]) for ind in range(num_of_reps)])
        both_observables = np.array([np.cumsum([np.dot(projection, both_A_array[gen][ind][:])-np.dot(projection, both_B_array[gen][ind][:]) for gen
                                                in range(20)]) for ind in range(num_of_reps)])


        # # # These are the theoretical predictions

        # The point at which we truncate the C_inf matrix, ie. the covariance matrix of the difference of the mothers
        trunc = 500

        # Create the Sister calculation
        sis_var_array = []
        U_array = []
        V_array = []
        for M in range(1,21):
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
            V_array.append(np.dot(projection.T, np.dot(first_matrix, projection)))
            U_array.append(np.dot(projection.T, np.dot(second_matrix, projection)))

        # Create the Non-Sister calculation
        non_sis_var_array = []
        K_array = []
        L_array = []
        for M in range(1,21):
            inside_matrix = np.sum([np.matmul(np.linalg.matrix_power(inheritance_matrix, j), np.linalg.matrix_power(
                inheritance_matrix.T, j)) for j in range(trunc)], 0)

            first_matrix = np.sum(
                [np.matmul(np.linalg.matrix_power(inheritance_matrix, j), np.matmul(inside_matrix, np.linalg.matrix_power(inheritance_matrix.T,
                                                                                                                    j))) for j in range(1, M + 1)], 0)
            second_matrix = np.sum(
                [np.sum([np.matmul(np.linalg.matrix_power(inheritance_matrix, j), np.linalg.matrix_power(inheritance_matrix.T,
                                                                                                         j)) for j in range(n)], 0)
                 for n in range(1, M + 1)], 0)
            non_sis_var = np.dot(projection.T, np.dot((2*(N**2+L**2)*first_matrix + 2 * (L ** 2 + N ** 2) * second_matrix),
                                                         projection))
            non_sis_var_array.append(non_sis_var)
            K_array.append(np.dot(projection.T, np.dot(first_matrix, projection)))
            L_array.append(np.dot(projection.T, np.dot(second_matrix, projection)))

        # Create the Control calculation
        both_var_array = []
        for M in range(1,21):
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

        # Trying to get the correct N, L, E for everything else given... but it's not working
        D_sis = 0.019
        D_non_sis = 0.026
        D_both = 0.156 # the one that I got with my random dataset
        b = np.array([(D_sis)/2, (D_non_sis)/(2*np.array(np.array(K_array)+np.array(L_array))[0]), (D_both)/(2*np.array(np.array(
            K_array)+np.array(L_array))[0])])
        a = np.array([[np.array(np.array(V_array)+np.array(U_array))[0], U_array[0], 0], [1, 1, 0], [1, 1, 1]])
        x = np.linalg.solve(a, b) # N^2, L^2, E^2
        print('N^2, L^2, E^2', x)

        sis_var = [np.var([sis_observables[ind][gen] for ind in range(num_of_reps)]) for gen in range(20)]
        non_sis_var = [np.var([non_sis_observables[ind][gen] for ind in range(num_of_reps)]) for gen in range(20)]
        both_var = [np.var([both_observables[ind][gen] for ind in range(num_of_reps)]) for gen in range(20)]
        plt.plot(sis_var, label='sis', marker='.')
        plt.plot(non_sis_var, label='NS', marker='.')
        plt.plot(both_var, label='control', marker='.')
        plt.plot(sis_var_array, label='sis - Theoretic', marker='.', linestyle='--', alpha=.5, color='#1f77b4')
        plt.plot(non_sis_var_array, label='NS - Theoretic', marker='.', linestyle='--', alpha=.5, color='#ff7f0e')
        plt.plot(both_var_array, label='control - Theoretic', marker='.', linestyle='--', alpha=.5, color='#2ca02c')
        plt.legend()
        plt.xlabel('generation')
        plt.ylabel('variance of projected hidden state')
        plt.show()
        # plt.savefig(str(dim) + ' dims. Theory vs. Simulation Comparison of Acc. Var.', dpi=300)
        # plt.close()

        # print(U_array)
        # print(V_array)
        # print(K_array)
        # print(L_array)
        # plt.plot(U_array, label='U')
        # plt.plot(np.array(V_array)+np.array(U_array), label='U+V')
        # plt.plot(np.array(K_array)+np.array(L_array), label='K+L')
        # plt.legend()
        # plt.xlabel('generation')
        # plt.ylabel('how much does it depend on the generation')
        # plt.show()


if __name__ == '__main__':
    main()
