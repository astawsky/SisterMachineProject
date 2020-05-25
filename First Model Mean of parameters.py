
from __future__ import print_function

import numpy as np

import matplotlib.pyplot as plt

import pickle

import scipy.stats as stats

def main():
    # Import the Data
    pickle_in = open("metastructdata.pickle", "rb")
    struct = pickle.load(pickle_in)
    pickle_in.close()


    # Where we will put all the means using just one generation across experiments
    sis_diff_array_mean_gen = []
    non_sis_diff_array_mean_gen = []
    both_diff_array_mean_gen = []

    # max_gen is the generation we want to look at, here we go up to 9 generations
    how_many = 20
    for max_gen in range(how_many):

        # Sister

        gen_time_array = np.mean([np.array(
            struct.A_dict_sis[keyA]['generationtime'].loc[max_gen] - struct.B_dict_sis[keyB]['generationtime'].loc[
                max_gen]) for keyA, keyB in
            zip(struct.A_dict_sis.keys(), struct.B_dict_sis.keys()) if
            min(len(struct.A_dict_sis[keyA]['generationtime']),
                len(struct.B_dict_sis[keyB]['generationtime'])) > max_gen])
        alpha_array = np.mean([np.array(
            struct.A_dict_sis[keyA]['growth_length'].loc[max_gen] - struct.B_dict_sis[keyB]['growth_length'].loc[
                max_gen]) for keyA, keyB in
            zip(struct.A_dict_sis.keys(), struct.B_dict_sis.keys()) if
            min(len(struct.A_dict_sis[keyA]['generationtime']),
                len(struct.B_dict_sis[keyB]['generationtime'])) > max_gen])
        phi_array = np.mean([np.array(
            struct.A_dict_sis[keyA]['generationtime'].loc[max_gen] * struct.A_dict_sis[keyA]['growth_length'].loc[
                max_gen] - struct.B_dict_sis[keyB][
                'generationtime'].loc[max_gen] *
            struct.B_dict_sis[keyB]['growth_length'].loc[max_gen]) for keyA, keyB in
            zip(struct.A_dict_sis.keys(), struct.B_dict_sis.keys()) if
            min(len(struct.A_dict_sis[keyA]['generationtime']),
                len(struct.B_dict_sis[keyB]['generationtime'])) > max_gen])
        div_ratios = np.mean([np.array(np.log(struct.A_dict_sis[keyA]['division_ratios__f_n'].loc[max_gen]) - np.log(
            struct.B_dict_sis[keyB]['division_ratios__f_n'].loc[max_gen])) for keyA, keyB in
                             zip(struct.A_dict_sis.keys(), struct.B_dict_sis.keys()) if
                             min(len(struct.A_dict_sis[keyA]['generationtime']),
                                 len(struct.B_dict_sis[keyB]['generationtime'])) > max_gen])
        birth_lengths = np.mean([np.array(
            np.log(struct.A_dict_sis[keyA]['length_birth'].loc[max_gen] / struct.x_avg) - np.log(
                struct.B_dict_sis[keyB]['length_birth'].loc[max_gen] / struct.x_avg)) for keyA, keyB in
            zip(struct.A_dict_sis.keys(), struct.B_dict_sis.keys()) if
            min(len(struct.A_dict_sis[keyA]['generationtime']),
                len(struct.B_dict_sis[keyB]['generationtime'])) > max_gen])

        sis_diff_array = [gen_time_array, alpha_array, phi_array, div_ratios, birth_lengths]
        sis_diff_array_mean_gen.append(np.array(sis_diff_array))

        # Non-Sister
        gen_time_array = np.mean([np.array(
            struct.A_dict_non_sis[keyA]['generationtime'].loc[max_gen] - struct.B_dict_non_sis[keyB][
                'generationtime'].loc[
                max_gen]) for keyA, keyB in
            zip(struct.A_dict_non_sis.keys(), struct.B_dict_non_sis.keys()) if
            min(len(struct.A_dict_non_sis[keyA]['generationtime']),
                len(struct.B_dict_non_sis[keyB]['generationtime'])) > max_gen])
        alpha_array = np.mean([np.array(
            struct.A_dict_non_sis[keyA]['growth_length'].loc[max_gen] - struct.B_dict_non_sis[keyB][
                'growth_length'].loc[
                max_gen]) for keyA, keyB in
            zip(struct.A_dict_non_sis.keys(), struct.B_dict_non_sis.keys()) if
            min(len(struct.A_dict_non_sis[keyA]['generationtime']),
                len(struct.B_dict_non_sis[keyB]['generationtime'])) > max_gen])
        phi_array = np.mean([np.array(
            struct.A_dict_non_sis[keyA]['generationtime'].loc[max_gen] * struct.A_dict_non_sis[keyA][
                'growth_length'].loc[
                max_gen] - struct.B_dict_non_sis[keyB][
                'generationtime'].loc[
                max_gen] *
            struct.B_dict_non_sis[keyB]['growth_length'].loc[max_gen]) for keyA, keyB in
            zip(struct.A_dict_non_sis.keys(), struct.B_dict_non_sis.keys()) if
            min(len(struct.A_dict_non_sis[keyA]['generationtime']),
                len(struct.B_dict_non_sis[keyB]['generationtime'])) > max_gen])
        div_ratios = np.mean([np.array(
            np.log(struct.A_dict_non_sis[keyA]['division_ratios__f_n'].loc[max_gen]) - np.log(
                struct.B_dict_non_sis[keyB]['division_ratios__f_n'].loc[max_gen])) for keyA, keyB in
            zip(struct.A_dict_non_sis.keys(), struct.B_dict_non_sis.keys()) if
            min(len(struct.A_dict_non_sis[keyA]['generationtime']),
                len(struct.B_dict_non_sis[keyB]['generationtime'])) > max_gen])
        birth_lengths = np.mean([np.array(
            np.log(struct.A_dict_non_sis[keyA]['length_birth'].loc[max_gen] / struct.x_avg) - np.log(
                struct.B_dict_non_sis[keyB]['length_birth'].loc[max_gen] / struct.x_avg)) for keyA, keyB in
            zip(struct.A_dict_non_sis.keys(), struct.B_dict_non_sis.keys()) if
            min(len(struct.A_dict_non_sis[keyA]['generationtime']),
                len(struct.B_dict_non_sis[keyB]['generationtime'])) > max_gen])

        non_sis_diff_array = [gen_time_array, alpha_array, phi_array, div_ratios, birth_lengths]
        non_sis_diff_array_mean_gen.append(np.array(non_sis_diff_array))

        # Control
        gen_time_array = np.mean([np.array(
            struct.A_dict_both[keyA]['generationtime'].loc[max_gen] - struct.B_dict_both[keyB]['generationtime'].loc[
                max_gen]) for keyA, keyB in
            zip(struct.A_dict_both.keys(), struct.B_dict_both.keys()) if
            min(len(struct.A_dict_both[keyA]['generationtime']),
                len(struct.B_dict_both[keyB]['generationtime'])) > max_gen])
        alpha_array = np.mean([np.array(
            struct.A_dict_both[keyA]['growth_length'].loc[max_gen] - struct.B_dict_both[keyB]['growth_length'].loc[
                max_gen]) for keyA, keyB in
            zip(struct.A_dict_both.keys(), struct.B_dict_both.keys()) if
            min(len(struct.A_dict_both[keyA]['generationtime']),
                len(struct.B_dict_both[keyB]['generationtime'])) > max_gen])
        phi_array = np.mean([np.array(
            struct.A_dict_both[keyA]['generationtime'].loc[max_gen] * struct.A_dict_both[keyA]['growth_length'].loc[
                max_gen] - struct.B_dict_both[keyB][
                'generationtime'].loc[max_gen] *
            struct.B_dict_both[keyB]['growth_length'].loc[max_gen]) for keyA, keyB in
            zip(struct.A_dict_both.keys(), struct.B_dict_both.keys()) if
            min(len(struct.A_dict_both[keyA]['generationtime']),
                len(struct.B_dict_both[keyB]['generationtime'])) > max_gen])
        div_ratios = np.mean([np.array(
            np.log(struct.A_dict_both[keyA]['division_ratios__f_n'].loc[max_gen]) - np.log(
                struct.B_dict_both[keyB]['division_ratios__f_n'].loc[max_gen])) for keyA, keyB in
            zip(struct.A_dict_both.keys(), struct.B_dict_both.keys()) if
            min(len(struct.A_dict_both[keyA]['generationtime']),
                len(struct.B_dict_both[keyB]['generationtime'])) > max_gen])
        birth_lengths = np.mean([np.array(
            np.log(struct.A_dict_both[keyA]['length_birth'].loc[max_gen] / struct.x_avg) - np.log(
                struct.B_dict_both[keyB]['length_birth'].loc[max_gen] / struct.x_avg)) for keyA, keyB in
            zip(struct.A_dict_both.keys(), struct.B_dict_both.keys()) if
            min(len(struct.A_dict_both[keyA]['generationtime']),
                len(struct.B_dict_both[keyB]['generationtime'])) > max_gen])

        both_diff_array = [gen_time_array, alpha_array, phi_array, div_ratios, birth_lengths]
        both_diff_array_mean_gen.append(np.array(both_diff_array))

    # first index is what generation, second index is what cycle parameter to use
    noise_mean = [np.array([both_diff_array_mean_gen[ind][param_num] for ind in range(max_gen)]) for param_num in range(5)]
    environment_mean = [np.array([non_sis_diff_array_mean_gen[ind][param_num]-both_diff_array_mean_gen[ind][param_num] for ind in range(max_gen)]) for param_num in
                      range(5)]
    genetic_mean = [np.array([sis_diff_array_mean_gen[ind][param_num] - non_sis_diff_array_mean_gen[ind][param_num] for ind in range(max_gen)]) for param_num in range(5)]


    # Name of the cycle parameters for labels
    param_array = [r'Generation time ($T$)', r'Growth rate ($\alpha$)', r'Accumulation Exponent ($\phi$)', r'Log of Division Ratio ($\log(f)$)', r'Normalized Birth Size ($\log(\frac{x}{x^*})$)']

    for param_num in range(5):
        plt.plot(range(len(noise_mean[param_num])), noise_mean[param_num], label=r'Noise mean, $\mu=$'+str(np.mean(noise_mean[param_num])), c='green')
        plt.plot(range(len(environment_mean[param_num])), environment_mean[param_num], label=r'Environment mean, $\mu=$'+str(np.mean(environment_mean[param_num])), c='orange')
        plt.plot(range(len(genetic_mean[param_num])), genetic_mean[param_num], label=r'Genetic mean, $\mu=$'+str(np.mean(genetic_mean[param_num])), c='blue')
        plt.axhline(y=np.mean(noise_mean[param_num]), xmin=0, xmax=max_gen-1, c='green', linestyle=':')
        plt.axhline(y=np.mean(environment_mean[param_num]), xmin=0, xmax=max_gen - 1, c='orange', linestyle=':')
        plt.axhline(y=np.mean(genetic_mean[param_num]), xmin=0, xmax=max_gen - 1, c='blue', linestyle=':')
        plt.title(param_array[param_num])
        plt.xlabel('generation')
        plt.legend()
        # plt.show()
        plt.savefig(param_array[param_num]+' mean and average mean of noise, environment and genetic influence', dpi=300)
        plt.close()


if __name__ == '__main__':
    main()