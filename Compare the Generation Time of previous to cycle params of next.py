
from __future__ import print_function

import numpy as np

import matplotlib.pyplot as plt

import pickle


def PearsonCorr(u, v):
    avg_u = np.mean(u)
    avg_v = np.mean(v)
    covariance = np.sum(np.array([(u[ID] - avg_u) * (v[ID] - avg_v) for ID in range(len(u))]))
    denominator = np.sqrt(np.sum(np.array([(u[ID] - avg_u) ** 2 for ID in range(len(u))]))) * \
                  np.sqrt(np.sum(np.array([(v[ID] - avg_v) ** 2 for ID in range(len(v))])))
    weighted_sum = covariance / denominator

    # np.corrcoef(np.array(u), np.array(v))[0, 1] --> Another way to calculate the pcorr coeff using numpy, gives similar answer

    return weighted_sum


def HistOfSlopes(dist_sis, dist_non, dist_both, label_sis, label_non, label_both, Nbins, abs_range, xlabel, title):
    arr_sis = plt.hist(x=dist_sis, label=label_sis, weights=np.ones_like(dist_sis) / float(len(dist_sis)), bins=Nbins,
                       range=abs_range)
    arr_non = plt.hist(x=dist_non, label=label_non, weights=np.ones_like(dist_non) / float(len(dist_non)), bins=Nbins,
                       range=abs_range)
    arr_both = plt.hist(x=dist_both, label=label_both, weights=np.ones_like(dist_both) / float(len(dist_both)),
                        bins=Nbins, range=abs_range)
    plt.close()

    # range=[0, abs_diff_range]

    plt.plot(np.array([(arr_sis[1][l] + arr_sis[1][l + 1]) / 2. for l in range(len(arr_sis[1]) - 1)]), arr_sis[0],
             label=label_sis, marker='.')
    plt.plot(np.array([(arr_non[1][l] + arr_non[1][l + 1]) / 2. for l in range(len(arr_non[1]) - 1)]), arr_non[0],
             label=label_non, marker='.')
    plt.plot(np.array([(arr_both[1][l] + arr_both[1][l + 1]) / 2. for l in range(len(arr_both[1]) - 1)]), arr_both[0],
             label=label_both, marker='.')
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel('PDF')
    plt.savefig(title+'.png', dpi=300)
    plt.close()


def GetLabels(array, dset):
    # calculate the means and variance for the array and puts it into a string
    label = dset + ' ' + r'$\mu=$' + '{:.2e}'.format(np.nanmean(array)) + r', $\sigma=$' + '{:.2e}'.format(np.nanstd(array))

    return label


def main():

    # Import the Refined Data
    pickle_in = open("metastructdata.pickle", "rb")
    struct = pickle.load(pickle_in)
    pickle_in.close()

    # Specifies the ranges for each parameter, decided by looking at the data,
    # if these ranges aren't used we won't be able to see the differences in the distributions that well
    abs_range_array = [[0, .5], [0, .6], [0, .6], [0, .3], [0, .6]]

    # max_gen is the generation we want to look at, here we go up to 9 generations
    for max_gen in range(9):


        # # # NOW WE DO IT FOR ALL GENERATIONS UP TO "max_gen"


        # # Format the cycle parameters like in Lee's paper, and in the POWERPOINT
        # Sister
        gen_time_array =

        gen_time_array = np.concatenate(np.array([abs(struct.A_dict_sis[keyA]['generationtime'].loc[:max_gen] - struct.B_dict_sis[keyB]['generationtime'].loc[:max_gen]) for keyA, keyB in zip(struct.A_dict_sis.keys(), struct.B_dict_sis.keys())]))
        alpha_array = np.concatenate(np.array([abs(struct.A_dict_sis[keyA]['growth_length'].loc[:max_gen] - struct.B_dict_sis[keyB]['growth_length'].loc[:max_gen]) for keyA, keyB in zip(struct.A_dict_sis.keys(), struct.B_dict_sis.keys())]))
        phi_array = np.concatenate(np.array([abs(struct.A_dict_sis[keyA]['generationtime'].loc[:max_gen]*struct.A_dict_sis[keyA]['growth_length'].loc[:max_gen] - struct.B_dict_sis[keyB]['generationtime'].loc[:max_gen]*struct.B_dict_sis[keyB]['growth_length'].loc[:max_gen]) for keyA, keyB in zip(struct.A_dict_sis.keys(), struct.B_dict_sis.keys())]))
        div_ratios = np.concatenate(np.array([abs(np.log(struct.A_dict_sis[keyA]['division_ratios__f_n'].loc[:max_gen]) - np.log(struct.B_dict_sis[keyB]['division_ratios__f_n'].loc[:max_gen])) for keyA, keyB in zip(struct.A_dict_sis.keys(), struct.B_dict_sis.keys())]))
        birth_lengths = np.concatenate(np.array([abs(np.log(struct.A_dict_sis[keyA]['length_birth'].loc[:max_gen] / struct.x_avg) - np.log(struct.B_dict_sis[keyB]['length_birth'].loc[:max_gen] / struct.x_avg)) for keyA, keyB in zip(struct.A_dict_sis.keys(), struct.B_dict_sis.keys())]))

        sis_diff_array = [gen_time_array, alpha_array, phi_array, div_ratios, birth_lengths]

        # Non-Sister
        gen_time_array = np.concatenate(np.array([abs(
            struct.A_dict_non_sis[keyA]['generationtime'].loc[:max_gen] - struct.B_dict_non_sis[keyB]['generationtime'].loc[
                                                                      :max_gen]) for keyA, keyB in
                                                  zip(struct.A_dict_non_sis.keys(), struct.B_dict_non_sis.keys())]))
        alpha_array = np.concatenate(np.array([abs(
            struct.A_dict_non_sis[keyA]['growth_length'].loc[:max_gen] - struct.B_dict_non_sis[keyB]['growth_length'].loc[
                                                                     :max_gen]) for keyA, keyB in
                                               zip(struct.A_dict_non_sis.keys(), struct.B_dict_non_sis.keys())]))
        phi_array = np.concatenate(np.array([abs(
            struct.A_dict_non_sis[keyA]['generationtime'].loc[:max_gen] * struct.A_dict_non_sis[keyA]['growth_length'].loc[
                                                                      :max_gen] - struct.B_dict_non_sis[keyB][
                                                                                      'generationtime'].loc[:max_gen] *
            struct.B_dict_non_sis[keyB]['growth_length'].loc[:max_gen]) for keyA, keyB in
                                             zip(struct.A_dict_non_sis.keys(), struct.B_dict_non_sis.keys())]))
        div_ratios = np.concatenate(np.array([abs(
            np.log(struct.A_dict_non_sis[keyA]['division_ratios__f_n'].loc[:max_gen]) - np.log(
                struct.B_dict_non_sis[keyB]['division_ratios__f_n'].loc[:max_gen])) for keyA, keyB in
                                              zip(struct.A_dict_non_sis.keys(), struct.B_dict_non_sis.keys())]))
        birth_lengths = np.concatenate(np.array([abs(
            np.log(struct.A_dict_non_sis[keyA]['length_birth'].loc[:max_gen] / struct.x_avg) - np.log(
                struct.B_dict_non_sis[keyB]['length_birth'].loc[:max_gen] / struct.x_avg)) for keyA, keyB in
                                                 zip(struct.A_dict_non_sis.keys(), struct.B_dict_non_sis.keys())]))

        non_sis_diff_array = [gen_time_array, alpha_array, phi_array, div_ratios, birth_lengths]

        # Control
        gen_time_array = np.concatenate(np.array([abs(
            struct.A_dict_both[keyA]['generationtime'].loc[:max_gen] - struct.B_dict_both[keyB]['generationtime'].loc[
                                                                      :max_gen]) for keyA, keyB in
                                                  zip(struct.A_dict_both.keys(), struct.B_dict_both.keys())]))
        alpha_array = np.concatenate(np.array([abs(
            struct.A_dict_both[keyA]['growth_length'].loc[:max_gen] - struct.B_dict_both[keyB]['growth_length'].loc[
                                                                     :max_gen]) for keyA, keyB in
                                               zip(struct.A_dict_both.keys(), struct.B_dict_both.keys())]))
        phi_array = np.concatenate(np.array([abs(
            struct.A_dict_both[keyA]['generationtime'].loc[:max_gen] * struct.A_dict_both[keyA]['growth_length'].loc[
                                                                      :max_gen] - struct.B_dict_both[keyB][
                                                                                      'generationtime'].loc[:max_gen] *
            struct.B_dict_both[keyB]['growth_length'].loc[:max_gen]) for keyA, keyB in
                                             zip(struct.A_dict_both.keys(), struct.B_dict_both.keys())]))
        div_ratios = np.concatenate(np.array([abs(
            np.log(struct.A_dict_both[keyA]['division_ratios__f_n'].loc[:max_gen]) - np.log(
                struct.B_dict_both[keyB]['division_ratios__f_n'].loc[:max_gen])) for keyA, keyB in
                                              zip(struct.A_dict_both.keys(), struct.B_dict_both.keys())]))
        birth_lengths = np.concatenate(np.array([abs(
            np.log(struct.A_dict_both[keyA]['length_birth'].loc[:max_gen] / struct.x_avg) - np.log(
                struct.B_dict_both[keyB]['length_birth'].loc[:max_gen] / struct.x_avg)) for keyA, keyB in
                                                 zip(struct.A_dict_both.keys(), struct.B_dict_both.keys())]))

        both_diff_array = [gen_time_array, alpha_array, phi_array, div_ratios, birth_lengths]

        # Parameter, number of boxes in histogram, that I found to be the most insightful
        Nbins = None

        # Name of the cycle parameters for labels
        param_array = [r'Generation time ($T$)', r'Growth rate ($\alpha$)', r'Accumulation Exponent ($\phi$)', r'Log of Division Ratio ($\log(f)$)', r'Normalized Birth Size ($\log(\frac{x}{x^*})$)']

        for ind in range(len(param_array)):
            # Create the labels for each data set
            label_sis = GetLabels(sis_diff_array[ind], "Sister")
            label_non = GetLabels(non_sis_diff_array[ind], "Non-Sister")
            label_both = GetLabels(both_diff_array[ind], "Control")

            # Create the x-axis label
            xlabel = 'Absolute difference for cycle parameter '+str(param_array[ind])

            # not really title but filename
            title = str(param_array[ind]) + ', ' + 'up to ' + str(max_gen) + ' generations'

            # Graph the Histograms and save them
            HistOfSlopes(sis_diff_array[ind], non_sis_diff_array[ind], both_diff_array[ind], label_sis, label_non, label_both, Nbins, abs_range_array[ind], xlabel, title)


        # # # NOW WE DO IT FOR ONE GENERATION AT A TIME


        # Sister
        gen_time_array = np.array([abs(
            struct.A_dict_sis[keyA]['generationtime'].loc[max_gen] - struct.B_dict_sis[keyB]['generationtime'].loc[
                                                                      max_gen]) for keyA, keyB in
                                   zip(struct.A_dict_sis.keys(), struct.B_dict_sis.keys())])
        alpha_array = np.array([abs(
            struct.A_dict_sis[keyA]['growth_length'].loc[max_gen] - struct.B_dict_sis[keyB]['growth_length'].loc[
                                                                     max_gen]) for keyA, keyB in
                                zip(struct.A_dict_sis.keys(), struct.B_dict_sis.keys())])
        phi_array = np.array([abs(
            struct.A_dict_sis[keyA]['generationtime'].loc[max_gen] * struct.A_dict_sis[keyA]['growth_length'].loc[
                                                                      max_gen] - struct.B_dict_sis[keyB][
                                                                                      'generationtime'].loc[
                                                                                  max_gen] *
            struct.B_dict_sis[keyB]['growth_length'].loc[max_gen]) for keyA, keyB in
                              zip(struct.A_dict_sis.keys(), struct.B_dict_sis.keys())])
        div_ratios = np.array([abs(np.log(struct.A_dict_sis[keyA]['division_ratios__f_n'].loc[max_gen]) - np.log(
            struct.B_dict_sis[keyB]['division_ratios__f_n'].loc[max_gen])) for keyA, keyB in
                               zip(struct.A_dict_sis.keys(), struct.B_dict_sis.keys())])
        birth_lengths = np.array([abs(
            np.log(struct.A_dict_sis[keyA]['length_birth'].loc[max_gen] / struct.x_avg) - np.log(
                struct.B_dict_sis[keyB]['length_birth'].loc[max_gen] / struct.x_avg)) for keyA, keyB in
                                  zip(struct.A_dict_sis.keys(), struct.B_dict_sis.keys())])

        sis_diff_array = [gen_time_array, alpha_array, phi_array, div_ratios, birth_lengths]

        # Non-Sister
        gen_time_array = np.array([abs(
            struct.A_dict_non_sis[keyA]['generationtime'].loc[max_gen] - struct.B_dict_non_sis[keyB][
                                                                              'generationtime'].loc[
                                                                          max_gen]) for keyA, keyB in
            zip(struct.A_dict_non_sis.keys(), struct.B_dict_non_sis.keys())])
        alpha_array = np.array([abs(
            struct.A_dict_non_sis[keyA]['growth_length'].loc[max_gen] - struct.B_dict_non_sis[keyB][
                                                                             'growth_length'].loc[
                                                                         max_gen]) for keyA, keyB in
            zip(struct.A_dict_non_sis.keys(), struct.B_dict_non_sis.keys())])
        phi_array = np.array([abs(
            struct.A_dict_non_sis[keyA]['generationtime'].loc[max_gen] * struct.A_dict_non_sis[keyA][
                                                                              'growth_length'].loc[
                                                                          max_gen] - struct.B_dict_non_sis[keyB][
                                                                                          'generationtime'].loc[
                                                                                      max_gen] *
            struct.B_dict_non_sis[keyB]['growth_length'].loc[max_gen]) for keyA, keyB in
            zip(struct.A_dict_non_sis.keys(), struct.B_dict_non_sis.keys())])
        div_ratios = np.array(
            [abs(np.log(struct.A_dict_non_sis[keyA]['division_ratios__f_n'].loc[max_gen]) - np.log(
                struct.B_dict_non_sis[keyB]['division_ratios__f_n'].loc[max_gen])) for keyA, keyB in
             zip(struct.A_dict_non_sis.keys(), struct.B_dict_non_sis.keys())])
        birth_lengths = np.array([abs(
            np.log(struct.A_dict_non_sis[keyA]['length_birth'].loc[max_gen] / struct.x_avg) - np.log(
                struct.B_dict_non_sis[keyB]['length_birth'].loc[max_gen] / struct.x_avg)) for keyA, keyB in
            zip(struct.A_dict_non_sis.keys(), struct.B_dict_non_sis.keys())])

        non_sis_diff_array = [gen_time_array, alpha_array, phi_array, div_ratios, birth_lengths]

        # Control
        gen_time_array = np.array([abs(
            struct.A_dict_both[keyA]['generationtime'].loc[max_gen] - struct.B_dict_both[keyB][
                                                                           'generationtime'].loc[
                                                                       max_gen]) for keyA, keyB in
            zip(struct.A_dict_both.keys(), struct.B_dict_both.keys())])
        alpha_array = np.array([abs(
            struct.A_dict_both[keyA]['growth_length'].loc[max_gen] - struct.B_dict_both[keyB]['growth_length'].loc[
                                                                      max_gen]) for keyA, keyB in
            zip(struct.A_dict_both.keys(), struct.B_dict_both.keys())])
        phi_array = np.array([abs(
            struct.A_dict_both[keyA]['generationtime'].loc[max_gen] * struct.A_dict_both[keyA][
                                                                           'growth_length'].loc[
                                                                       max_gen] - struct.B_dict_both[keyB][
                                                                                       'generationtime'].loc[
                                                                                   max_gen] *
            struct.B_dict_both[keyB]['growth_length'].loc[max_gen]) for keyA, keyB in
            zip(struct.A_dict_both.keys(), struct.B_dict_both.keys())])
        div_ratios = np.array([abs(np.log(struct.A_dict_both[keyA]['division_ratios__f_n'].loc[max_gen]) - np.log(
            struct.B_dict_both[keyB]['division_ratios__f_n'].loc[max_gen])) for keyA, keyB in
                               zip(struct.A_dict_both.keys(), struct.B_dict_both.keys())])
        birth_lengths = np.array([abs(
            np.log(struct.A_dict_both[keyA]['length_birth'].loc[max_gen] / struct.x_avg) - np.log(
                struct.B_dict_both[keyB]['length_birth'].loc[max_gen] / struct.x_avg)) for keyA, keyB in
            zip(struct.A_dict_both.keys(), struct.B_dict_both.keys())])

        both_diff_array = [gen_time_array, alpha_array, phi_array, div_ratios, birth_lengths]

        # Parameter, number of boxes in histogram, that I found to be the most insightful
        Nbins = None

        for ind in range(len(param_array)):
            # Create the labels for each data set
            label_sis = GetLabels(sis_diff_array[ind], "Sister")
            label_non = GetLabels(non_sis_diff_array[ind], "Non-Sister")
            label_both = GetLabels(both_diff_array[ind], "Control")

            # Create the x-axis label
            xlabel = 'Absolute difference for cycle parameter ' + str(param_array[ind])

            # not really title but filename
            title = str(param_array[ind]) + ', ' + 'using only generation number ' + str(max_gen)

            # Graph the Histograms and save them
            HistOfSlopes(sis_diff_array[ind], non_sis_diff_array[ind], both_diff_array[ind], label_sis, label_non,
                         label_both, Nbins, abs_range_array[ind], xlabel, title)


if __name__ == '__main__':
    main()
