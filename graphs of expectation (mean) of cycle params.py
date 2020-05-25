
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


def HistOfmeanPerGen(dist_sis, dist_non, dist_both, label_sis, label_non, label_both, xlabel, filename, title, ylabel):
    sis_best_fit = np.array([np.mean(dist_sis) for inp in range(len(dist_sis))])
    non_best_fit = np.array([np.mean(dist_non) for inp in range(len(dist_non))])
    both_best_fit = np.array([np.mean(dist_both) for inp in range(len(dist_both))])

    plt.plot(dist_sis, label=label_sis, marker='.', color='b')
    plt.plot(dist_non, label=label_non, marker='.', color='orange')
    plt.plot(dist_both, label=label_both, marker='.', color='green')
    plt.plot(sis_best_fit, alpha=.5, linewidth='3', color='b')
    plt.plot(non_best_fit, alpha=.5, linewidth='3', color='orange')
    plt.plot(both_best_fit, alpha=.5, linewidth='3', color='green')
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
    # plt.savefig(filename+'.png', dpi=300)
    # plt.close()


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

    # Where we will put all the variances for up to and then just generation
    sis_diff_array_mean = []
    non_sis_diff_array_mean = []
    both_diff_array_mean = []
    sis_diff_array_mean_gen = []
    non_sis_diff_array_mean_gen = []
    both_diff_array_mean_gen = []

    # max_gen is the generation we want to look at, here we go up to 9 generations
    how_many = 20
    for max_gen in range(how_many):


        # # # NOW WE DO IT FOR ALL GENERATIONS UP TO "max_gen"


        # # Format the cycle parameters like in Lee's paper, and in the POWERPOINT
        # Sister

        gen_time_array = np.mean([np.sum(struct.A_dict_sis[keyA]['generationtime'].loc[:max_gen] - struct.B_dict_sis[keyB]['generationtime'].loc[:max_gen]) for keyA, keyB in zip(struct.A_dict_sis.keys(), struct.B_dict_sis.keys()) if min(len(struct.A_dict_sis[keyA]['generationtime']), len(struct.B_dict_sis[keyB]['generationtime'])) > max_gen])
        alpha_array = np.mean([np.sum(struct.A_dict_sis[keyA]['growth_length'].loc[:max_gen] - struct.B_dict_sis[keyB]['growth_length'].loc[:max_gen]) for keyA, keyB in zip(struct.A_dict_sis.keys(), struct.B_dict_sis.keys()) if min(len(struct.A_dict_sis[keyA]['generationtime']), len(struct.B_dict_sis[keyB]['generationtime'])) > max_gen])
        phi_array = np.mean([np.sum(struct.A_dict_sis[keyA]['generationtime'].loc[:max_gen]*struct.A_dict_sis[keyA]['growth_length'].loc[:max_gen] - struct.B_dict_sis[keyB]['generationtime'].loc[:max_gen]*struct.B_dict_sis[keyB]['growth_length'].loc[:max_gen]) for keyA, keyB in zip(struct.A_dict_sis.keys(), struct.B_dict_sis.keys()) if min(len(struct.A_dict_sis[keyA]['generationtime']), len(struct.B_dict_sis[keyB]['generationtime'])) > max_gen])
        div_ratios = np.mean([np.sum(np.log(struct.A_dict_sis[keyA]['division_ratios__f_n'].loc[:max_gen]) - np.log(struct.B_dict_sis[keyB]['division_ratios__f_n'].loc[:max_gen])) for keyA, keyB in zip(struct.A_dict_sis.keys(), struct.B_dict_sis.keys()) if min(len(struct.A_dict_sis[keyA]['generationtime']), len(struct.B_dict_sis[keyB]['generationtime'])) > max_gen])
        birth_lengths = np.mean([np.sum(np.log(struct.A_dict_sis[keyA]['length_birth'].loc[:max_gen] / struct.x_avg) - np.log(struct.B_dict_sis[keyB]['length_birth'].loc[:max_gen] / struct.x_avg)) for keyA, keyB in zip(struct.A_dict_sis.keys(), struct.B_dict_sis.keys()) if min(len(struct.A_dict_sis[keyA]['generationtime']), len(struct.B_dict_sis[keyB]['generationtime'])) > max_gen])

        sis_diff_array = [gen_time_array, alpha_array, phi_array, div_ratios, birth_lengths]
        sis_diff_array_mean.append(np.array(sis_diff_array))

        # Non-Sister
        gen_time_array = np.mean([np.sum(
            struct.A_dict_non_sis[keyA]['generationtime'].loc[:max_gen] - struct.B_dict_non_sis[keyB]['generationtime'].loc[
                                                                      :max_gen]) for keyA, keyB in
                                                  zip(struct.A_dict_non_sis.keys(), struct.B_dict_non_sis.keys()) if min(len(struct.A_dict_non_sis[keyA]['generationtime']), len(struct.B_dict_non_sis[keyB]['generationtime'])) > max_gen])
        alpha_array = np.mean([np.sum(
            struct.A_dict_non_sis[keyA]['growth_length'].loc[:max_gen] - struct.B_dict_non_sis[keyB]['growth_length'].loc[
                                                                     :max_gen]) for keyA, keyB in
                                               zip(struct.A_dict_non_sis.keys(), struct.B_dict_non_sis.keys()) if min(len(struct.A_dict_non_sis[keyA]['generationtime']), len(struct.B_dict_non_sis[keyB]['generationtime'])) > max_gen])
        phi_array = np.mean([np.sum(
            struct.A_dict_non_sis[keyA]['generationtime'].loc[:max_gen] * struct.A_dict_non_sis[keyA]['growth_length'].loc[
                                                                      :max_gen] - struct.B_dict_non_sis[keyB][
                                                                                      'generationtime'].loc[:max_gen] *
            struct.B_dict_non_sis[keyB]['growth_length'].loc[:max_gen]) for keyA, keyB in
                                             zip(struct.A_dict_non_sis.keys(), struct.B_dict_non_sis.keys()) if min(len(struct.A_dict_non_sis[keyA]['generationtime']), len(struct.B_dict_non_sis[keyB]['generationtime'])) > max_gen])
        div_ratios = np.mean([np.sum(
            np.log(struct.A_dict_non_sis[keyA]['division_ratios__f_n'].loc[:max_gen]) - np.log(
                struct.B_dict_non_sis[keyB]['division_ratios__f_n'].loc[:max_gen])) for keyA, keyB in
                                              zip(struct.A_dict_non_sis.keys(), struct.B_dict_non_sis.keys()) if min(len(struct.A_dict_non_sis[keyA]['generationtime']), len(struct.B_dict_non_sis[keyB]['generationtime'])) > max_gen])
        birth_lengths = np.mean([np.sum(
            np.log(struct.A_dict_non_sis[keyA]['length_birth'].loc[:max_gen] / struct.x_avg) - np.log(
                struct.B_dict_non_sis[keyB]['length_birth'].loc[:max_gen] / struct.x_avg)) for keyA, keyB in
                                                 zip(struct.A_dict_non_sis.keys(), struct.B_dict_non_sis.keys()) if min(len(struct.A_dict_non_sis[keyA]['generationtime']), len(struct.B_dict_non_sis[keyB]['generationtime'])) > max_gen])

        non_sis_diff_array = [gen_time_array, alpha_array, phi_array, div_ratios, birth_lengths]
        non_sis_diff_array_mean.append(np.array(non_sis_diff_array))

        # Control
        gen_time_array = np.mean([np.sum(
            struct.A_dict_both[keyA]['generationtime'].loc[:max_gen] - struct.B_dict_both[keyB]['generationtime'].loc[
                                                                      :max_gen]) for keyA, keyB in
                                                  zip(struct.A_dict_both.keys(), struct.B_dict_both.keys()) if min(len(struct.A_dict_both[keyA]['generationtime']), len(struct.B_dict_both[keyB]['generationtime'])) > max_gen])
        alpha_array = np.mean([np.sum(
            struct.A_dict_both[keyA]['growth_length'].loc[:max_gen] - struct.B_dict_both[keyB]['growth_length'].loc[
                                                                     :max_gen]) for keyA, keyB in
                                               zip(struct.A_dict_both.keys(), struct.B_dict_both.keys()) if min(len(struct.A_dict_both[keyA]['generationtime']), len(struct.B_dict_both[keyB]['generationtime'])) > max_gen])
        phi_array = np.mean([np.sum(
            struct.A_dict_both[keyA]['generationtime'].loc[:max_gen] * struct.A_dict_both[keyA]['growth_length'].loc[
                                                                      :max_gen] - struct.B_dict_both[keyB][
                                                                                      'generationtime'].loc[:max_gen] *
            struct.B_dict_both[keyB]['growth_length'].loc[:max_gen]) for keyA, keyB in
                                             zip(struct.A_dict_both.keys(), struct.B_dict_both.keys()) if min(len(struct.A_dict_both[keyA]['generationtime']), len(struct.B_dict_both[keyB]['generationtime'])) > max_gen])
        div_ratios = np.mean([np.sum(
            np.log(struct.A_dict_both[keyA]['division_ratios__f_n'].loc[:max_gen]) - np.log(
                struct.B_dict_both[keyB]['division_ratios__f_n'].loc[:max_gen])) for keyA, keyB in
                                              zip(struct.A_dict_both.keys(), struct.B_dict_both.keys()) if min(len(struct.A_dict_both[keyA]['generationtime']), len(struct.B_dict_both[keyB]['generationtime'])) > max_gen])
        birth_lengths = np.mean([np.sum(
            np.log(struct.A_dict_both[keyA]['length_birth'].loc[:max_gen] / struct.x_avg) - np.log(
                struct.B_dict_both[keyB]['length_birth'].loc[:max_gen] / struct.x_avg)) for keyA, keyB in
                                                 zip(struct.A_dict_both.keys(), struct.B_dict_both.keys()) if min(len(struct.A_dict_both[keyA]['generationtime']), len(struct.B_dict_both[keyB]['generationtime'])) > max_gen])

        both_diff_array = [gen_time_array, alpha_array, phi_array, div_ratios, birth_lengths]
        both_diff_array_mean.append(np.array(both_diff_array))


        # # # NOW WE DO IT FOR ONE GENERATION AT A TIME

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

    # Name of the cycle parameters for labels
    param_array = [r'Generation time ($T$)', r'Growth rate ($\alpha$)', r'Accumulation Exponent ($\phi$)', r'Log of Division Ratio ($\log(f)$)', r'Normalized Birth Size ($\log(\frac{x}{x^*})$)']

    # separates up to and only options for data
    filename = [str(param_array[ind]) + ', ' + 'using up to said generation ' for ind in range(len(param_array))]
    title = filename

    for ind in range(len(param_array)):
        # Create the labels for each data set
        label_sis = "Sister" # GetLabels(sis_diff_array_mean[ind], "Sister")
        label_non = "Non-Sister" # GetLabels(non_sis_diff_array_mean[ind], "Non-Sister")
        label_both = "Control" # GetLabels(both_diff_array_mean[ind], "Control")

        # Create the x-axis label
        xlabel = 'generation number'
        ylabel = 'mean of cycle parameter '+str(param_array[ind])

        # Graph the Histograms and save them
        HistOfmeanPerGen([sis_diff_array_mean[dex][ind] for dex in range(len(sis_diff_array_mean))], [non_sis_diff_array_mean[dex][ind] for dex in range(len(non_sis_diff_array_mean))], [both_diff_array_mean[dex][ind] for dex in range(len(both_diff_array_mean))],
                        label_sis, label_non, label_both, xlabel, filename[ind], title[ind], ylabel)


    # Parameter, number of boxes in histogram, that I found to be the most insightful
    Nbins = None

    # separates up to and only options for data
    filename = [str(param_array[ind]) + ', ' + 'using only one generation ' for ind in range(len(param_array))]
    title = filename

    for ind in range(len(param_array)):
        # Create the labels for each data set
        label_sis = "Sister" # GetLabels(sis_diff_array_mean_gen[ind], "Sister")
        label_non = "Non-Sister" # GetLabels(non_sis_diff_array_mean_gen[ind], "Non-Sister")
        label_both = "Control" # GetLabels(both_diff_array_mean_gen[ind], "Control")

        # Create the x-axis label
        xlabel = 'generation number'
        ylabel = 'mean of cycle parameter ' + str(param_array[ind])

        # Graph the Histograms and save them
        HistOfmeanPerGen([sis_diff_array_mean_gen[dex][ind] for dex in range(len(sis_diff_array_mean_gen))], [non_sis_diff_array_mean_gen[dex][ind] for dex in range(len(non_sis_diff_array_mean_gen))], [both_diff_array_mean_gen[dex][ind] for dex in range(len(both_diff_array_mean_gen))], label_sis, label_non,
                     label_both, xlabel, filename[ind], title[ind], ylabel)


if __name__ == '__main__':
    main()
