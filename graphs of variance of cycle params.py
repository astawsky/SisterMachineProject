
from __future__ import print_function

import numpy as np

import matplotlib.pyplot as plt

import pickle

import scipy.stats as stats


def PearsonCorr(u, v):
    avg_u = np.mean(u)
    avg_v = np.mean(v)
    covariance = np.sum(np.array([(u[ID] - avg_u) * (v[ID] - avg_v) for ID in range(len(u))]))
    denominator = np.sqrt(np.sum(np.array([(u[ID] - avg_u) ** 2 for ID in range(len(u))]))) * \
                  np.sqrt(np.sum(np.array([(v[ID] - avg_v) ** 2 for ID in range(len(v))])))
    weighted_sum = covariance / denominator

    # np.corrcoef(np.array(u), np.array(v))[0, 1] --> Another way to calculate the pcorr coeff using numpy, gives similar answer

    return weighted_sum


def HistOfVarPerGen(dist_sis, dist_non, dist_both, label_sis, label_non, label_both, xlabel, filename, title, ylabel):
    sis_best_fit = np.array([stats.linregress(x=range(len(dist_sis)), y=dist_sis)[0]*inp + stats.linregress(x=range(len(dist_sis)), y=dist_sis)[1] for inp in range(len(dist_sis))])
    non_best_fit = np.array([stats.linregress(x=range(len(dist_non)), y=dist_non)[0] * inp +
                             stats.linregress(x=range(len(dist_non)), y=dist_non)[1] for inp in range(len(dist_non))])
    both_best_fit = np.array([stats.linregress(x=range(len(dist_both)), y=dist_both)[0] * inp +
                             stats.linregress(x=range(len(dist_both)), y=dist_both)[1] for inp in range(len(dist_both))])

    x = np.arange(len(dist_sis))
    x = x[:, np.newaxis]
    a_sis, _, _, _ = np.linalg.lstsq(x, dist_sis, rcond=None)
    a_non, _, _, _ = np.linalg.lstsq(x, dist_non, rcond=None)
    a_both, _, _, _ = np.linalg.lstsq(x, dist_both, rcond=None)

    # plt.plot(dist_sis, label=label_sis+r' {:.2e}+{:.2e}*(gen. num.)'.format(stats.linregress(x=range(len(dist_sis)), y=dist_sis)[1], stats.linregress(
    #     x=range(len(dist_sis)), y=dist_sis)[0]), marker='.', color='b')
    # plt.plot(dist_non, label=label_non+r' {:.2e}+{:.2e}*(gen. num.)'.format(stats.linregress(x=range(len(dist_non)), y=dist_non)[1], stats.linregress(
    #     x=range(len(dist_non)), y=dist_non)[0]), marker='.', color='orange')
    # plt.plot(dist_both, label=label_both+r' {:.2e}+{:.2e}*(gen. num.)'.format(stats.linregress(x=range(len(dist_both)), y=dist_both)[1],
    #     stats.linregress(x=range(len(dist_both)), y=dist_both)[0]), marker='.', color='green')
    # plt.plot(sis_best_fit, alpha=.5, linewidth='3')
    # plt.plot(non_best_fit, alpha=.5, linewidth='3')
    # plt.plot(both_best_fit, alpha=.5, linewidth='3')
    plt.plot(dist_sis,
             label=label_sis + r' {}+{:.2e}*(gen. num.)'.format(0, a_sis[0]), marker='.', color='b')
    plt.plot(dist_non,
             label=label_non + r' {}+{:.2e}*(gen. num.)'.format(0, a_non[0]), marker='.', color='orange')
    plt.plot(dist_both, label=label_both + r' {}+{:.2e}*(gen. num.)'.format(0, a_both[0]),
             marker='.', color='green')
    plt.plot(x, a_sis*x, alpha=.5, linewidth='3')
    plt.plot(x, a_non*x, alpha=.5, linewidth='3')
    plt.plot(x, a_both*x, alpha=.5, linewidth='3')
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    # plt.show()
    plt.savefig(filename+'.png', dpi=300)
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

    # Where we will put all the variances for up to and then just generation
    sis_diff_array_var = []
    non_sis_diff_array_var = []
    both_diff_array_var = []
    sis_diff_array_var_gen = []
    non_sis_diff_array_var_gen = []
    both_diff_array_var_gen = []

    # max_gen is the generation we want to look at, here we go up to 9 generations
    how_many = 20
    for max_gen in range(how_many):


        # # # NOW WE DO IT FOR ALL GENERATIONS UP TO "max_gen"


        # # Format the cycle parameters like in Lee's paper, and in the POWERPOINT
        # Sister

        gen_time_array = np.var([np.sum(struct.A_dict_sis[keyA]['generationtime'].loc[:max_gen] - struct.B_dict_sis[keyB]['generationtime'].loc[:max_gen]) for keyA, keyB in zip(struct.A_dict_sis.keys(), struct.B_dict_sis.keys()) if min(len(struct.A_dict_sis[keyA]['generationtime']), len(struct.B_dict_sis[keyB]['generationtime'])) > max_gen])
        alpha_array = np.var([np.sum(struct.A_dict_sis[keyA]['growth_length'].loc[:max_gen] - struct.B_dict_sis[keyB]['growth_length'].loc[:max_gen]) for keyA, keyB in zip(struct.A_dict_sis.keys(), struct.B_dict_sis.keys()) if min(len(struct.A_dict_sis[keyA]['generationtime']), len(struct.B_dict_sis[keyB]['generationtime'])) > max_gen])
        phi_array = np.var([np.sum(struct.A_dict_sis[keyA]['generationtime'].loc[:max_gen]*struct.A_dict_sis[keyA]['growth_length'].loc[:max_gen] - struct.B_dict_sis[keyB]['generationtime'].loc[:max_gen]*struct.B_dict_sis[keyB]['growth_length'].loc[:max_gen]) for keyA, keyB in zip(struct.A_dict_sis.keys(), struct.B_dict_sis.keys()) if min(len(struct.A_dict_sis[keyA]['generationtime']), len(struct.B_dict_sis[keyB]['generationtime'])) > max_gen])
        div_ratios = np.var([np.sum(np.log(struct.A_dict_sis[keyA]['division_ratios__f_n'].loc[:max_gen]) - np.log(struct.B_dict_sis[keyB]['division_ratios__f_n'].loc[:max_gen])) for keyA, keyB in zip(struct.A_dict_sis.keys(), struct.B_dict_sis.keys()) if min(len(struct.A_dict_sis[keyA]['generationtime']), len(struct.B_dict_sis[keyB]['generationtime'])) > max_gen])
        # div_ratios = np.var([np.sum(struct.A_dict_sis[keyA]['growth_length'].loc[:max_gen] - struct.B_dict_sis[keyB]['growth_length'].loc[:max_gen] - np.log(struct.A_dict_sis[keyA]['division_ratios__f_n'].loc[:max_gen]) + np.log(struct.B_dict_sis[keyB]['division_ratios__f_n'].loc[:max_gen])) for keyA, keyB in zip(struct.A_dict_sis.keys(), struct.B_dict_sis.keys()) if min(len(struct.A_dict_sis[keyA]['generationtime']), len(struct.B_dict_sis[keyB]['generationtime'])) > max_gen])
        birth_lengths = np.var([np.sum(np.log(struct.A_dict_sis[keyA]['length_birth'].loc[:max_gen] / struct.x_avg) - np.log(struct.B_dict_sis[keyB]['length_birth'].loc[:max_gen] / struct.x_avg)) for keyA, keyB in zip(struct.A_dict_sis.keys(), struct.B_dict_sis.keys()) if min(len(struct.A_dict_sis[keyA]['generationtime']), len(struct.B_dict_sis[keyB]['generationtime'])) > max_gen])
        
        sis_diff_array = [gen_time_array, alpha_array, phi_array, div_ratios, birth_lengths]
        sis_diff_array_var.append(np.array(sis_diff_array))

        # Non-Sister
        gen_time_array = np.var([np.sum(
            struct.A_dict_non_sis[keyA]['generationtime'].loc[:max_gen] - struct.B_dict_non_sis[keyB]['generationtime'].loc[
                                                                      :max_gen]) for keyA, keyB in
                                                  zip(struct.A_dict_non_sis.keys(), struct.B_dict_non_sis.keys()) if min(len(struct.A_dict_non_sis[keyA]['generationtime']), len(struct.B_dict_non_sis[keyB]['generationtime'])) > max_gen])
        alpha_array = np.var([np.sum(
            struct.A_dict_non_sis[keyA]['growth_length'].loc[:max_gen] - struct.B_dict_non_sis[keyB]['growth_length'].loc[
                                                                     :max_gen]) for keyA, keyB in
                                               zip(struct.A_dict_non_sis.keys(), struct.B_dict_non_sis.keys()) if min(len(struct.A_dict_non_sis[keyA]['generationtime']), len(struct.B_dict_non_sis[keyB]['generationtime'])) > max_gen])
        phi_array = np.var([np.sum(
            struct.A_dict_non_sis[keyA]['generationtime'].loc[:max_gen] * struct.A_dict_non_sis[keyA]['growth_length'].loc[
                                                                      :max_gen] - struct.B_dict_non_sis[keyB][
                                                                                      'generationtime'].loc[:max_gen] *
            struct.B_dict_non_sis[keyB]['growth_length'].loc[:max_gen]) for keyA, keyB in
                                             zip(struct.A_dict_non_sis.keys(), struct.B_dict_non_sis.keys()) if min(len(struct.A_dict_non_sis[keyA]['generationtime']), len(struct.B_dict_non_sis[keyB]['generationtime'])) > max_gen])
        div_ratios = np.var([np.sum(
            np.log(struct.A_dict_non_sis[keyA]['division_ratios__f_n'].loc[:max_gen]) - np.log(
                struct.B_dict_non_sis[keyB]['division_ratios__f_n'].loc[:max_gen])) for keyA, keyB in
                                              zip(struct.A_dict_non_sis.keys(), struct.B_dict_non_sis.keys()) if min(len(struct.A_dict_non_sis[keyA]['generationtime']), len(struct.B_dict_non_sis[keyB]['generationtime'])) > max_gen])
        # div_ratios = np.var([np.sum(struct.A_dict_non_sis[keyA]['growth_length'].loc[:max_gen] - struct.B_dict_non_sis[keyB]['growth_length'].loc[:max_gen] - np.log(struct.A_dict_non_sis[keyA]['division_ratios__f_n'].loc[:max_gen]) + np.log(struct.B_dict_non_sis[keyB]['division_ratios__f_n'].loc[:max_gen])) for keyA, keyB in zip(struct.A_dict_non_sis.keys(), struct.B_dict_non_sis.keys()) if min(len(struct.A_dict_non_sis[keyA]['generationtime']), len(struct.B_dict_non_sis[keyB]['generationtime'])) > max_gen])
        birth_lengths = np.var([np.sum(
            np.log(struct.A_dict_non_sis[keyA]['length_birth'].loc[:max_gen] / struct.x_avg) - np.log(
                struct.B_dict_non_sis[keyB]['length_birth'].loc[:max_gen] / struct.x_avg)) for keyA, keyB in
                                                 zip(struct.A_dict_non_sis.keys(), struct.B_dict_non_sis.keys()) if min(len(struct.A_dict_non_sis[keyA]['generationtime']), len(struct.B_dict_non_sis[keyB]['generationtime'])) > max_gen])

        non_sis_diff_array = [gen_time_array, alpha_array, phi_array, div_ratios, birth_lengths]
        non_sis_diff_array_var.append(np.array(non_sis_diff_array))

        # Control
        gen_time_array = np.var([np.sum(
            struct.A_dict_both[keyA]['generationtime'].loc[:max_gen] - struct.B_dict_both[keyB]['generationtime'].loc[
                                                                      :max_gen]) for keyA, keyB in
                                                  zip(struct.A_dict_both.keys(), struct.B_dict_both.keys()) if min(len(struct.A_dict_both[keyA]['generationtime']), len(struct.B_dict_both[keyB]['generationtime'])) > max_gen])
        alpha_array = np.var([np.sum(
            struct.A_dict_both[keyA]['growth_length'].loc[:max_gen] - struct.B_dict_both[keyB]['growth_length'].loc[
                                                                     :max_gen]) for keyA, keyB in
                                               zip(struct.A_dict_both.keys(), struct.B_dict_both.keys()) if min(len(struct.A_dict_both[keyA]['generationtime']), len(struct.B_dict_both[keyB]['generationtime'])) > max_gen])
        phi_array = np.var([np.sum(
            struct.A_dict_both[keyA]['generationtime'].loc[:max_gen] * struct.A_dict_both[keyA]['growth_length'].loc[
                                                                      :max_gen] - struct.B_dict_both[keyB][
                                                                                      'generationtime'].loc[:max_gen] *
            struct.B_dict_both[keyB]['growth_length'].loc[:max_gen]) for keyA, keyB in
                                             zip(struct.A_dict_both.keys(), struct.B_dict_both.keys()) if min(len(struct.A_dict_both[keyA]['generationtime']), len(struct.B_dict_both[keyB]['generationtime'])) > max_gen])
        div_ratios = np.var([np.sum(
            np.log(struct.A_dict_both[keyA]['division_ratios__f_n'].loc[:max_gen]) - np.log(
                struct.B_dict_both[keyB]['division_ratios__f_n'].loc[:max_gen])) for keyA, keyB in
                                              zip(struct.A_dict_both.keys(), struct.B_dict_both.keys()) if min(len(struct.A_dict_both[keyA]['generationtime']), len(struct.B_dict_both[keyB]['generationtime'])) > max_gen])
        # div_ratios = np.var([np.sum(struct.A_dict_both[keyA]['growth_length'].loc[:max_gen] - struct.B_dict_both[keyB]['growth_length'].loc[:max_gen] - np.log(struct.A_dict_both[keyA]['division_ratios__f_n'].loc[:max_gen]) + np.log(struct.B_dict_both[keyB]['division_ratios__f_n'].loc[:max_gen])) for keyA, keyB in zip(struct.A_dict_both.keys(), struct.B_dict_both.keys()) if min(len(struct.A_dict_both[keyA]['generationtime']), len(struct.B_dict_both[keyB]['generationtime'])) > max_gen])
        birth_lengths = np.var([np.sum(
            np.log(struct.A_dict_both[keyA]['length_birth'].loc[:max_gen] / struct.x_avg) - np.log(
                struct.B_dict_both[keyB]['length_birth'].loc[:max_gen] / struct.x_avg)) for keyA, keyB in
                                                 zip(struct.A_dict_both.keys(), struct.B_dict_both.keys()) if min(len(struct.A_dict_both[keyA]['generationtime']), len(struct.B_dict_both[keyB]['generationtime'])) > max_gen])

        both_diff_array = [gen_time_array, alpha_array, phi_array, div_ratios, birth_lengths]
        both_diff_array_var.append(np.array(both_diff_array))


        """

        # # # NOW WE DO IT FOR ONE GENERATION AT A TIME

        # Sister

        gen_time_array = np.var([np.array(
            struct.A_dict_sis[keyA]['generationtime'].loc[max_gen] - struct.B_dict_sis[keyB]['generationtime'].loc[
                                                                      max_gen]) for keyA, keyB in
                                 zip(struct.A_dict_sis.keys(), struct.B_dict_sis.keys()) if
                                 min(len(struct.A_dict_sis[keyA]['generationtime']),
                                     len(struct.B_dict_sis[keyB]['generationtime'])) > max_gen])
        alpha_array = np.var([np.array(
            struct.A_dict_sis[keyA]['growth_length'].loc[max_gen] - struct.B_dict_sis[keyB]['growth_length'].loc[
                                                                     max_gen]) for keyA, keyB in
                              zip(struct.A_dict_sis.keys(), struct.B_dict_sis.keys()) if
                              min(len(struct.A_dict_sis[keyA]['generationtime']),
                                  len(struct.B_dict_sis[keyB]['generationtime'])) > max_gen])
        phi_array = np.var([np.array(
            struct.A_dict_sis[keyA]['generationtime'].loc[max_gen] * struct.A_dict_sis[keyA]['growth_length'].loc[
                                                                      max_gen] - struct.B_dict_sis[keyB][
                                                                                      'generationtime'].loc[max_gen] *
            struct.B_dict_sis[keyB]['growth_length'].loc[max_gen]) for keyA, keyB in
                            zip(struct.A_dict_sis.keys(), struct.B_dict_sis.keys()) if
                            min(len(struct.A_dict_sis[keyA]['generationtime']),
                                len(struct.B_dict_sis[keyB]['generationtime'])) > max_gen])
        div_ratios = np.var([np.array(np.log(struct.A_dict_sis[keyA]['division_ratios__f_n'].loc[max_gen]) - np.log(
            struct.B_dict_sis[keyB]['division_ratios__f_n'].loc[max_gen])) for keyA, keyB in
                             zip(struct.A_dict_sis.keys(), struct.B_dict_sis.keys()) if
                             min(len(struct.A_dict_sis[keyA]['generationtime']),
                                 len(struct.B_dict_sis[keyB]['generationtime'])) > max_gen])
        birth_lengths = np.var([np.array(
            np.log(struct.A_dict_sis[keyA]['length_birth'].loc[max_gen] / struct.x_avg) - np.log(
                struct.B_dict_sis[keyB]['length_birth'].loc[max_gen] / struct.x_avg)) for keyA, keyB in
                                zip(struct.A_dict_sis.keys(), struct.B_dict_sis.keys()) if
                                min(len(struct.A_dict_sis[keyA]['generationtime']),
                                    len(struct.B_dict_sis[keyB]['generationtime'])) > max_gen])

        sis_diff_array = [gen_time_array, alpha_array, phi_array, div_ratios, birth_lengths]
        sis_diff_array_var_gen.append(np.array(sis_diff_array))

        # Non-Sister
        gen_time_array = np.var([np.array(
            struct.A_dict_non_sis[keyA]['generationtime'].loc[max_gen] - struct.B_dict_non_sis[keyB][
                                                                              'generationtime'].loc[
                                                                          max_gen]) for keyA, keyB in
            zip(struct.A_dict_non_sis.keys(), struct.B_dict_non_sis.keys()) if
            min(len(struct.A_dict_non_sis[keyA]['generationtime']),
                len(struct.B_dict_non_sis[keyB]['generationtime'])) > max_gen])
        alpha_array = np.var([np.array(
            struct.A_dict_non_sis[keyA]['growth_length'].loc[max_gen] - struct.B_dict_non_sis[keyB][
                                                                             'growth_length'].loc[
                                                                         max_gen]) for keyA, keyB in
            zip(struct.A_dict_non_sis.keys(), struct.B_dict_non_sis.keys()) if
            min(len(struct.A_dict_non_sis[keyA]['generationtime']),
                len(struct.B_dict_non_sis[keyB]['generationtime'])) > max_gen])
        phi_array = np.var([np.array(
            struct.A_dict_non_sis[keyA]['generationtime'].loc[max_gen] * struct.A_dict_non_sis[keyA][
                                                                              'growth_length'].loc[
                                                                          max_gen] - struct.B_dict_non_sis[keyB][
                                                                                          'generationtime'].loc[
                                                                                      max_gen] *
            struct.B_dict_non_sis[keyB]['growth_length'].loc[max_gen]) for keyA, keyB in
            zip(struct.A_dict_non_sis.keys(), struct.B_dict_non_sis.keys()) if
            min(len(struct.A_dict_non_sis[keyA]['generationtime']),
                len(struct.B_dict_non_sis[keyB]['generationtime'])) > max_gen])
        div_ratios = np.var([np.array(
            np.log(struct.A_dict_non_sis[keyA]['division_ratios__f_n'].loc[max_gen]) - np.log(
                struct.B_dict_non_sis[keyB]['division_ratios__f_n'].loc[max_gen])) for keyA, keyB in
            zip(struct.A_dict_non_sis.keys(), struct.B_dict_non_sis.keys()) if
            min(len(struct.A_dict_non_sis[keyA]['generationtime']),
                len(struct.B_dict_non_sis[keyB]['generationtime'])) > max_gen])
        birth_lengths = np.var([np.array(
            np.log(struct.A_dict_non_sis[keyA]['length_birth'].loc[max_gen] / struct.x_avg) - np.log(
                struct.B_dict_non_sis[keyB]['length_birth'].loc[max_gen] / struct.x_avg)) for keyA, keyB in
            zip(struct.A_dict_non_sis.keys(), struct.B_dict_non_sis.keys()) if
            min(len(struct.A_dict_non_sis[keyA]['generationtime']),
                len(struct.B_dict_non_sis[keyB]['generationtime'])) > max_gen])

        non_sis_diff_array = [gen_time_array, alpha_array, phi_array, div_ratios, birth_lengths]
        non_sis_diff_array_var_gen.append(np.array(non_sis_diff_array))

        # Control
        gen_time_array = np.var([np.array(
            struct.A_dict_both[keyA]['generationtime'].loc[max_gen] - struct.B_dict_both[keyB]['generationtime'].loc[
                                                                       max_gen]) for keyA, keyB in
            zip(struct.A_dict_both.keys(), struct.B_dict_both.keys()) if
            min(len(struct.A_dict_both[keyA]['generationtime']),
                len(struct.B_dict_both[keyB]['generationtime'])) > max_gen])
        alpha_array = np.var([np.array(
            struct.A_dict_both[keyA]['growth_length'].loc[max_gen] - struct.B_dict_both[keyB]['growth_length'].loc[
                                                                      max_gen]) for keyA, keyB in
            zip(struct.A_dict_both.keys(), struct.B_dict_both.keys()) if
            min(len(struct.A_dict_both[keyA]['generationtime']),
                len(struct.B_dict_both[keyB]['generationtime'])) > max_gen])
        phi_array = np.var([np.array(
            struct.A_dict_both[keyA]['generationtime'].loc[max_gen] * struct.A_dict_both[keyA]['growth_length'].loc[
                                                                       max_gen] - struct.B_dict_both[keyB][
                                                                                       'generationtime'].loc[max_gen] *
            struct.B_dict_both[keyB]['growth_length'].loc[max_gen]) for keyA, keyB in
            zip(struct.A_dict_both.keys(), struct.B_dict_both.keys()) if
            min(len(struct.A_dict_both[keyA]['generationtime']),
                len(struct.B_dict_both[keyB]['generationtime'])) > max_gen])
        div_ratios = np.var([np.array(
            np.log(struct.A_dict_both[keyA]['division_ratios__f_n'].loc[max_gen]) - np.log(
                struct.B_dict_both[keyB]['division_ratios__f_n'].loc[max_gen])) for keyA, keyB in
            zip(struct.A_dict_both.keys(), struct.B_dict_both.keys()) if
            min(len(struct.A_dict_both[keyA]['generationtime']),
                len(struct.B_dict_both[keyB]['generationtime'])) > max_gen])
        birth_lengths = np.var([np.array(
            np.log(struct.A_dict_both[keyA]['length_birth'].loc[max_gen] / struct.x_avg) - np.log(
                struct.B_dict_both[keyB]['length_birth'].loc[max_gen] / struct.x_avg)) for keyA, keyB in
            zip(struct.A_dict_both.keys(), struct.B_dict_both.keys()) if
            min(len(struct.A_dict_both[keyA]['generationtime']),
                len(struct.B_dict_both[keyB]['generationtime'])) > max_gen])

        both_diff_array = [gen_time_array, alpha_array, phi_array, div_ratios, birth_lengths]
        both_diff_array_var_gen.append(np.array(both_diff_array))
        """

    # Name of the cycle parameters for labels
    param_array = [r'Generation time ($T$)', r'Growth rate ($\alpha$)', r'Accumulation Exponent ($\phi$)', r'Log of Division Ratio ($\log(f)$)', r'Normalized Birth Size ($\log(\frac{x}{x^*})$)']

    # separates up to and only options for data
    filename = [str(param_array[ind]) + ', ' + 'using up to said generation ' for ind in range(len(param_array))]
    title = filename

    for ind in range(len(param_array)):
        # # Create the labels for each data set
        # label_sis = "Sister" # GetLabels(sis_diff_array_var[ind], "Sister")
        # label_non = "Non-Sister" # GetLabels(non_sis_diff_array_var[ind], "Non-Sister")
        # label_both = "Control" # GetLabels(both_diff_array_var[ind], "Control")
        # 
        # # Create the x-axis label
        # xlabel = 'generation number'
        # ylabel = 'Variance of difference in cycle parameter '+str(param_array[ind])
        # 
        # # Graph the Histograms and save them
        # HistOfVarPerGen([sis_diff_array_var[dex][ind] for dex in range(len(sis_diff_array_var))], [non_sis_diff_array_var[dex][ind] for dex in range(len(non_sis_diff_array_var))], [both_diff_array_var[dex][ind] for dex in range(len(both_diff_array_var))],
        #                 label_sis, label_non, label_both, xlabel, filename[ind], title[ind], ylabel)
        

        sis_log = [np.log(sis_diff_array_var[dex][ind]-sis_diff_array_var[0][ind]) for dex in range(1, len(sis_diff_array_var))]
        non_sis_log = [np.log(non_sis_diff_array_var[dex][ind]-non_sis_diff_array_var[0][ind]) for dex in range(1, len(non_sis_diff_array_var))]
        both_log = [np.log(both_diff_array_var[dex][ind]-both_diff_array_var[0][ind]) for dex in range(1, len(both_diff_array_var))]

        gamma_sis = []
        gamma_non_sis = []
        gamma_both = []
        intercept_sis = []
        intercept_non_sis = []
        intercept_both = []
        for mingen in range(18):
            sis_log = [np.log(sis_diff_array_var[dex][ind] - sis_diff_array_var[0][ind]) for dex in range(1, len(sis_diff_array_var))]
            non_sis_log = [np.log(non_sis_diff_array_var[dex][ind] - non_sis_diff_array_var[0][ind]) for dex in range(1, len(non_sis_diff_array_var))]
            both_log = [np.log(both_diff_array_var[dex][ind] - both_diff_array_var[0][ind]) for dex in range(1, len(both_diff_array_var))]

            print(mingen)
            sis_log = sis_log[mingen:]
            non_sis_log = non_sis_log[mingen:]
            both_log = both_log[mingen:]

            print(len(sis_log))
            print(len(non_sis_log))
            print(len(both_log))

            slope_sis, intercep_sis, r_value_sis, p_value_sis, std_err_sis = stats.linregress(np.log(np.arange(1+mingen, len(sis_log) + 1+mingen)),
                                                                                               sis_log)
            slope_non_sis, intercep_non_sis, r_value_non_sis, p_value_non_sis, std_err_non_sis = stats.linregress(np.log(np.arange(1+mingen,
                                                                                                        len(non_sis_log) + 1+mingen)), non_sis_log)
            slope_both, intercep_both, r_value_both, p_value_both, std_err_both = stats.linregress(np.log(np.arange(1+mingen, len(both_log) +
                                                                                                                     1+mingen)), both_log)

            gamma_sis.append(slope_sis)
            gamma_non_sis.append(slope_non_sis)
            gamma_both.append(slope_both)

            intercept_sis.append(intercep_sis)
            intercept_non_sis.append(intercep_non_sis)
            intercept_both.append(intercep_both)

        # for the slopes
        # plt.plot(np.arange(2, 20), gamma_sis, label=r'Sis $\gamma$', marker='v')
        # plt.plot(np.arange(2, 20), gamma_non_sis, label=r'NonSis $\gamma$', marker='v')
        # plt.plot(np.arange(2, 20), gamma_both, label=r'Control $\gamma$', marker='v')

        # for the intercepts
        plt.plot(np.arange(2, 20), intercept_sis, label=r'Sis intercept', marker='v')
        plt.plot(np.arange(2, 20), intercept_non_sis, label=r'NonSis intercept', marker='v')
        plt.plot(np.arange(2, 20), intercept_both, label=r'Control intercept', marker='v')

        plt.legend()
        plt.xlabel('What generation the linear fit started from, (gen. 1 is sisters)')
        plt.ylabel(r'intercept values')
        plt.title(param_array[ind])
        plt.grid(True)
        # plt.show()
        plt.savefig('IMPROVED intercept per generation, {}, minus initial variance.png'.format(param_array[ind]), dpi=300)
        plt.close()


        # plt.plot(np.log(np.arange(1, len(sis_log) + 1)), sis_log, color='blue', label=r'sister $\gamma=${:.2e}'.format(slope_sis), marker='.')
        # plt.plot(np.log(np.arange(1, len(sis_log) + 1)), non_sis_log, color='orange', label=r'NS $\gamma=${:.2e}'.format(slope_non_sis), marker='.')
        # plt.plot(np.log(np.arange(1, len(sis_log) + 1)), both_log, color='green', label=r'Control $\gamma=${:.2e}'.format(slope_both), marker='.')
        # plt.plot(np.log(np.arange(1, len(sis_log) + 1)), intercept_sis + slope_sis*np.log(np.arange(1, len(sis_log) + 1)), color='blue', alpha=.5,
        #          linewidth='3')
        # plt.plot(np.log(np.arange(1, len(sis_log) + 1)), intercept_non_sis + slope_non_sis*np.log(np.arange(1, len(non_sis_log) + 1)), color='orange', alpha=.5, linewidth='3')
        # plt.plot(np.log(np.arange(1, len(sis_log) + 1)), intercept_both + slope_both*np.log(np.arange(1, len(both_log) + 1)), color='green', alpha=.5, linewidth='3')
        # plt.legend()
        # plt.xlabel('log(generation)')
        # plt.ylabel('log(acc. var. of diff. in {})'.format(param_array[ind]))
        # plt.show()
        # plt.savefig('minus initial variance, DATA scaling index for '+str(param_array[ind]), dpi=300)
        # plt.close()


    # # Parameter, number of boxes in histogram, that I found to be the most insightful
    # Nbins = None
    #
    # # separates up to and only options for data
    # filename = [str(param_array[ind]) + ', ' + 'using only one generation ' for ind in range(len(param_array))]
    # title = filename
    #
    # for ind in range(len(param_array)):
    #     # Create the labels for each data set
    #     label_sis = "Sister" # GetLabels(sis_diff_array_var_gen[ind], "Sister")
    #     label_non = "Non-Sister" # GetLabels(non_sis_diff_array_var_gen[ind], "Non-Sister")
    #     label_both = "Control" # GetLabels(both_diff_array_var_gen[ind], "Control")
    #
    #     # Create the x-axis label
    #     xlabel = 'generation number'
    #     ylabel = 'Variance of difference in cycle parameter ' + str(param_array[ind])
    #
    #     # Graph the Histograms and save them
    #     HistOfVarPerGen([sis_diff_array_var_gen[dex][ind] for dex in range(len(sis_diff_array_var_gen))], [non_sis_diff_array_var_gen[dex][ind] for dex in range(len(non_sis_diff_array_var_gen))], [both_diff_array_var_gen[dex][ind] for dex in range(len(both_diff_array_var_gen))], label_sis, label_non,
    #                  label_both, xlabel, filename[ind], title[ind], ylabel)


if __name__ == '__main__':
    main()
