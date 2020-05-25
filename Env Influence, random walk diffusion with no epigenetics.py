from __future__ import print_function

import numpy as np

import matplotlib.pyplot as plt

import pickle

import scipy.stats as stats


def HistOfVarPerGen(dist_sis, dist_non, dist_both, label_sis, label_non, label_both, xlabel, filename, title, ylabel):

    x = np.arange(len(dist_sis))
    x = x[:, np.newaxis]
    a_sis, _, _, _ = np.linalg.lstsq(x, dist_sis-dist_sis[0], rcond=None)
    a_non, _, _, _ = np.linalg.lstsq(x, dist_non-dist_non[0], rcond=None)
    a_both, _, _, _ = np.linalg.lstsq(x, dist_both-dist_both[0], rcond=None)

    plt.plot(dist_sis,
             label=label_sis + r' {}+{:.2e}*(gen. num.)'.format(0, a_sis[0]), marker='.', color='b')
    plt.plot(dist_non,
             label=label_non + r' {}+{:.2e}*(gen. num.)'.format(0, a_non[0]), marker='.', color='orange')
    plt.plot(dist_both, label=label_both + r' {}+{:.2e}*(gen. num.)'.format(0, a_both[0]),
             marker='.', color='green')
    plt.plot(x, dist_sis[0]+a_sis * x, alpha=.5, linewidth='3')
    plt.plot(x, dist_non[0]+a_non * x, alpha=.5, linewidth='3')
    plt.plot(x, dist_both[0]+a_both * x, alpha=.5, linewidth='3')
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    # plt.show()
    plt.savefig(filename + '.png', dpi=300)
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

    # Where we will put all the variances for up to and then just generation
    sis_diff_array_var = []
    non_sis_diff_array_var = []
    both_diff_array_var = []

    # max_gen is the generation we want to look at, here we go up to 9 generations
    how_many = 20
    start_gen = 5
    for max_gen in range(start_gen, start_gen+how_many):

        # # Format the cycle parameters like in Lee's paper, and in the POWERPOINT
        # Sister

        gen_time_array = np.var(
            [np.sum(struct.A_dict_sis[keyA]['generationtime'].loc[start_gen:max_gen] - struct.B_dict_sis[keyB]['generationtime'].loc[start_gen:max_gen]) for keyA, keyB
             in zip(struct.A_dict_sis.keys(), struct.B_dict_sis.keys()) if
             min(len(struct.A_dict_sis[keyA]['generationtime']), len(struct.B_dict_sis[keyB]['generationtime'])) > max_gen])
        alpha_array = np.var(
            [np.sum(struct.A_dict_sis[keyA]['growth_length'].loc[start_gen:max_gen] - struct.B_dict_sis[keyB]['growth_length'].loc[start_gen:max_gen]) for keyA, keyB in
             zip(struct.A_dict_sis.keys(), struct.B_dict_sis.keys()) if
             min(len(struct.A_dict_sis[keyA]['generationtime']), len(struct.B_dict_sis[keyB]['generationtime'])) > max_gen])
        phi_array = np.var([np.sum(struct.A_dict_sis[keyA]['generationtime'].loc[start_gen:max_gen] * struct.A_dict_sis[keyA]['growth_length'].loc[start_gen:max_gen] -
                                   struct.B_dict_sis[keyB]['generationtime'].loc[start_gen:max_gen] * struct.B_dict_sis[keyB]['growth_length'].loc[start_gen:max_gen])
                            for keyA, keyB in zip(struct.A_dict_sis.keys(), struct.B_dict_sis.keys()) if
                            min(len(struct.A_dict_sis[keyA]['generationtime']), len(struct.B_dict_sis[keyB]['generationtime'])) > max_gen])
        div_ratios = np.var([np.sum(np.log(struct.A_dict_sis[keyA]['division_ratios__f_n'].loc[start_gen:max_gen]) - np.log(
            struct.B_dict_sis[keyB]['division_ratios__f_n'].loc[start_gen:max_gen])) for keyA, keyB in zip(struct.A_dict_sis.keys(), struct.B_dict_sis.keys())
                             if min(len(struct.A_dict_sis[keyA]['generationtime']), len(struct.B_dict_sis[keyB]['generationtime'])) > max_gen])
        birth_lengths = np.var([np.sum(np.log(struct.A_dict_sis[keyA]['length_birth'].loc[start_gen:max_gen] / struct.x_avg) - np.log(
            struct.B_dict_sis[keyB]['length_birth'].loc[start_gen:max_gen] / struct.x_avg)) for keyA, keyB in
                                zip(struct.A_dict_sis.keys(), struct.B_dict_sis.keys()) if
                                min(len(struct.A_dict_sis[keyA]['generationtime']), len(struct.B_dict_sis[keyB]['generationtime'])) > max_gen])

        sis_diff_array = [gen_time_array, alpha_array, phi_array, div_ratios, birth_lengths]
        sis_diff_array_var.append(np.array(sis_diff_array))

        # Non-Sister
        gen_time_array = np.var([np.sum(
            struct.A_dict_non_sis[keyA]['generationtime'].loc[start_gen:max_gen] - struct.B_dict_non_sis[keyB]['generationtime'].loc[
                                                                          :max_gen]) for keyA, keyB in
            zip(struct.A_dict_non_sis.keys(), struct.B_dict_non_sis.keys()) if
            min(len(struct.A_dict_non_sis[keyA]['generationtime']), len(struct.B_dict_non_sis[keyB]['generationtime'])) > max_gen])
        alpha_array = np.var([np.sum(
            struct.A_dict_non_sis[keyA]['growth_length'].loc[start_gen:max_gen] - struct.B_dict_non_sis[keyB]['growth_length'].loc[
                                                                         :max_gen]) for keyA, keyB in
            zip(struct.A_dict_non_sis.keys(), struct.B_dict_non_sis.keys()) if
            min(len(struct.A_dict_non_sis[keyA]['generationtime']), len(struct.B_dict_non_sis[keyB]['generationtime'])) > max_gen])
        phi_array = np.var([np.sum(
            struct.A_dict_non_sis[keyA]['generationtime'].loc[start_gen:max_gen] * struct.A_dict_non_sis[keyA]['growth_length'].loc[
                                                                          :max_gen] - struct.B_dict_non_sis[keyB][
                                                                                          'generationtime'].loc[start_gen:max_gen] *
            struct.B_dict_non_sis[keyB]['growth_length'].loc[start_gen:max_gen]) for keyA, keyB in
            zip(struct.A_dict_non_sis.keys(), struct.B_dict_non_sis.keys()) if
            min(len(struct.A_dict_non_sis[keyA]['generationtime']), len(struct.B_dict_non_sis[keyB]['generationtime'])) > max_gen])
        div_ratios = np.var([np.sum(
            np.log(struct.A_dict_non_sis[keyA]['division_ratios__f_n'].loc[start_gen:max_gen]) - np.log(
                struct.B_dict_non_sis[keyB]['division_ratios__f_n'].loc[start_gen:max_gen])) for keyA, keyB in
            zip(struct.A_dict_non_sis.keys(), struct.B_dict_non_sis.keys()) if
            min(len(struct.A_dict_non_sis[keyA]['generationtime']), len(struct.B_dict_non_sis[keyB]['generationtime'])) > max_gen])
        birth_lengths = np.var([np.sum(
            np.log(struct.A_dict_non_sis[keyA]['length_birth'].loc[start_gen:max_gen] / struct.x_avg) - np.log(
                struct.B_dict_non_sis[keyB]['length_birth'].loc[start_gen:max_gen] / struct.x_avg)) for keyA, keyB in
            zip(struct.A_dict_non_sis.keys(), struct.B_dict_non_sis.keys()) if
            min(len(struct.A_dict_non_sis[keyA]['generationtime']), len(struct.B_dict_non_sis[keyB]['generationtime'])) > max_gen])

        non_sis_diff_array = [gen_time_array, alpha_array, phi_array, div_ratios, birth_lengths]
        non_sis_diff_array_var.append(np.array(non_sis_diff_array))

        # Control
        gen_time_array = np.var([np.sum(
            struct.A_dict_both[keyA]['generationtime'].loc[start_gen:max_gen] - struct.B_dict_both[keyB]['generationtime'].loc[
                                                                       :max_gen]) for keyA, keyB in
            zip(struct.A_dict_both.keys(), struct.B_dict_both.keys()) if
            min(len(struct.A_dict_both[keyA]['generationtime']), len(struct.B_dict_both[keyB]['generationtime'])) > max_gen])
        alpha_array = np.var([np.sum(
            struct.A_dict_both[keyA]['growth_length'].loc[start_gen:max_gen] - struct.B_dict_both[keyB]['growth_length'].loc[
                                                                      :max_gen]) for keyA, keyB in
            zip(struct.A_dict_both.keys(), struct.B_dict_both.keys()) if
            min(len(struct.A_dict_both[keyA]['generationtime']), len(struct.B_dict_both[keyB]['generationtime'])) > max_gen])
        phi_array = np.var([np.sum(
            struct.A_dict_both[keyA]['generationtime'].loc[start_gen:max_gen] * struct.A_dict_both[keyA]['growth_length'].loc[
                                                                       :max_gen] - struct.B_dict_both[keyB][
                                                                                       'generationtime'].loc[start_gen:max_gen] *
            struct.B_dict_both[keyB]['growth_length'].loc[start_gen:max_gen]) for keyA, keyB in
            zip(struct.A_dict_both.keys(), struct.B_dict_both.keys()) if
            min(len(struct.A_dict_both[keyA]['generationtime']), len(struct.B_dict_both[keyB]['generationtime'])) > max_gen])
        div_ratios = np.var([np.sum(
            np.log(struct.A_dict_both[keyA]['division_ratios__f_n'].loc[start_gen:max_gen]) - np.log(
                struct.B_dict_both[keyB]['division_ratios__f_n'].loc[start_gen:max_gen])) for keyA, keyB in
            zip(struct.A_dict_both.keys(), struct.B_dict_both.keys()) if
            min(len(struct.A_dict_both[keyA]['generationtime']), len(struct.B_dict_both[keyB]['generationtime'])) > max_gen])
        birth_lengths = np.var([np.sum(
            np.log(struct.A_dict_both[keyA]['length_birth'].loc[start_gen:max_gen] / struct.x_avg) - np.log(
                struct.B_dict_both[keyB]['length_birth'].loc[start_gen:max_gen] / struct.x_avg)) for keyA, keyB in
            zip(struct.A_dict_both.keys(), struct.B_dict_both.keys()) if
            min(len(struct.A_dict_both[keyA]['generationtime']), len(struct.B_dict_both[keyB]['generationtime'])) > max_gen])

        both_diff_array = [gen_time_array, alpha_array, phi_array, div_ratios, birth_lengths]
        both_diff_array_var.append(np.array(both_diff_array))

    # Name of the cycle parameters for labels
    param_array = [r'Generation time ($T$)', r'Growth rate ($\alpha$)', r'Accumulation Exponent ($\phi$)', r'Log of Division Ratio ($\log(f)$)',
                   r'Normalized Birth Size ($\log(\frac{x}{x^*})$)']

    # separates up to and only options for data
    filename = ['Environmental Influence, random walk diffusion of '+str(param_array[ind]) for ind in range(len(param_array))]
    title = 'With no Epigenetic influence, starting from the fifth generation'

    for ind in range(len(param_array)):
        # Create the labels for each data set
        label_sis = "Sister" # GetLabels(sis_diff_array_var[ind], "Sister")
        label_non = "Non-Sister" # GetLabels(non_sis_diff_array_var[ind], "Non-Sister")
        label_both = "Control" # GetLabels(both_diff_array_var[ind], "Control")

        # Create the x-axis label
        xlabel = 'generation number'
        ylabel = 'Acc. Var. of diff. in '+str(param_array[ind])

        # Graph the Histograms and save them
        HistOfVarPerGen(np.array([sis_diff_array_var[dex][ind] for dex in range(len(sis_diff_array_var))]),
                        np.array([non_sis_diff_array_var[dex][ind] for dex in range(len(non_sis_diff_array_var))]),
                        np.array([both_diff_array_var[dex][ind] for dex in range(len(both_diff_array_var))]),
                        label_sis, label_non, label_both, xlabel, filename[ind], title, ylabel)


if __name__ == '__main__':
    main()
