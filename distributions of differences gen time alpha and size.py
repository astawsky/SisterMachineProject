
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

    """
    sis_diff_array = [[struct.A_dict_sis[IDA]['generationtime'][:min(len(struct.A_dict_sis[IDA]), len(struct.B_dict_sis[IDB]))]-
     struct.B_dict_sis[IDB]['generationtime'][:min(len(struct.A_dict_sis[IDA]), len(struct.B_dict_sis[IDB]))] for IDA, IDB in
        zip(struct.A_dict_sis.keys(), struct.B_dict_sis.keys())],
                      [struct.A_dict_sis[IDA]['growth_length'][:min(len(struct.A_dict_sis[IDA]),
                                                                                                                len(struct.B_dict_sis[IDB]))]-
     struct.B_dict_sis[IDB]['growth_length'][:min(len(struct.A_dict_sis[IDA]), len(struct.B_dict_sis[IDB]))] for IDA, IDB in
        zip(struct.A_dict_sis.keys(), struct.B_dict_sis.keys())], [struct.A_dict_sis[IDA]['length_birth'][:min(len(struct.A_dict_sis[IDA]),
                                                                                                               len(struct.B_dict_sis[IDB]))]-
     struct.B_dict_sis[IDB]['length_birth'][:min(len(struct.A_dict_sis[IDA]), len(struct.B_dict_sis[IDB]))] for IDA, IDB in
        zip(struct.A_dict_sis.keys(), struct.B_dict_sis.keys())]]
    non_sis_diff_array = [[struct.A_dict_non_sis[IDA]['generationtime'][:min(len(struct.A_dict_non_sis[IDA]), len(struct.B_dict_non_sis[IDB]))] -
                       struct.B_dict_non_sis[IDB]['generationtime'][:min(len(struct.A_dict_non_sis[IDA]), len(struct.B_dict_non_sis[IDB]))] for IDA, IDB in
                       zip(struct.A_dict_non_sis.keys(), struct.B_dict_non_sis.keys())],
                      [struct.A_dict_non_sis[IDA]['growth_length'][:min(len(struct.A_dict_non_sis[IDA]), len(struct.B_dict_non_sis[IDB]))] -
                       struct.B_dict_non_sis[IDB]['growth_length'][:min(len(struct.A_dict_non_sis[IDA]), len(struct.B_dict_non_sis[IDB]))] for IDA, IDB in
                       zip(struct.A_dict_non_sis.keys(), struct.B_dict_non_sis.keys())],
                      [struct.A_dict_non_sis[IDA]['length_birth'][:min(len(struct.A_dict_non_sis[IDA]), len(struct.B_dict_non_sis[IDB]))] -
                       struct.B_dict_non_sis[IDB]['length_birth'][:min(len(struct.A_dict_non_sis[IDA]), len(struct.B_dict_non_sis[IDB]))] for IDA, IDB in
                       zip(struct.A_dict_non_sis.keys(), struct.B_dict_non_sis.keys())]]
    both_diff_array = [[struct.A_dict_both[IDA]['generationtime'][:min(len(struct.A_dict_both[IDA]), len(struct.B_dict_both[IDB]))] -
                       struct.B_dict_both[IDB]['generationtime'][:min(len(struct.A_dict_both[IDA]), len(struct.B_dict_both[IDB]))] for IDA, IDB in
                       zip(struct.A_dict_both.keys(), struct.B_dict_both.keys())],
                      [struct.A_dict_both[IDA]['growth_length'][:min(len(struct.A_dict_both[IDA]), len(struct.B_dict_both[IDB]))] -
                       struct.B_dict_both[IDB]['growth_length'][:min(len(struct.A_dict_both[IDA]), len(struct.B_dict_both[IDB]))] for IDA, IDB in
                       zip(struct.A_dict_both.keys(), struct.B_dict_both.keys())],
                      [struct.A_dict_both[IDA]['length_birth'][:min(len(struct.A_dict_both[IDA]), len(struct.B_dict_both[IDB]))] -
                       struct.B_dict_both[IDB]['length_birth'][:min(len(struct.A_dict_both[IDA]), len(struct.B_dict_both[IDB]))] for IDA, IDB in
                       zip(struct.A_dict_both.keys(), struct.B_dict_both.keys())]]

    Nbins = None

    param_array = [r'$\tau$ generationtime', r'$\alpha$ growth rate', r'$ln(\frac{x}{x^*})$ Normalized birth size']
    filename_array = ['generationtime', 'growth rate', 'Normalized birth size']

    range_array = [[-.5,.5], [-1,1], [-1.3,1.3]]

    for param_ind in range(3):

        for j in range(len(sis_diff_array[param_ind])):
            sis_diff_array[param_ind][j] = np.array(sis_diff_array[param_ind][j])
        for j in range(len(non_sis_diff_array[param_ind])):
            non_sis_diff_array[param_ind][j] = np.array(non_sis_diff_array[param_ind][j])
        for j in range(len(both_diff_array[param_ind])):
            both_diff_array[param_ind][j] = np.array(both_diff_array[param_ind][j])

        sis_diff_array[param_ind] = np.concatenate(sis_diff_array[param_ind])
        non_sis_diff_array[param_ind] = np.concatenate(non_sis_diff_array[param_ind])
        both_diff_array[param_ind] = np.concatenate(both_diff_array[param_ind])

        sis_label = 'Sister ' + r'$\mu=$' + '{:.2e}'.format(np.mean(sis_diff_array[param_ind])) + r', $\sigma=$' + '{:.2e}'.format(
            np.std(sis_diff_array[param_ind]))
        non_label = 'Non Sister ' + r'$\mu=$' + '{:.2e}'.format(
            np.mean(non_sis_diff_array[param_ind])) + r', $\sigma=$' + '{:.2e}'.format(np.std(non_sis_diff_array[param_ind]))
        both_label = 'Control ' + r'$\mu=$' + '{:.2e}'.format(np.mean(both_diff_array[param_ind])) + r', $\sigma=$' + '{:.2e}'.format(
            np.std(both_diff_array[param_ind]))
    
        arr_sis = plt.hist(x=sis_diff_array[param_ind], label=sis_label, weights=np.ones_like(sis_diff_array[param_ind]) / float(len(sis_diff_array[param_ind])), bins=Nbins, range=range_array[param_ind])
        arr_non_sis = plt.hist(x=non_sis_diff_array[param_ind], label=non_label,
                               weights=np.ones_like(non_sis_diff_array[param_ind]) / float(len(non_sis_diff_array[param_ind])), bins=Nbins, range=range_array[param_ind])
        arr_both = plt.hist(x=both_diff_array[param_ind], label=both_label, weights=np.ones_like(both_diff_array[param_ind]) / float(len(
            both_diff_array[param_ind])), bins=Nbins, range=range_array[param_ind])
        plt.close()
    
        # print('arr_sis[0]:', arr_sis[0])
        # print('arr_non_sis[0]:', arr_non_sis[0])
        # print('arr_both[0]:', arr_both[0])
    
        plt.plot(np.array([(arr_sis[1][l] + arr_sis[1][l + 1]) / 2. for l in range(len(arr_sis[1]) - 1)]), arr_sis[0], label=sis_label, marker='.')
        plt.plot(np.array([(arr_non_sis[1][l] + arr_non_sis[1][l + 1]) / 2. for l in range(len(arr_non_sis[1]) - 1)]), arr_non_sis[0], label=non_label,
                 marker='.')
        plt.plot(np.array([(arr_both[1][l] + arr_both[1][l + 1]) / 2. for l in range(len(arr_both[1]) - 1)]), arr_both[0], label=both_label, marker='.')
        plt.xlabel(r'diff. of '+param_array[param_ind])
        plt.ylabel('PDF (Weighted Histogram)')
        plt.legend()
        # plt.show()
        plt.savefig('pdf of diff. in '+filename_array[param_ind]+' pooled across all generations available.png', dpi=300)
        plt.close()
    """

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

        gen_time_array = [np.sum(struct.A_dict_sis[keyA]['generationtime'].loc[:max_gen] - struct.B_dict_sis[keyB]['generationtime'].loc[:max_gen]) for keyA, keyB
             in zip(struct.A_dict_sis.keys(), struct.B_dict_sis.keys()) if
             min(len(struct.A_dict_sis[keyA]['generationtime']), len(struct.B_dict_sis[keyB]['generationtime'])) > max_gen]
        alpha_array = [np.sum(struct.A_dict_sis[keyA]['growth_length'].loc[:max_gen] - struct.B_dict_sis[keyB]['growth_length'].loc[:max_gen]) for keyA, keyB in
             zip(struct.A_dict_sis.keys(), struct.B_dict_sis.keys()) if
             min(len(struct.A_dict_sis[keyA]['generationtime']), len(struct.B_dict_sis[keyB]['generationtime'])) > max_gen]
        phi_array = [np.sum(struct.A_dict_sis[keyA]['generationtime'].loc[:max_gen] * struct.A_dict_sis[keyA]['growth_length'].loc[:max_gen] -
                                   struct.B_dict_sis[keyB]['generationtime'].loc[:max_gen] * struct.B_dict_sis[keyB]['growth_length'].loc[:max_gen])
                            for keyA, keyB in zip(struct.A_dict_sis.keys(), struct.B_dict_sis.keys()) if
                            min(len(struct.A_dict_sis[keyA]['generationtime']), len(struct.B_dict_sis[keyB]['generationtime'])) > max_gen]
        div_ratios = [np.sum(np.log(struct.A_dict_sis[keyA]['division_ratios__f_n'].loc[:max_gen]) - np.log(
            struct.B_dict_sis[keyB]['division_ratios__f_n'].loc[:max_gen])) for keyA, keyB in zip(struct.A_dict_sis.keys(), struct.B_dict_sis.keys())
                             if min(len(struct.A_dict_sis[keyA]['generationtime']), len(struct.B_dict_sis[keyB]['generationtime'])) > max_gen]
        birth_lengths = [np.sum(np.log(struct.A_dict_sis[keyA]['length_birth'].loc[:max_gen] / struct.x_avg) - np.log(
            struct.B_dict_sis[keyB]['length_birth'].loc[:max_gen] / struct.x_avg)) for keyA, keyB in
                                zip(struct.A_dict_sis.keys(), struct.B_dict_sis.keys()) if
                                min(len(struct.A_dict_sis[keyA]['generationtime']), len(struct.B_dict_sis[keyB]['generationtime'])) > max_gen]

        sis_diff_array = [gen_time_array, alpha_array, phi_array, div_ratios, birth_lengths]
        sis_diff_array_var.append(np.array(sis_diff_array))

        # Non-Sister
        gen_time_array = [np.sum(
            struct.A_dict_non_sis[keyA]['generationtime'].loc[:max_gen] - struct.B_dict_non_sis[keyB]['generationtime'].loc[
                                                                          :max_gen]) for keyA, keyB in
            zip(struct.A_dict_non_sis.keys(), struct.B_dict_non_sis.keys()) if
            min(len(struct.A_dict_non_sis[keyA]['generationtime']), len(struct.B_dict_non_sis[keyB]['generationtime'])) > max_gen]
        alpha_array = [np.sum(
            struct.A_dict_non_sis[keyA]['growth_length'].loc[:max_gen] - struct.B_dict_non_sis[keyB]['growth_length'].loc[
                                                                         :max_gen]) for keyA, keyB in
            zip(struct.A_dict_non_sis.keys(), struct.B_dict_non_sis.keys()) if
            min(len(struct.A_dict_non_sis[keyA]['generationtime']), len(struct.B_dict_non_sis[keyB]['generationtime'])) > max_gen]
        phi_array = [np.sum(
            struct.A_dict_non_sis[keyA]['generationtime'].loc[:max_gen] * struct.A_dict_non_sis[keyA]['growth_length'].loc[
                                                                          :max_gen] - struct.B_dict_non_sis[keyB][
                                                                                          'generationtime'].loc[:max_gen] *
            struct.B_dict_non_sis[keyB]['growth_length'].loc[:max_gen]) for keyA, keyB in
            zip(struct.A_dict_non_sis.keys(), struct.B_dict_non_sis.keys()) if
            min(len(struct.A_dict_non_sis[keyA]['generationtime']), len(struct.B_dict_non_sis[keyB]['generationtime'])) > max_gen]
        div_ratios = [np.sum(
            np.log(struct.A_dict_non_sis[keyA]['division_ratios__f_n'].loc[:max_gen]) - np.log(
                struct.B_dict_non_sis[keyB]['division_ratios__f_n'].loc[:max_gen])) for keyA, keyB in
            zip(struct.A_dict_non_sis.keys(), struct.B_dict_non_sis.keys()) if
            min(len(struct.A_dict_non_sis[keyA]['generationtime']), len(struct.B_dict_non_sis[keyB]['generationtime'])) > max_gen]
        birth_lengths = [np.sum(
            np.log(struct.A_dict_non_sis[keyA]['length_birth'].loc[:max_gen] / struct.x_avg) - np.log(
                struct.B_dict_non_sis[keyB]['length_birth'].loc[:max_gen] / struct.x_avg)) for keyA, keyB in
            zip(struct.A_dict_non_sis.keys(), struct.B_dict_non_sis.keys()) if
            min(len(struct.A_dict_non_sis[keyA]['generationtime']), len(struct.B_dict_non_sis[keyB]['generationtime'])) > max_gen]

        non_sis_diff_array = [gen_time_array, alpha_array, phi_array, div_ratios, birth_lengths]
        non_sis_diff_array_var.append(np.array(non_sis_diff_array))

        # Control
        gen_time_array = [np.sum(
            struct.A_dict_both[keyA]['generationtime'].loc[:max_gen] - struct.B_dict_both[keyB]['generationtime'].loc[
                                                                       :max_gen]) for keyA, keyB in
            zip(struct.A_dict_both.keys(), struct.B_dict_both.keys()) if
            min(len(struct.A_dict_both[keyA]['generationtime']), len(struct.B_dict_both[keyB]['generationtime'])) > max_gen]
        alpha_array = [np.sum(
            struct.A_dict_both[keyA]['growth_length'].loc[:max_gen] - struct.B_dict_both[keyB]['growth_length'].loc[
                                                                      :max_gen]) for keyA, keyB in
            zip(struct.A_dict_both.keys(), struct.B_dict_both.keys()) if
            min(len(struct.A_dict_both[keyA]['generationtime']), len(struct.B_dict_both[keyB]['generationtime'])) > max_gen]
        phi_array = [np.sum(
            struct.A_dict_both[keyA]['generationtime'].loc[:max_gen] * struct.A_dict_both[keyA]['growth_length'].loc[
                                                                       :max_gen] - struct.B_dict_both[keyB][
                                                                                       'generationtime'].loc[:max_gen] *
            struct.B_dict_both[keyB]['growth_length'].loc[:max_gen]) for keyA, keyB in
            zip(struct.A_dict_both.keys(), struct.B_dict_both.keys()) if
            min(len(struct.A_dict_both[keyA]['generationtime']), len(struct.B_dict_both[keyB]['generationtime'])) > max_gen]
        div_ratios = [np.sum(
            np.log(struct.A_dict_both[keyA]['division_ratios__f_n'].loc[:max_gen]) - np.log(
                struct.B_dict_both[keyB]['division_ratios__f_n'].loc[:max_gen])) for keyA, keyB in
            zip(struct.A_dict_both.keys(), struct.B_dict_both.keys()) if
            min(len(struct.A_dict_both[keyA]['generationtime']), len(struct.B_dict_both[keyB]['generationtime'])) > max_gen]
        birth_lengths = [np.sum(
            np.log(struct.A_dict_both[keyA]['length_birth'].loc[:max_gen] / struct.x_avg) - np.log(
                struct.B_dict_both[keyB]['length_birth'].loc[:max_gen] / struct.x_avg)) for keyA, keyB in
            zip(struct.A_dict_both.keys(), struct.B_dict_both.keys()) if
            min(len(struct.A_dict_both[keyA]['generationtime']), len(struct.B_dict_both[keyB]['generationtime'])) > max_gen]

        both_diff_array = [gen_time_array, alpha_array, phi_array, div_ratios, birth_lengths]
        both_diff_array_var.append(np.array(both_diff_array))

    # Name of the cycle parameters for labels
    param_array = [r'Generation time ($T$)', r'Growth rate ($\alpha$)', r'Accumulation Exponent ($\phi$)', r'Log of Division Ratio ($\log(f)$)', r'Normalized Birth Size ($\log(\frac{x}{x^*})$)']
    range_array = [[-.5,.5], [-.7,.7], [-1,1], [-1,1], [-1.5,1.5], [-1.5,1.5], [-1.7,1.7], [-1.7,1.7], [-2, 2], [-2, 2], [-2.5, 2.5], [-2.5, 2.5]
                   , [-2.5, 2.5], [-3,3], [-3,3], [-3,3], [-3,3], [-3.5,3.5], [-3.5,3.5], [-4,4]]
    Nbins = 20
    filename_array = ['generationtime', 'growth rate', 'Normalized birth size']
    alph_array = [1/b for b in range(1, 21)]

    param_ind = 0
    arr_both_array =[]
    for gen in range(20):
        sis_diff_array = sis_diff_array_var[gen][param_ind]
        non_sis_diff_array = non_sis_diff_array_var[gen][param_ind]
        both_diff_array = both_diff_array_var[gen][param_ind]

        print(len(sis_diff_array))

        sis_label = 'Sister ' + r'$\mu=$' + '{:.2e}'.format(np.mean(sis_diff_array)) + r', $\sigma=$' + '{:.2e}'.format(
            np.std(sis_diff_array))
        non_label = 'Non Sister ' + r'$\mu=$' + '{:.2e}'.format(
            np.mean(non_sis_diff_array)) + r', $\sigma=$' + '{:.2e}'.format(np.std(non_sis_diff_array))
        both_label = 'Control ' + r'$\mu=$' + '{:.2e}'.format(np.mean(both_diff_array)) + r', $\sigma=$' + '{:.2e}'.format(
            np.std(both_diff_array))

        # arr_sis = plt.hist(x=sis_diff_array, label=sis_label,
        #                    weights=np.ones_like(sis_diff_array) / float(len(sis_diff_array)), bins=Nbins,
        #                    range=range_array[gen])
        # arr_non_sis = plt.hist(x=non_sis_diff_array, label=non_label,
        #                        weights=np.ones_like(non_sis_diff_array) / float(len(non_sis_diff_array)), bins=Nbins,
        #                        range=range_array[gen])
        # arr_both = plt.hist(x=both_diff_array, label=both_label, weights=np.ones_like(both_diff_array) / float(len(
        #     both_diff_array)), bins=Nbins, range=range_array[gen])
        arr_both = plt.hist(x=both_diff_array, label=both_label, weights=np.ones_like(both_diff_array) / float(len(
            both_diff_array)), bins=Nbins, range=[-4,4])
        arr_both_array.append(arr_both)
        plt.close()

        # print('arr_sis[0]:', arr_sis[0])
        # print('arr_non_sis[0]:', arr_non_sis[0])
        # print('arr_both[0]:', arr_both[0])

        # plt.plot(np.array([(arr_sis[1][l] + arr_sis[1][l + 1]) / 2. for l in range(len(arr_sis[1]) - 1)]), arr_sis[0], label=sis_label, marker='.')
        # plt.plot(np.array([(arr_non_sis[1][l] + arr_non_sis[1][l + 1]) / 2. for l in range(len(arr_non_sis[1]) - 1)]), arr_non_sis[0],
        #          label=non_label,
        #          marker='.')
        # plt.plot(np.array([(arr_both[1][l] + arr_both[1][l + 1]) / 2. for l in range(len(arr_both[1]) - 1)]), arr_both[0], label=both_label,
        #          marker='.')
        # plt.xlabel(r'acc. diff. of ' + param_array[param_ind])
        # plt.ylabel('PDF (Weighted Histogram)' + ', gen '+str(gen))
        # plt.legend()
        # # plt.show()
        # plt.savefig('pdf of acc. diff. in ' + filename_array[param_ind] + ', gen '+str(gen)+'.png', dpi=300)
        # plt.close()

    for k in range(0, len(arr_both_array), 5):
        plt.plot(np.array([(arr_both_array[k][1][l] + arr_both_array[k][1][l + 1]) / 2. for l in range(len(arr_both_array[k][1]) - 1)]), arr_both_array[k][0],
                 marker='.', alpha=1/((k+1)/5), color='green')
    plt.xlabel(r'acc. diff. of ' + param_array[param_ind])
    plt.ylabel('PDF (Weighted Histogram)' + ' all gens')
    plt.title('Control')
    # plt.legend()
    plt.show()
    # plt.savefig('pdf of acc. diff. in ' + filename_array[param_ind] + ', gen ' + str(gen) + '.png', dpi=300)
    # plt.close()


if __name__ == '__main__':
    main()
