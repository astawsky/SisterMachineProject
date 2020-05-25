
from __future__ import print_function

import numpy as np

import matplotlib.pyplot as plt

import pickle

import scipy.stats as stats


def Distribution(diffs_sis, diffs_non_sis, diffs_both, xlabel, Nbins, abs_diff_range):
    # PoolID == 1
    sis_label = 'Sister ' + r'$\mu=$' + '{:.2e}'.format(np.mean(diffs_sis)) + r', $\sigma=$' + '{:.2e}'.format(
        np.std(diffs_sis))
    non_label = 'Non Sister ' + r'$\mu=$' + '{:.2e}'.format(
        np.mean(diffs_non_sis)) + r', $\sigma=$' + '{:.2e}'.format(np.std(diffs_non_sis))
    both_label = 'Control ' + r'$\mu=$' + '{:.2e}'.format(np.mean(diffs_both)) + r', $\sigma=$' + '{:.2e}'.format(
        np.std(diffs_both))

    arr_sis = plt.hist(x=diffs_sis, label=sis_label, weights=np.ones_like(diffs_sis) / float(len(diffs_sis)), bins=Nbins, range=abs_diff_range)
    arr_non_sis = plt.hist(x=diffs_non_sis, label=non_label,
                           weights=np.ones_like(diffs_non_sis) / float(len(diffs_non_sis)), bins=Nbins, range=abs_diff_range)
    arr_both = plt.hist(x=diffs_both, label=both_label, weights=np.ones_like(diffs_both) / float(len(diffs_both)), bins=Nbins, range=abs_diff_range)
    plt.close()

    # print('arr_sis[0]:', arr_sis[0])
    # print('arr_non_sis[0]:', arr_non_sis[0])
    # print('arr_both[0]:', arr_both[0])

    plt.plot(np.array([(arr_sis[1][l] + arr_sis[1][l + 1]) / 2. for l in range(len(arr_sis[1]) - 1)]), arr_sis[0], label=sis_label, marker='.')
    plt.plot(np.array([(arr_non_sis[1][l] + arr_non_sis[1][l + 1]) / 2. for l in range(len(arr_non_sis[1]) - 1)]), arr_non_sis[0], label=non_label, marker='.')
    plt.plot(np.array([(arr_both[1][l] + arr_both[1][l + 1]) / 2. for l in range(len(arr_both[1]) - 1)]), arr_both[0], label=both_label, marker='.')
    plt.xlabel(xlabel)
    plt.ylabel('PDF (Weighted Histogram)')
    plt.legend()
    plt.show()


def same_plot_with_only_data_from_one_experiment(mingen=50):
    # Import the Refined Data
    pickle_in = open("metastructdata.pickle", "rb")
    struct = pickle.load(pickle_in)
    pickle_in.close()

    sis_without_environment = [struct.A_dict_sis[keyA]['generationtime'].loc[6:] - struct.B_dict_sis[keyB]['generationtime'].loc[6:] for keyA, keyB in
                               zip(struct.A_dict_sis.keys(), struct.B_dict_sis.keys()) if
                               min(len(struct.A_dict_sis[keyA]['generationtime']), len(struct.B_dict_sis[keyB][
                                                                                           'generationtime'])) >= mingen]
    non_sis_without_environment = [struct.A_dict_non_sis[keyA]['generationtime'].loc[6:] - struct.B_dict_non_sis[keyB]['generationtime'].loc[6:] for
                                   keyA, keyB in
                                   zip(struct.A_dict_non_sis.keys(), struct.B_dict_non_sis.keys()) if
                                   min(len(struct.A_dict_non_sis[keyA]['generationtime']), len(struct.B_dict_non_sis[keyB][
                                                                                                   'generationtime'])) >= mingen]
    both_without_environment = [struct.A_dict_both[keyA]['generationtime'].loc[6:] - struct.B_dict_both[keyB]['generationtime'].loc[6:] for keyA, keyB
                                in
                                zip(struct.A_dict_both.keys(), struct.B_dict_both.keys()) if
                                min(len(struct.A_dict_both[keyA]['generationtime']), len(struct.B_dict_both[keyB][
                                                                                             'generationtime'])) >= mingen]

    # put them all together no matter the environment
    all_pairings_sis = [[[sis_without_environment[exp].iloc[ind1], sis_without_environment[exp].iloc[ind2]] for ind1, ind2 in zip(range(len(
        sis_without_environment[exp]) - 1), range(1, len(sis_without_environment[exp])))] for exp in range(len(sis_without_environment))]

    # all_pairings_sis = [val for sublist in all_pairings_sis for val in sublist]

    all_pairings_non_sis = [
        [[non_sis_without_environment[exp].iloc[ind1], non_sis_without_environment[exp].iloc[ind2]] for ind1, ind2 in zip(range(len(
            non_sis_without_environment[exp]) - 1), range(1, len(non_sis_without_environment[exp])))] for exp in
        range(len(non_sis_without_environment))]

    # all_pairings_non_sis = [val for sublist in all_pairings_non_sis for val in sublist]

    all_pairings_both = [[[both_without_environment[exp].iloc[ind1], both_without_environment[exp].iloc[ind2]] for ind1, ind2 in zip(range(len(
        both_without_environment[exp]) - 1), range(1, len(both_without_environment[exp])))] for exp in range(len(both_without_environment))]
    
    print('how many sis pairs are there that are over {} generations? '.format(mingen), len(all_pairings_sis))
    print('how many non_sis pairs are there that are over {} generations? '.format(mingen), len(all_pairings_non_sis))
    print('how many both pairs are there that are over {} generations? '.format(mingen), len(all_pairings_both))

    # all_pairings_both = [val for sublist in all_pairings_both for val in sublist]
    slopes_sis = []
    pcoeff_sis = []
    slopes_non_sis = []
    pcoeff_non_sis = []
    slopes_both = []
    pcoeff_both = []
    for exp_sis, exp_non_sis, exp_both in zip(range(len(all_pairings_sis)), range(len(all_pairings_non_sis)), range(len(all_pairings_both))):
        sis_x = [all_pairings_sis[exp_sis][instance][0] for instance in range(len(all_pairings_sis[exp_sis]))]
        sis_y = [all_pairings_sis[exp_sis][instance][1] for instance in range(len(all_pairings_sis[exp_sis]))]
        non_sis_x = [all_pairings_non_sis[exp_non_sis][instance][0] for instance in range(len(all_pairings_non_sis[exp_non_sis]))]
        non_sis_y = [all_pairings_non_sis[exp_non_sis][instance][1] for instance in range(len(all_pairings_non_sis[exp_non_sis]))]
        both_x = [all_pairings_both[exp_both][instance][0] for instance in range(len(all_pairings_both[exp_both]))]
        both_y = [all_pairings_both[exp_both][instance][1] for instance in range(len(all_pairings_both[exp_both]))]

        mask = ~np.isnan(np.concatenate((np.array(sis_x), np.array(non_sis_x)), axis=0)) & ~np.isnan(np.concatenate((np.array(sis_y),
                                                                                                                     np.array(non_sis_y)), axis=0))
        mask_sis = ~np.isnan(np.array(sis_x)) & ~np.isnan(np.array(sis_y))
        mask_non_sis = ~np.isnan(np.array(non_sis_x)) & ~np.isnan(np.array(non_sis_y))
        mask_both = ~np.isnan(np.array(both_x)) & ~np.isnan(np.array(both_y))
        together_pcoeff = stats.pearsonr(np.concatenate((np.array(sis_x), np.array(non_sis_x)), axis=0)[mask], np.concatenate((np.array(sis_y),
                                                                                                                               np.array(non_sis_y)),
                                                                                                                              axis=0)[mask])
        best_fit_together = stats.linregress(np.concatenate((np.array(sis_x), np.array(non_sis_x)), axis=0)[mask], np.concatenate((np.array(sis_y),
                                                                                                                                   np.array(non_sis_y)),
                                                                                                                                  axis=0)[mask])
        best_fit_sis = stats.linregress(np.array(sis_x)[mask_sis], np.array(sis_y)[mask_sis])
        sis_pcoeff = stats.pearsonr(np.array(sis_x)[mask_sis], np.array(sis_y)[mask_sis])
        best_fit_non_sis = stats.linregress(np.array(non_sis_x)[mask_non_sis], np.array(non_sis_y)[mask_non_sis])
        non_sis_pcoeff = stats.pearsonr(np.array(non_sis_x)[mask_non_sis], np.array(non_sis_y)[mask_non_sis])
        best_fit_both = stats.linregress(np.array(both_x)[mask_both], np.array(both_y)[mask_both])
        both_pcoeff = stats.pearsonr(np.array(both_x)[mask_both], np.array(both_y)[mask_both])

        slopes_sis.append(best_fit_sis[1])
        pcoeff_sis.append(sis_pcoeff)
        slopes_non_sis.append(best_fit_non_sis[1])
        pcoeff_non_sis.append(non_sis_pcoeff)
        slopes_both.append(best_fit_both[1])
        pcoeff_both.append(both_pcoeff)
        # 
        # # For Sister and nonsister, which have the same environment
        # plt.scatter(sis_x, sis_y, label='Sisters, {} samples'.format(len(sis_x)))
        # plt.scatter(non_sis_x, non_sis_y, label='Non-Sisters, {} samples'.format(len(non_sis_x)))
        # plt.plot(np.arange(-1.2, 1.4, .2), best_fit_sis[1] + best_fit_sis[0] * np.arange(-1.2, 1.4, .2), label='Sis Pearson Coeff={:.2e}'.format(
        #     sis_pcoeff[0]), color='green')
        # plt.plot(np.arange(-1.2, 1.4, .2), best_fit_non_sis[1] + best_fit_non_sis[0] * np.arange(-1.2, 1.4, .2),
        #          label='NonSis Pearson Coeff={:.2e}'.format(
        #              non_sis_pcoeff[0]), color='black')
        # plt.plot(np.arange(-1.2, 1.4, .2), best_fit_together[1] + best_fit_together[0] * np.arange(-1.2, 1.4, .2), label='Pearson Coeff={:.2e}'.format(
        #     together_pcoeff[0]), color='magenta')
        # plt.legend()
        # plt.xlabel(r'$\Delta\tau_i$')
        # plt.ylabel(r'$\Delta\tau_{i+1}$')
        # # plt.ylim([-1.2, 1.2])
        # # plt.xlim([-1.2, 1.2])
        # plt.show()
        # # plt.savefig('PCorr S and NS gen. time diff. of consequent gens.png', dpi=300)
        # # plt.close()
        # 
        # # For control, which doesn't have the same environment
        # plt.scatter(both_x, both_y, label='Control, {} samples'.format(len(np.array(both_x)[mask_both])))
        # plt.plot(np.arange(-1.2, 1.4, .2), best_fit_both[1] + best_fit_both[0] * np.arange(-1.2, 1.4, .2), label='Control Pearson Coeff={:.2e}'.format(
        #     both_pcoeff[0]), color='black')
        # plt.legend()
        # plt.xlabel(r'$\Delta\tau_i$')
        # plt.ylabel(r'$\Delta\tau_{i+1}$')
        # # plt.ylim([-1.2, 1.2])
        # # plt.xlim([-1.2, 1.2])
        # plt.show()
        # # plt.savefig('PCorr Control gen. time diff. of consequent gens.png', dpi=300)
        # # plt.close()

    # print(np.array(slopes_sis))
    Distribution(np.array(slopes_sis), np.array(slopes_non_sis), np.array(slopes_both), xlabel='dist of slopes of consequent gen time diffs',
                 Nbins=None, abs_diff_range=np.array([-.2,.3]))
    Distribution(np.array([pcoeff_sis[blah][0] for blah in range(len(pcoeff_sis))]), np.array([pcoeff_non_sis[blah][0] for blah in range(len(
        pcoeff_non_sis))]), np.array([pcoeff_both[blah][0] for blah in range(len(pcoeff_both))]), xlabel='dist of pcoeff of consequent '
                                                                                                                    'gen time diffs',
                 Nbins=None, abs_diff_range=np.array([-.6,.2]))
    Distribution(np.array([pcoeff_sis[blah][1] for blah in range(len(pcoeff_sis))]), np.array([pcoeff_non_sis[blah][1] for blah in range(len(
        pcoeff_non_sis))]), np.array([pcoeff_both[blah][1] for blah in range(len(pcoeff_both))]), xlabel='dist of pvalues of pcoeff of consequent gen '
                                                                                                        'time diffs',
                 Nbins=None, abs_diff_range=np.array([-.2, 1.2]))


def main():
    # Import the Refined Data
    pickle_in = open("metastructdata.pickle", "rb")
    struct = pickle.load(pickle_in)
    pickle_in.close()

    sis_without_environment = [struct.A_dict_sis[keyA]['generationtime'].loc[6:] - struct.B_dict_sis[keyB]['generationtime'].loc[6:] for keyA, keyB in
     zip(struct.A_dict_sis.keys(), struct.B_dict_sis.keys()) if min(len(struct.A_dict_sis[keyA]['generationtime']), len(struct.B_dict_sis[keyB][
                                                                                                                            'generationtime'])) > 5]
    non_sis_without_environment = [struct.A_dict_non_sis[keyA]['generationtime'].loc[6:] - struct.B_dict_non_sis[keyB]['generationtime'].loc[6:] for keyA, keyB in
                               zip(struct.A_dict_non_sis.keys(), struct.B_dict_non_sis.keys()) if
                               min(len(struct.A_dict_non_sis[keyA]['generationtime']), len(struct.B_dict_non_sis[keyB][
                                                                                           'generationtime'])) > 5]
    both_without_environment = [struct.A_dict_both[keyA]['generationtime'].loc[6:] - struct.B_dict_both[keyB]['generationtime'].loc[6:] for keyA, keyB in
                               zip(struct.A_dict_both.keys(), struct.B_dict_both.keys()) if
                               min(len(struct.A_dict_both[keyA]['generationtime']), len(struct.B_dict_both[keyB][
                                                                                           'generationtime'])) > 5]

    # put them all together no matter the environment
    all_pairings_sis = [[[sis_without_environment[exp].iloc[ind1], sis_without_environment[exp].iloc[ind2]] for ind1, ind2 in zip(range(len(
        sis_without_environment[exp]) - 1), range(1, len(sis_without_environment[exp]))) ] for exp in range(len(sis_without_environment))]

    all_pairings_sis = [val for sublist in all_pairings_sis for val in sublist]

    all_pairings_non_sis = [[[non_sis_without_environment[exp].iloc[ind1], non_sis_without_environment[exp].iloc[ind2]] for ind1, ind2 in zip(range(len(
        non_sis_without_environment[exp]) - 1), range(1, len(non_sis_without_environment[exp])))] for exp in range(len(non_sis_without_environment))]

    all_pairings_non_sis = [val for sublist in all_pairings_non_sis for val in sublist]

    all_pairings_both = [[[both_without_environment[exp].iloc[ind1], both_without_environment[exp].iloc[ind2]] for ind1, ind2 in zip(range(len(
        both_without_environment[exp]) - 1), range(1, len(both_without_environment[exp])))] for exp in range(len(both_without_environment))]

    all_pairings_both = [val for sublist in all_pairings_both for val in sublist]


    sis_x = [all_pairings_sis[instance][0] for instance in range(len(all_pairings_sis))]
    sis_y = [all_pairings_sis[instance][1] for instance in range(len(all_pairings_sis))]
    non_sis_x = [all_pairings_non_sis[instance][0] for instance in range(len(all_pairings_non_sis))]
    non_sis_y = [all_pairings_non_sis[instance][1] for instance in range(len(all_pairings_non_sis))]
    both_x = [all_pairings_both[instance][0] for instance in range(len(all_pairings_both))]
    both_y = [all_pairings_both[instance][1] for instance in range(len(all_pairings_both))]

    mask = ~np.isnan(np.concatenate((np.array(sis_x), np.array(non_sis_x)), axis=0)) & ~np.isnan(np.concatenate((np.array(sis_y),
                                                                                                                     np.array(non_sis_y)), axis=0))
    mask_sis = ~np.isnan(np.array(sis_x)) & ~np.isnan(np.array(sis_y))
    mask_non_sis = ~np.isnan(np.array(non_sis_x)) & ~np.isnan(np.array(non_sis_y))
    mask_both = ~np.isnan(np.array(both_x)) & ~np.isnan(np.array(both_y))
    together_pcoeff = stats.pearsonr(np.concatenate((np.array(sis_x), np.array(non_sis_x)), axis=0)[mask], np.concatenate((np.array(sis_y),
                                                                                                                     np.array(non_sis_y)),
                                                                                                                          axis=0)[mask])
    best_fit_together = stats.linregress(np.concatenate((np.array(sis_x), np.array(non_sis_x)), axis=0)[mask], np.concatenate((np.array(sis_y),
                                                                                                                     np.array(non_sis_y)),
                                                                                                                        axis=0)[mask])
    best_fit_sis = stats.linregress(np.array(sis_x)[mask_sis], np.array(sis_y)[mask_sis])
    sis_pcoeff = stats.pearsonr(np.array(sis_x)[mask_sis], np.array(sis_y)[mask_sis])
    best_fit_non_sis = stats.linregress(np.array(non_sis_x)[mask_non_sis], np.array(non_sis_y)[mask_non_sis])
    non_sis_pcoeff = stats.pearsonr(np.array(non_sis_x)[mask_non_sis], np.array(non_sis_y)[mask_non_sis])
    best_fit_both = stats.linregress(np.array(both_x)[mask_both], np.array(both_y)[mask_both])
    both_pcoeff = stats.pearsonr(np.array(both_x)[mask_both], np.array(both_y)[mask_both])

    # For Sister and nonsister, which have the same environment
    plt.scatter(sis_x, sis_y, label='Sisters, {} samples'.format(len(all_pairings_sis)))
    plt.scatter(non_sis_x, non_sis_y, label='Non-Sisters, {} samples'.format(len(all_pairings_non_sis)))
    plt.plot(np.arange(-1.2, 1.4, .2), best_fit_sis[1] + best_fit_sis[0] * np.arange(-1.2, 1.4, .2), label='Sis Pearson Coeff={:.2e}'.format(
        sis_pcoeff[0]), color='green')
    plt.plot(np.arange(-1.2, 1.4, .2), best_fit_non_sis[1] + best_fit_non_sis[0] * np.arange(-1.2, 1.4, .2), label='NonSis Pearson Coeff={:.2e}'.format(
        non_sis_pcoeff[0]), color='black')
    plt.plot(np.arange(-1.2, 1.4, .2), best_fit_together[1] + best_fit_together[0] * np.arange(-1.2, 1.4, .2), label='Pearson Coeff={:.2e}'.format(
        together_pcoeff[0]), color='magenta')
    plt.legend()
    plt.xlabel(r'$\Delta\tau_i$')
    plt.ylabel(r'$\Delta\tau_{i+1}$')
    plt.ylim([-1.2,1.2])
    plt.xlim([-1.2, 1.2])
    # plt.show()
    plt.savefig('PCorr S and NS gen. time diff. of consequent gens.png', dpi=300)
    plt.close()

    # For control, which doesn't have the same environment
    plt.scatter(both_x, both_y, label='Control, {} samples'.format(len(np.array(both_x)[mask_both])))
    plt.plot(np.arange(-1.2, 1.4, .2), best_fit_both[1] + best_fit_both[0] * np.arange(-1.2, 1.4, .2), label='Control Pearson Coeff={:.2e}'.format(
        both_pcoeff[0]), color='black')
    plt.legend()
    plt.xlabel(r'$\Delta\tau_i$')
    plt.ylabel(r'$\Delta\tau_{i+1}$')
    plt.ylim([-1.2,1.2])
    plt.xlim([-1.2, 1.2])
    # plt.show()
    plt.savefig('PCorr Control gen. time diff. of consequent gens.png', dpi=300)
    plt.close()


if __name__ == "__main__":
    # main()
    same_plot_with_only_data_from_one_experiment(mingen=30)
