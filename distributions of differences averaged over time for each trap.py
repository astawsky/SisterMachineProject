
from __future__ import print_function

import numpy as np

import matplotlib.pyplot as plt

import pickle

import scipy.stats as stats


def Distribution1(diffs_sis, xlabel, abs_range, dataset):
    # PoolID == 1
    sis_label = dataset + r'$\mu=$' + '{:.2e}'.format(np.mean(diffs_sis)) + r', $\sigma=$' + '{:.2e}'.format(
        np.std(diffs_sis))

    plt.hist(x=diffs_sis, label=sis_label, weights=np.ones_like(diffs_sis) / float(len(diffs_sis)), range=abs_range)
    # arr_sis =
    # plt.close()
    #
    # plt.plot(np.array([(arr_sis[1][l] + arr_sis[1][l + 1]) / 2. for l in range(len(arr_sis[1]) - 1)]), arr_sis[0], label=sis_label, marker='.')
    plt.xlabel(xlabel)
    plt.ylabel('PDF (Weighted Histogram)')
    plt.legend()
    # plt.show()
    plt.savefig('difference of mean of '+xlabel+' '+dataset+'distribution.png', dpi=300)
    plt.close()


def main():
    # Import the Refined Data
    pickle_in = open("metastructdata.pickle", "rb")
    struct = pickle.load(pickle_in)
    pickle_in.close()

    for cycle_param, abs_range in zip(['generationtime', 'growth_length', 'length_birth', 'length_final'], [[.1,1],[.5,2],[1.5,4],[3,7.5]]):

        sis_dist = np.array([np.mean(valA[cycle_param])-np.mean(valB[cycle_param]) for valA, valB in zip(struct.A_dict_sis.values(), 
                                                                                               struct.B_dict_sis.values())])
        non_sis_dist = np.array([np.mean(valA[cycle_param]) - np.mean(valB[cycle_param]) for valA, valB in zip(struct.A_dict_non_sis.values(),
                                                                                                           struct.B_dict_non_sis.values())])
        both_dist = np.array([np.mean(valA[cycle_param]) - np.mean(valB[cycle_param]) for valA, valB in zip(struct.A_dict_both.values(),
                                                                                                           struct.B_dict_both.values())])

        Distribution1(sis_dist, cycle_param, None, 'S: ')

        Distribution1(non_sis_dist, cycle_param, None, 'NS: ')

        Distribution1(both_dist, cycle_param, None, 'Control: ')


if __name__ == '__main__':
    main()
