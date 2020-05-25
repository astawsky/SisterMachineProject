
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

    sis_array_A = []
    non_sis_array_A = []
    both_array_A = []
    sis_array_B = []
    non_sis_array_B = []
    both_array_B = []
    for index, index1, index2, index3 in zip(range(len(struct.Sisters)), range(len(struct.Nonsisters) - 1), np.flip(struct.Control[0], 0),
                                             np.flip(struct.Control[1], 0)):
        x = struct.Sisters[index]['meanfluorescenceA']
        y = struct.Sisters[index]['meanfluorescenceB']
        min_gen = min(len(x), len(y))

        x = x[:min_gen]
        y = y[:min_gen]

        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        sis_array_A.append(x_mean)
        sis_array_B.append(y_mean)

        x = struct.Nonsisters[index1]['meanfluorescenceA']
        y = struct.Nonsisters[index1]['meanfluorescenceB']
        min_gen = min(len(x), len(y))

        x = x[:min_gen]
        y = y[:min_gen]

        x_mean = np.mean(x)
        y_mean = np.mean(y)

        non_sis_array_A.append(x_mean)
        non_sis_array_B.append(y_mean)

        x = struct.Sisters[index2]['meanfluorescenceA']
        y = struct.Nonsisters[index3]['meanfluorescenceB']
        min_gen = min(len(x), len(y))

        x = x[:min_gen]
        y = y[:min_gen]

        x_mean = np.mean(x)
        y_mean = np.mean(y)

        both_array_A.append(x_mean)
        both_array_B.append(y_mean)

    plt.scatter(sis_array_A, sis_array_B, label='Sisters, pearson coeff={:.2e}'.format(stats.pearsonr(sis_array_A,
                                                                                                                sis_array_B)[0]))
    plt.scatter(non_sis_array_A, non_sis_array_B, label='Non-Sisters, pearson coeff={:.2e}'.format(stats.pearsonr(non_sis_array_A,
                                                                                                                            non_sis_array_B)[0]))
    plt.scatter(both_array_A, both_array_B, label='Control, pearson coeff={:.2e}'.format(stats.pearsonr(both_array_A,
                                                                                                                  both_array_B)[0]))
    title = 'Mean Fluor averaged over generations in the trap'
    plt.title(title)
    plt.xlabel('A trace')
    plt.ylabel('B trace')
    # plt.xlim([.65, .85])
    # plt.ylim([.65, .85])
    plt.legend()
    plt.show()
    # plt.savefig(title + '.png', dpi=300)
    # plt.close()


if __name__ == '__main__':
    main()
