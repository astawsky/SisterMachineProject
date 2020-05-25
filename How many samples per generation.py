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
    
    total_num_of_gens = 61 # in reality it is = 161

    sis_samples = [len([1 for IDA, IDB in zip(struct.A_dict_sis.keys(), struct.B_dict_sis.keys()) if
                        min(len(struct.A_dict_sis[IDA]['generationtime']), len(struct.B_dict_sis[IDB]['generationtime'])) > gen_num]) for gen_num
                   in range(1, total_num_of_gens)]

    non_sis_samples = [len([1 for IDA, IDB in zip(struct.A_dict_non_sis.keys(), struct.B_dict_non_sis.keys()) if
                        min(len(struct.A_dict_non_sis[IDA]['generationtime']), len(struct.B_dict_non_sis[IDB]['generationtime'])) > gen_num]) for gen_num
                   in range(1, total_num_of_gens)]

    both_samples = [len([1 for IDA, IDB in zip(struct.A_dict_both.keys(), struct.B_dict_both.keys()) if
                        min(len(struct.A_dict_both[IDA]['generationtime']), len(struct.B_dict_both[IDB]['generationtime'])) > gen_num]) for gen_num
                   in range(1, total_num_of_gens)]

    plt.plot(range(1,total_num_of_gens), sis_samples, marker='.', label='Sisters')
    plt.plot(range(1,total_num_of_gens), non_sis_samples, marker='^', label='Non-Sisters')
    plt.plot(range(1,total_num_of_gens), both_samples, marker='*', label='Control')
    plt.xlabel('Generation')
    plt.xticks([1, 10, 20 ,30 ,40 ,50, 60])
    plt.yticks(range(0, 132, 10))
    plt.ylabel('Number of samples available')
    plt.grid(True)
    plt.title('How many samples are available per generation?')
    plt.legend()
    plt.show()
    # plt.savefig('How many samples are available per generation?', dpi=300)


if __name__ == '__main__':
    main()
