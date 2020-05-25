
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

    max_gen = 8
    
    gen_times_A_sis = [np.array([struct.A_dict_sis[keyA]['generationtime'].loc[gen] for keyA, keyB in zip(struct.A_dict_sis.keys(), struct.B_dict_sis.keys()) if len(struct.A_dict_sis[keyA]['generationtime'])>gen and len(struct.B_dict_sis[keyB]['generationtime'])>gen]) for gen in range(max_gen)]

    gen_times_B_sis = [np.array([struct.B_dict_sis[keyB]['generationtime'].loc[gen] for keyA, keyB in zip(struct.A_dict_sis.keys(), struct.B_dict_sis.keys()) if len(struct.A_dict_sis[keyA]['generationtime'])>gen and len(struct.B_dict_sis[keyB]['generationtime'])>gen]) for gen in range(max_gen)]
    
    for gen in range(max_gen):
        print(str(gen), gen_times_A_sis[gen])

    pearson_array_sis = [stats.pearsonr(gen_times_A_sis[gen], gen_times_B_sis[gen])[1] for gen in range(max_gen)]

    gen_times_A_non_sis = [np.array([struct.A_dict_non_sis[keyA]['generationtime'].loc[gen] for keyA, keyB in zip(struct.A_dict_non_sis.keys(), struct.B_dict_non_sis.keys()) if len(struct.A_dict_non_sis[keyA]['generationtime'])>gen and len(struct.B_dict_non_sis[keyB]['generationtime'])>gen]) for gen in
                       range(max_gen)]

    gen_times_B_non_sis = [np.array([struct.B_dict_non_sis[keyB]['generationtime'].loc[gen] for keyA, keyB in zip(struct.A_dict_non_sis.keys(), struct.B_dict_non_sis.keys()) if len(struct.A_dict_non_sis[keyA]['generationtime'])>gen and len(struct.B_dict_non_sis[keyB]['generationtime'])>gen]) for gen in
                       range(max_gen)]

    pearson_array_non_sis = [stats.pearsonr(gen_times_A_non_sis[gen], gen_times_B_non_sis[gen])[1] for gen in range(max_gen)]
    
    gen_times_A_both = [np.array([struct.A_dict_both[keyA]['generationtime'].loc[gen] for keyA, keyB in zip(struct.A_dict_both.keys(), struct.B_dict_both.keys()) if len(struct.A_dict_both[keyA]['generationtime'])>gen and len(struct.B_dict_both[keyB]['generationtime'])>gen]) for gen in
                       range(max_gen)]

    gen_times_B_both = [np.array([struct.B_dict_both[keyB]['generationtime'].loc[gen] for keyA, keyB in zip(struct.A_dict_both.keys(), struct.B_dict_both.keys()) if len(struct.A_dict_both[keyA]['generationtime'])>gen and len(struct.B_dict_both[keyB]['generationtime'])>gen]) for gen in
                       range(max_gen)]


    pearson_array_both = [stats.pearsonr(gen_times_A_both[gen], gen_times_B_both[gen])[1] for gen in range(max_gen)]
    
    plt.plot(pearson_array_sis, label=r'sis', marker='.')
    plt.plot(pearson_array_non_sis, label=r'non_sis', marker='.')
    plt.plot(pearson_array_both, label=r'both', marker='.')
    plt.xlabel('generation')
    plt.legend()
    plt.ylabel('pearson correlation of generation time')
    plt.show()

    # print(len(gen_times_A_non_sis[6]), len(gen_times_A_non_sis[4]), len(gen_times_A_non_sis[5]))
    #
    # plt.scatter(gen_times_A_non_sis[6], gen_times_B_non_sis[6], c='g', marker='+')
    #
    # plt.scatter(gen_times_A_non_sis[4], gen_times_B_non_sis[4], c='b', marker='^')
    #
    # plt.scatter(gen_times_A_non_sis[5], gen_times_B_non_sis[5], c='r')
    # plt.show()

    # plt.plot([len(gen_times_A_sis[gen]) for gen in range(max_gen)], label='sis', marker='.')
    # plt.plot([len(gen_times_A_non_sis[gen]) for gen in range(max_gen)], label='non', marker='.')
    # plt.plot([len(gen_times_A_both[gen]) for gen in range(max_gen)], label='both', marker='.')
    # plt.show()
    
    
if __name__ == '__main__':
    main()