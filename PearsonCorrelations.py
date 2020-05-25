
from __future__ import print_function

import numpy as np

import pickle

def main():
    # Import the Refined Data
    pickle_in = open("metastructdata.pickle", "rb")
    metadatastruct = pickle.load(pickle_in)
    pickle_in.close()

    # Be careful because this function writes on the directory and takes a lot of time
    # Just run it once
    metadatastruct.PearsonOverIndividual(separatebysister = False, significance_thresh = None, comparelineages = False,
                              dsets = ['s'], graph = False, seperatebydsets_only=True, cloud = False,
                                         time_graph = False)


    # Not sure how this is needed anymore or why I used it

    # A_pop = np.array([])
    # B_pop = np.array([])
    # A_pval = np.array([])
    # B_pval = np.array([])
    # for dsetA, dsetB, dtype in zip(np.array([metadatastruct.A_dict_sis, metadatastruct.A_dict_non_sis, metadatastruct.A_dict_both]),
    #                         np.array([metadatastruct.B_dict_sis, metadatastruct.B_dict_non_sis, metadatastruct.B_dict_both]),
    #                                ['Sister', 'Non-Sister', 'Control']):
    #     pop_param_A, pop_param_B, pval_param_A, pval_param_B = metadatastruct.PearsonAndPopulation(dsetA=dsetA,
    #                                             dsetB=dsetB, displacement=None)
    #     A_pop = np.append(A_pop, pop_param_A)
    #     B_pop = np.append(B_pop, pop_param_B)
    #     A_pval = np.append(A_pval, pval_param_A)
    #     B_pval = np.append(B_pval, pval_param_B)
    #
    # print(A_pop, B_pop, A_pval, B_pval)



if __name__ == '__main__':
    main()









