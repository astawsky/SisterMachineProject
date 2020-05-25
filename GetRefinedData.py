
from __future__ import print_function

import numpy as np
import argparse
import sys,math
import glob

import pickle

import sistercellclass as ssc

import CALCULATETHEBETAS

import CALCULATETHEBETAS1


def main():
    # For Mac
    infiles_sisters = glob.glob(r'/Users/alestawsky/PycharmProjects/untitled/SISTERS/*.xls')
    infiles_nonsisters = glob.glob(r'/Users/alestawsky/PycharmProjects/untitled/NONSISTERS/*.xls')

    metadatastruct = ssc.CycleData(infiles_sisters = infiles_sisters,  infiles_nonsisters = infiles_nonsisters)

    print(type(metadatastruct))

    pickle_out = open("metastructdata.pickle", "wb")
    pickle.dump(metadatastruct, pickle_out)
    pickle_out.close()
    print('imported sis/nonsis/both classes')

    # Import the Refined Data
    pickle_in = open("metastructdata.pickle", "rb")
    metadatastruct = pickle.load(pickle_in)
    pickle_in.close()

    # for Sisters and nonsisters
    # Calculate the betas
    sis_phi_star_A_array, sis_phi_star_B_array, sis_beta_A_array, sis_beta_B_array, sis_pooled_data_dict = \
        CALCULATETHEBETAS1.main(ensemble='Sisters only', graph=False, metadatastruct=metadatastruct, pooled=True)

    # CALCULATE THE ACF OF TRAJECTORY
    sis_acfA_array = np.array([])
    sis_acfB_array = np.array([])
    print(len(metadatastruct.Sisters))
    for dataID in np.arange(len(metadatastruct.Sisters), dtype=int):
        acfA, acfB = metadatastruct.Sisters.AutocorrelationFFT(dataID, normalize=True, maxlen=len(metadatastruct.
                                                                                                  Sisters),
                                                               enforcelen=False)
        sis_acfA_array = np.append(sis_acfA_array, acfA)
        sis_acfB_array = np.append(sis_acfB_array, acfB)

    # add them as attributes
    metadatastruct.sis_phi_star_A_array = sis_phi_star_A_array
    metadatastruct.sis_phi_star_B_array = sis_phi_star_B_array
    metadatastruct.sis_beta_A_array = sis_beta_A_array
    metadatastruct.sis_beta_B_array = sis_beta_B_array
    metadatastruct.sis_acfA_array = sis_acfA_array
    metadatastruct.sis_acfB_array = sis_acfB_array
    metadatastruct.sis_pooled_data_dict = sis_pooled_data_dict

    # Calculate the betas, maybe plot them
    non_sis_phi_star_A_array, non_sis_phi_star_B_array, non_sis_beta_A_array, non_sis_beta_B_array, non_sis_pooled_data_dict = \
        CALCULATETHEBETAS1.main(ensemble='NonSisters only', graph=False, metadatastruct=metadatastruct, pooled=True)

    non_sis_acfA_array = np.array([])
    non_sis_acfB_array = np.array([])

    print(len(metadatastruct.Nonsisters))
    for dataID in np.arange(len(metadatastruct.Nonsisters), dtype=int):
        acfA1, acfB1 = metadatastruct.Nonsisters.AutocorrelationFFT(dataID, normalize=True,
                                                                maxlen=len(metadatastruct.Nonsisters), enforcelen=False)
        non_sis_acfA_array = np.append(non_sis_acfA_array, acfA1)
        non_sis_acfB_array = np.append(non_sis_acfB_array, acfB1)

    # add them as attributes
    metadatastruct.non_sis_phi_star_A_array = non_sis_phi_star_A_array
    metadatastruct.non_sis_phi_star_B_array = non_sis_phi_star_B_array
    metadatastruct.non_sis_beta_A_array = non_sis_beta_A_array
    metadatastruct.non_sis_beta_B_array = non_sis_beta_B_array
    metadatastruct.non_sis_acfA_array = non_sis_acfA_array
    metadatastruct.non_sis_acfB_array = non_sis_acfB_array
    metadatastruct.non_sis_pooled_data_dict = non_sis_pooled_data_dict

    # Calculate the betas, maybe plot them
    both_phi_star_A_array, both_phi_star_B_array, both_beta_A_array, both_beta_B_array, both_pooled_data_dict = \
        CALCULATETHEBETAS1.main(ensemble='Both', graph=False, metadatastruct=metadatastruct, pooled=True)

    both_acfA_array = np.array([])
    both_acfB_array = np.array([])

    for trajA, trajB in zip(metadatastruct.A_dict_both.values(), metadatastruct.B_dict_both.values()):
        acfA2, acfB2 = metadatastruct.AutocorrelationFFT1(trajA, trajB, normalize=True, maxlen=len(metadatastruct.A_dict_both),
                                                               enforcelen=False)
        both_acfA_array = np.append(both_acfA_array, acfA2)
        both_acfB_array = np.append(both_acfB_array, acfB2)

    # add them as attributes
    metadatastruct.both_phi_star_A_array = both_phi_star_A_array
    metadatastruct.both_phi_star_B_array = both_phi_star_B_array
    metadatastruct.both_beta_A_array = both_beta_A_array
    metadatastruct.both_beta_B_array = both_beta_B_array
    metadatastruct.both_acfA_array = both_acfA_array
    metadatastruct.both_acfB_array = both_acfB_array
    metadatastruct.both_pooled_data_dict = both_pooled_data_dict

    # # # # # # # # # # # # #  # # #  # # # #  # # # # # #  # # # #  #

    # FOR THE "TRUSTWORTHY (HOPEFULLY)" BETAS
    s_phi_star_A_array, s_phi_star_B_array, s_beta_A_array, s_beta_B_array, s_pooled_data_dict = \
        CALCULATETHEBETAS.main(ensemble='Sisters only', graph=False, metadatastruct=metadatastruct, pooled=True)

    # CALCULATE THE ACF OF TRAJECTORY
    s_acfA_array = np.array([])
    s_acfB_array = np.array([])
    print(len(metadatastruct.Sisters))
    for dataID in np.arange(len(metadatastruct.Sisters), dtype=int):
        acfA, acfB = metadatastruct.Sisters.AutocorrelationFFT(dataID, normalize=True, maxlen=len(metadatastruct.
                                                                                                  Sisters),
                                                               enforcelen=False)
        s_acfA_array = np.append(s_acfA_array, acfA)
        s_acfB_array = np.append(s_acfB_array, acfB)

    # add them as attributes
    metadatastruct.s_phi_star_A_array = s_phi_star_A_array
    metadatastruct.s_phi_star_B_array = s_phi_star_B_array
    metadatastruct.s_beta_A_array = s_beta_A_array
    metadatastruct.s_beta_B_array = s_beta_B_array
    metadatastruct.s_acfA_array = s_acfA_array
    metadatastruct.s_acfB_array = s_acfB_array
    metadatastruct.s_pooled_data_dict = s_pooled_data_dict

    # Calculate the betas, maybe plot them
    n_phi_star_A_array, n_phi_star_B_array, n_beta_A_array, n_beta_B_array, n_pooled_data_dict = \
        CALCULATETHEBETAS.main(ensemble='NonSisters only', graph=False, metadatastruct=metadatastruct, pooled=True)

    n_acfA_array = np.array([])
    n_acfB_array = np.array([])

    print(len(metadatastruct.Nonsisters))
    for dataID in np.arange(len(metadatastruct.Nonsisters), dtype=int):
        acfA1, acfB1 = metadatastruct.Nonsisters.AutocorrelationFFT(dataID, normalize=True,
                                                                    maxlen=len(metadatastruct.Nonsisters),
                                                                    enforcelen=False)
        n_acfA_array = np.append(n_acfA_array, acfA1)
        n_acfB_array = np.append(n_acfB_array, acfB1)

    # add them as attributes
    metadatastruct.n_phi_star_A_array = n_phi_star_A_array
    metadatastruct.n_phi_star_B_array = n_phi_star_B_array
    metadatastruct.n_beta_A_array = n_beta_A_array
    metadatastruct.n_beta_B_array = n_beta_B_array
    metadatastruct.n_acfA_array = n_acfA_array
    metadatastruct.n_acfB_array = n_acfB_array
    metadatastruct.n_pooled_data_dict = n_pooled_data_dict

    # Calculate the betas, maybe plot them
    b_phi_star_A_array, b_phi_star_B_array, b_beta_A_array, b_beta_B_array, b_pooled_data_dict = \
        CALCULATETHEBETAS.main(ensemble='Both', graph=False, metadatastruct=metadatastruct, pooled=True)

    b_acfA_array = np.array([])
    b_acfB_array = np.array([])

    for trajA, trajB in zip(metadatastruct.A_dict_both.values(), metadatastruct.B_dict_both.values()):
        acfA2, acfB2 = metadatastruct.AutocorrelationFFT1(trajA, trajB, normalize=True,
                                                          maxlen=len(metadatastruct.A_dict_both),
                                                          enforcelen=False)
        b_acfA_array = np.append(b_acfA_array, acfA2)
        b_acfB_array = np.append(b_acfB_array, acfB2)

    # add them as attributes
    metadatastruct.b_phi_star_A_array = b_phi_star_A_array
    metadatastruct.b_phi_star_B_array = b_phi_star_B_array
    metadatastruct.b_beta_A_array = b_beta_A_array
    metadatastruct.b_beta_B_array = b_beta_B_array
    metadatastruct.b_acfA_array = b_acfA_array
    metadatastruct.b_acfB_array = b_acfB_array
    metadatastruct.b_pooled_data_dict = b_pooled_data_dict

    pickle_out = open("metastructdata.pickle", "wb")
    pickle.dump(metadatastruct, pickle_out)
    pickle_out.close()

    print('imported sis')

    print(len(metadatastruct.A_dict_both), len(metadatastruct.B_dict_both))


if __name__ == '__main__':
    main()
