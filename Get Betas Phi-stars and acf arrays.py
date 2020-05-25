
from __future__ import print_function

import numpy as np

import glob

import pickle

import SisterMachineDataProcessing as ssc

import CALCULATETHEBETAS

import CALCULATETHEBETAS1


def main():
    # For Mac
    infiles_sisters = glob.glob(r'/Users/alestawsky/PycharmProjects/untitled/SISTERS/*.xls')
    infiles_nonsisters = glob.glob(r'/Users/alestawsky/PycharmProjects/untitled/NONSISTERS/*.xls')

    saved_data = ssc.CycleData(infiles_sisters = infiles_sisters,  infiles_nonsisters = infiles_nonsisters, discretize_by='length')

    pickle_out = open("metastructdata.pickle", "wb")
    pickle.dump(saved_data, pickle_out)
    pickle_out.close()

    # Import the Refined Data
    pickle_in = open("metastructdata.pickle", "rb")
    saved_data = pickle.load(pickle_in)
    pickle_in.close()

    # for Sisters and nonsisters
    # Calculate the betas
    sis_phi_star_A_array, sis_phi_star_B_array, sis_beta_A_array, sis_beta_B_array, sis_pooled_data_dict = \
        CALCULATETHEBETAS1.main(ensemble='Sisters only', graph=False, metadatastruct=saved_data, pooled=True)

    # CALCULATE THE ACF OF TRAJECTORY
    sis_acfA_array = np.array([])
    sis_acfB_array = np.array([])

    for dataID in np.arange(len(saved_data.Sisters), dtype=int):
        acfA, acfB = saved_data.Sisters.AutocorrelationFFT(dataID, normalize=True, maxlen=len(saved_data.
                                                                                                  Sisters),
                                                               enforcelen=False)
        sis_acfA_array = np.append(sis_acfA_array, acfA)
        sis_acfB_array = np.append(sis_acfB_array, acfB)

    # add them as attributes
    saved_data.sis_phi_star_A_array = sis_phi_star_A_array
    saved_data.sis_phi_star_B_array = sis_phi_star_B_array
    saved_data.sis_beta_A_array = sis_beta_A_array
    saved_data.sis_beta_B_array = sis_beta_B_array
    saved_data.sis_acfA_array = sis_acfA_array
    saved_data.sis_acfB_array = sis_acfB_array
    saved_data.sis_pooled_data_dict = sis_pooled_data_dict

    # Calculate the betas, maybe plot them
    non_sis_phi_star_A_array, non_sis_phi_star_B_array, non_sis_beta_A_array, non_sis_beta_B_array, non_sis_pooled_data_dict = \
        CALCULATETHEBETAS1.main(ensemble='NonSisters only', graph=False, metadatastruct=saved_data, pooled=True)

    non_sis_acfA_array = np.array([])
    non_sis_acfB_array = np.array([])

    for dataID in np.arange(len(saved_data.Nonsisters), dtype=int):
        acfA1, acfB1 = saved_data.Nonsisters.AutocorrelationFFT(dataID, normalize=True,
                                                                maxlen=len(saved_data.Nonsisters), enforcelen=False)
        non_sis_acfA_array = np.append(non_sis_acfA_array, acfA1)
        non_sis_acfB_array = np.append(non_sis_acfB_array, acfB1)

    # add them as attributes
    saved_data.non_sis_phi_star_A_array = non_sis_phi_star_A_array
    saved_data.non_sis_phi_star_B_array = non_sis_phi_star_B_array
    saved_data.non_sis_beta_A_array = non_sis_beta_A_array
    saved_data.non_sis_beta_B_array = non_sis_beta_B_array
    saved_data.non_sis_acfA_array = non_sis_acfA_array
    saved_data.non_sis_acfB_array = non_sis_acfB_array
    saved_data.non_sis_pooled_data_dict = non_sis_pooled_data_dict

    # Calculate the betas, maybe plot them
    both_phi_star_A_array, both_phi_star_B_array, both_beta_A_array, both_beta_B_array, both_pooled_data_dict = \
        CALCULATETHEBETAS1.main(ensemble='Both', graph=False, metadatastruct=saved_data, pooled=True)

    both_acfA_array = np.array([])
    both_acfB_array = np.array([])

    for trajA, trajB in zip(saved_data.A_dict_both.values(), saved_data.B_dict_both.values()):
        acfA2, acfB2 = saved_data.AutocorrelationFFT1(trajA, trajB, normalize=True, maxlen=len(saved_data.A_dict_both),
                                                               enforcelen=False)
        both_acfA_array = np.append(both_acfA_array, acfA2)
        both_acfB_array = np.append(both_acfB_array, acfB2)

    # add them as attributes
    saved_data.both_phi_star_A_array = both_phi_star_A_array
    saved_data.both_phi_star_B_array = both_phi_star_B_array
    saved_data.both_beta_A_array = both_beta_A_array
    saved_data.both_beta_B_array = both_beta_B_array
    saved_data.both_acfA_array = both_acfA_array
    saved_data.both_acfB_array = both_acfB_array
    saved_data.both_pooled_data_dict = both_pooled_data_dict

    # # # # # # # # # # # # #  # # #  # # # #  # # # # # #  # # # #  #

    # FOR THE "TRUSTWORTHY (HOPEFULLY)" BETAS
    s_phi_star_A_array, s_phi_star_B_array, s_beta_A_array, s_beta_B_array, s_pooled_data_dict = \
        CALCULATETHEBETAS.main(ensemble='Sisters only', graph=False, metadatastruct=saved_data, pooled=True)

    # CALCULATE THE ACF OF TRAJECTORY
    s_acfA_array = np.array([])
    s_acfB_array = np.array([])

    for dataID in np.arange(len(saved_data.Sisters), dtype=int):
        acfA, acfB = saved_data.Sisters.AutocorrelationFFT(dataID, normalize=True, maxlen=len(saved_data.
                                                                                                  Sisters),
                                                               enforcelen=False)
        s_acfA_array = np.append(s_acfA_array, acfA)
        s_acfB_array = np.append(s_acfB_array, acfB)

    # add them as attributes
    saved_data.s_phi_star_A_array = s_phi_star_A_array
    saved_data.s_phi_star_B_array = s_phi_star_B_array
    saved_data.s_beta_A_array = s_beta_A_array
    saved_data.s_beta_B_array = s_beta_B_array
    saved_data.s_acfA_array = s_acfA_array
    saved_data.s_acfB_array = s_acfB_array
    saved_data.s_pooled_data_dict = s_pooled_data_dict

    # Calculate the betas, maybe plot them
    n_phi_star_A_array, n_phi_star_B_array, n_beta_A_array, n_beta_B_array, n_pooled_data_dict = \
        CALCULATETHEBETAS.main(ensemble='NonSisters only', graph=False, metadatastruct=saved_data, pooled=True)

    n_acfA_array = np.array([])
    n_acfB_array = np.array([])

    for dataID in np.arange(len(saved_data.Nonsisters), dtype=int):
        acfA1, acfB1 = saved_data.Nonsisters.AutocorrelationFFT(dataID, normalize=True,
                                                                    maxlen=len(saved_data.Nonsisters),
                                                                    enforcelen=False)
        n_acfA_array = np.append(n_acfA_array, acfA1)
        n_acfB_array = np.append(n_acfB_array, acfB1)

    # add them as attributes
    saved_data.n_phi_star_A_array = n_phi_star_A_array
    saved_data.n_phi_star_B_array = n_phi_star_B_array
    saved_data.n_beta_A_array = n_beta_A_array
    saved_data.n_beta_B_array = n_beta_B_array
    saved_data.n_acfA_array = n_acfA_array
    saved_data.n_acfB_array = n_acfB_array
    saved_data.n_pooled_data_dict = n_pooled_data_dict

    # Calculate the betas, maybe plot them
    b_phi_star_A_array, b_phi_star_B_array, b_beta_A_array, b_beta_B_array, b_pooled_data_dict = \
        CALCULATETHEBETAS.main(ensemble='Both', graph=False, metadatastruct=saved_data, pooled=True)

    b_acfA_array = np.array([])
    b_acfB_array = np.array([])

    for trajA, trajB in zip(saved_data.A_dict_both.values(), saved_data.B_dict_both.values()):
        acfA2, acfB2 = saved_data.AutocorrelationFFT1(trajA, trajB, normalize=True,
                                                          maxlen=len(saved_data.A_dict_both),
                                                          enforcelen=False)
        b_acfA_array = np.append(b_acfA_array, acfA2)
        b_acfB_array = np.append(b_acfB_array, acfB2)

    # add them as attributes
    saved_data.b_phi_star_A_array = b_phi_star_A_array
    saved_data.b_phi_star_B_array = b_phi_star_B_array
    saved_data.b_beta_A_array = b_beta_A_array
    saved_data.b_beta_B_array = b_beta_B_array
    saved_data.b_acfA_array = b_acfA_array
    saved_data.b_acfB_array = b_acfB_array
    saved_data.b_pooled_data_dict = b_pooled_data_dict

    pickle_out = open("metastructdata.pickle", "wb")
    pickle.dump(saved_data, pickle_out)
    pickle_out.close()


if __name__ == '__main__':
    main()
