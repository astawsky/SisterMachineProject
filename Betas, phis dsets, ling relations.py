
from __future__ import print_function

import numpy as np

import matplotlib.pyplot as plt

import pickle

import matplotlib.patches as mpatches

import random

def main():
    # Import the Refined Data
    pickle_in = open("metastructdata.pickle", "rb")
    metadatastruct = pickle.load(pickle_in)
    pickle_in.close()

    # """
    plt.figure(dpi=300)
    for Abeta, Bbeta, Aphi, Bphi, Al_birth, Bl_birth in zip([metadatastruct.sis_beta_A_array,
                         metadatastruct.non_sis_beta_A_array, metadatastruct.both_beta_A_array],
                        [metadatastruct.sis_beta_B_array, metadatastruct.non_sis_beta_B_array,
                         metadatastruct.both_beta_B_array],

                        [metadatastruct.sis_phi_star_A_array, metadatastruct.non_sis_phi_star_A_array,
                         metadatastruct.both_phi_star_A_array], [metadatastruct.sis_phi_star_B_array,
                         metadatastruct.non_sis_phi_star_B_array, metadatastruct.both_phi_star_B_array],

                        [np.array([metadatastruct.A_dict_sis[ID]['length_birth'] for ID in metadatastruct.A_dict_sis.keys()]),
                         np.array([metadatastruct.A_dict_non_sis[ID]['length_birth'] for ID in
                                   metadatastruct.A_dict_non_sis.keys()]),
                         np.array([metadatastruct.A_dict_both[ID]['length_birth'] for ID in metadatastruct.A_dict_both.keys()])],

                        [np.array([metadatastruct.B_dict_sis[ID]['length_birth'] for ID in metadatastruct.B_dict_sis.keys()]),
                         np.array([metadatastruct.B_dict_non_sis[ID]['length_birth'] for ID in
                                   metadatastruct.B_dict_non_sis.keys()]),
                         np.array([metadatastruct.B_dict_both[ID]['length_birth'] for ID in metadatastruct.B_dict_both.keys()])],
                                                            
                                                            ):

        num_of_examples = 1
        if Abeta == metadatastruct.sis_beta_A_array:
            label='Sister'
            color = 'r'
        elif Abeta == metadatastruct.non_sis_beta_A_array:
            label='Non-Sister'
            color = 'b'
        elif Abeta == metadatastruct.both_beta_A_array:
            label='Control'
            color = 'g'


        # np.log(np.divide(Al_birth, metadatastruct.x_avg))
        for ID in range(len(Abeta)):
            # For A cell
            plt.plot(np.array([-1,-.5,0,.5,1]), Aphi[ID] + Abeta[ID] * np.array([-1,-.5,0,.5,1]), color=color)
            # For B cell
            plt.plot(np.array([-1,-.5,0,.5,1]), Bphi[ID] + Bbeta[ID] * np.array([-1,-.5,0,.5,1]), color=color)

    # plt.legend(['Sister','Non-Sister','Control'])
    plt.xlabel(r'$ln(\frac{x_n}{x^*})$')
    plt.ylabel(r'$\phi^{(k)}=\alpha_nT_n$')
    red_patch = mpatches.Patch(color='red', label='Sister')
    blue_patch = mpatches.Patch(color='blue', label='Non-Sister')
    green_patch = mpatches.Patch(color='green', label='Control')
    plt.legend(handles=[red_patch, blue_patch, green_patch])
    plt.title(r'Fitted $\beta$ and $\phi$ from all samples at all cycles')
    plt.savefig('New beta and phi plots/beta and phi for all samples in all 3 sets')
    # """

    plt.figure(dpi=300)
    for Abeta, Bbeta, Aphi, Bphi, Al_birth, Bl_birth in zip([metadatastruct.sis_beta_A_array,
                                                             metadatastruct.non_sis_beta_A_array,
                                                             metadatastruct.both_beta_A_array],
                                                            [metadatastruct.sis_beta_B_array,
                                                             metadatastruct.non_sis_beta_B_array,
                                                             metadatastruct.both_beta_B_array],

                                                            [metadatastruct.sis_phi_star_A_array,
                                                             metadatastruct.non_sis_phi_star_A_array,
                                                             metadatastruct.both_phi_star_A_array],
                                                            [metadatastruct.sis_phi_star_B_array,
                                                             metadatastruct.non_sis_phi_star_B_array,
                                                             metadatastruct.both_phi_star_B_array],

                                                            [np.array(
                                                                [metadatastruct.A_dict_sis[ID]['length_birth'] for ID in
                                                                 metadatastruct.A_dict_sis.keys()]),
                                                             np.array(
                                                                 [metadatastruct.A_dict_non_sis[ID]['length_birth'] for
                                                                  ID in
                                                                  metadatastruct.A_dict_non_sis.keys()]),
                                                             np.array(
                                                                 [metadatastruct.A_dict_both[ID]['length_birth'] for ID
                                                                  in metadatastruct.A_dict_both.keys()])],

                                                            [np.array(
                                                                [metadatastruct.B_dict_sis[ID]['length_birth'] for ID in
                                                                 metadatastruct.B_dict_sis.keys()]),
                                                             np.array(
                                                                 [metadatastruct.B_dict_non_sis[ID]['length_birth'] for
                                                                  ID in
                                                                  metadatastruct.B_dict_non_sis.keys()]),
                                                             np.array(
                                                                 [metadatastruct.B_dict_both[ID]['length_birth'] for ID
                                                                  in metadatastruct.B_dict_both.keys()])],

                                                            ):

        num_of_examples = 1
        if Abeta == metadatastruct.sis_beta_A_array:
            label = 'Sister'
            color = 'r'
        elif Abeta == metadatastruct.non_sis_beta_A_array:
            label = 'Non-Sister'
            color = 'b'
        elif Abeta == metadatastruct.both_beta_A_array:
            label = 'Control'
            color = 'g'

        # np.log(np.divide(Al_birth, metadatastruct.x_avg))
        for ID in random.sample(range(len(Abeta)), k=2):
            # For A cell
            plt.plot(np.array([-.5, 0, .5]), Aphi[ID] + Abeta[ID] * np.array([-.5, 0, .5]), color=color, label=label+str(ID))
            # For B cell
            plt.plot(np.array([-.5, 0, .5]), Bphi[ID] + Bbeta[ID] * np.array([-.5, 0, .5]), color=color, label=label+str(ID))

    # plt.legend(['Sister','Non-Sister','Control'])
    plt.xlabel(r'$ln(\frac{x_n}{x^*})$')
    plt.ylabel(r'$\phi^{(k)}=\alpha_nT_n$')
    red_patch = mpatches.Patch(color='red', label='Sister')
    blue_patch = mpatches.Patch(color='blue', label='Non-Sister')
    green_patch = mpatches.Patch(color='green', label='Control')
    plt.legend(handles=[red_patch, blue_patch, green_patch])
    plt.title(r'Fitted $\beta$ and $\phi$ from all samples at all cycles')
    plt.savefig('New beta and phi plots/beta and phi for some samples')


if __name__ == '__main__':
    main()

