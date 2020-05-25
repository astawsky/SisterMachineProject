
from __future__ import print_function

import numpy as np
import argparse
import sys,math
import glob
import matplotlib.pyplot as plt

import pickle
import matplotlib.patches as mpatches

import sistercellclass as ssc

import CALCULATETHEBETAS
import os

import scipy.stats as stats
import random

def main():
    # Import the Refined Data
    pickle_in = open("metastructdata.pickle", "rb")
    metadatastruct = pickle.load(pickle_in)
    pickle_in.close()

    # beta_array = np.array([])
    # phi_array = np.array([])
    cycle_array = np.array([])
    pop_array = np.array([])

#############################################

    # # OPTIONAL, make the directories
    # os.mkdir('displacement from first cell in Traj. A')
    # os.mkdir('displacement from first cell in Traj. B')

    # graph is true if we want to see/save the actual scatter plots
    graph = True

    # IF WE WANT TO USE POINTS FROM ONLY THE CYCLE CYCLE
    for disp in range(10):  # aunt and uncle by disp generation(s)

        pop_array = np.array([])
        cycle_array = np.array([])
        for cycle in range(100):


            dispA = disp
            dispB = 0

            s_cycle_sample = [
                np.array(
                    [metadatastruct.A_dict_sis[ID_A]['length_birth'][cycle + dispA]/metadatastruct.x_avg,
                     metadatastruct.B_dict_sis[ID_B]
                     ['length_birth'][cycle + dispB]/metadatastruct.x_avg]) for ID_A, ID_B in
                zip(metadatastruct.A_dict_sis.keys(), metadatastruct.B_dict_sis.keys()) if
                min(len(metadatastruct.A_dict_sis[ID_A]), len(metadatastruct.B_dict_sis[ID_B])) > cycle + max(dispA, dispB)]

            n_cycle_sample = [
                np.array([metadatastruct.A_dict_non_sis[ID_A]['length_birth'][cycle + dispA]/metadatastruct.x_avg,
                          metadatastruct.B_dict_non_sis[ID_B]
                          ['length_birth'][cycle + dispB]/metadatastruct.x_avg]) for ID_A, ID_B in
                zip(metadatastruct.A_dict_non_sis.keys(), metadatastruct.B_dict_non_sis.keys()) if
                min(len(metadatastruct.A_dict_non_sis[ID_A]), len(metadatastruct.B_dict_non_sis[ID_B])) > cycle + max(dispA, dispB)]

            b_cycle_sample = [
                np.array([metadatastruct.A_dict_both[ID_A]['length_birth'][cycle + dispA]/metadatastruct.x_avg,
                          metadatastruct.B_dict_both[ID_B]
                          ['length_birth'][cycle + dispB]/metadatastruct.x_avg]) for ID_A, ID_B in
                zip(metadatastruct.A_dict_both.keys(), metadatastruct.B_dict_both.keys()) if
                min(len(metadatastruct.A_dict_both[ID_A]), len(metadatastruct.B_dict_both[ID_B])) > cycle + max(dispA, dispB)]

            beta_sis, phi_sis, r_value_sis, p_value_sis, std_err_sis = stats.linregress(
                np.array([s_cycle_sample[ID][0] for ID in range(len(s_cycle_sample))]),
                np.array([s_cycle_sample[ID][1] for ID in range(len(s_cycle_sample))]))
            beta_non, phi_non, r_value_non, p_value_non, std_err_non = stats.linregress(
                np.array([n_cycle_sample[ID][0] for ID in range(len(n_cycle_sample))]),
                np.array([n_cycle_sample[ID][1] for ID in range(len(n_cycle_sample))]))
            beta_both, phi_both, r_value_both, p_value_both, std_err_both = stats.linregress(
                np.array([b_cycle_sample[ID][0] for ID in range(len(b_cycle_sample))]),
                np.array([b_cycle_sample[ID][1] for ID in range(len(b_cycle_sample))]))

            if cycle == 0:
                beta_array = np.array([[beta_sis], [beta_non], [beta_both]])
                phi_array = np.array([[phi_sis], [phi_non], [phi_both]])
                cycle_array = np.append(cycle_array, cycle)
                pop_array = np.append(pop_array,
                                      np.mean([len(s_cycle_sample), len(n_cycle_sample), len(b_cycle_sample)]))

            else:
                pop_array = np.append(pop_array,
                                      np.mean([len(s_cycle_sample), len(n_cycle_sample), len(b_cycle_sample)]))

                beta_array = np.append(beta_array, np.array([[beta_sis], [beta_non], [beta_both]]), axis=1)
                phi_array = np.append(phi_array, np.array([[phi_sis], [phi_non], [phi_both]]), axis=1)
                cycle_array = np.append(cycle_array, cycle)


        fig, ax1 = plt.subplots()
        ax1.plot(cycle_array, beta_array[0], label='Sister', marker='.')
        ax1.plot(cycle_array, beta_array[1], label='Non-Sister', marker='.')
        ax1.plot(cycle_array, beta_array[2], label='Control', marker='.')
        ax1.set_xlabel('cycle number')
        ax1.set_ylabel('beta value', color='b')
        ax1.tick_params('y', colors='b')
        ax1.legend()
        # IF WE WANT TO USE POINTS FROM ONLY THE CYCLE CYCLE
        plt.ylim([-.25,1])
        plt.xlim([0,20])

        ax2 = ax1.twinx()
        ax2.plot(cycle_array, pop_array, label='mean amount of samples', color='r', marker='.')
        ax2.set_ylabel('average number of samples of all data sets', color='r')
        ax2.tick_params('y', colors='r')
        ax2.legend()
        # IF WE WANT TO USE POINTS FROM ONLY THE CYCLE CYCLE
        plt.xlim([0,20])
        plt.savefig('displacement from first cell in Traj. A new/beta '+str(disp), dpi=300)
        plt.close()
        # plt.show()

        fig, ax1 = plt.subplots()
        ax1.plot(cycle_array, phi_array[0], label='Sister', marker='.')
        ax1.plot(cycle_array, phi_array[1], label='Non-Sister', marker='.')
        ax1.plot(cycle_array, phi_array[2], label='Control', marker='.')
        ax1.set_xlabel('cycle number')
        ax1.set_ylabel('phi value', color='b')
        ax1.tick_params('y', colors='b')
        ax1.legend()
        # IF WE WANT TO USE POINTS FROM ONLY THE CYCLE CYCLE
        plt.ylim([0, 4])
        plt.xlim([0,20])

        ax2 = ax1.twinx()
        ax2.plot(cycle_array, pop_array, label='mean amount of samples', color='r', marker='.')
        ax2.set_ylabel('average number of samples of all data sets', color='r')
        ax2.tick_params('y', colors='r')
        ax2.legend()
        # IF WE WANT TO USE POINTS FROM ONLY THE CYCLE CYCLE
        plt.xlim([0,20])
        plt.savefig('displacement from first cell in Traj. A new/phi '+str(disp), dpi=300)
        plt.close()
        # plt.show()

        # part 2
        cycle_array = np.array([])
        pop_array = np.array([])
        for cycle in range(100):


            dispA = 0
            dispB = disp

            s_cycle_sample = [
                np.array(
                    [metadatastruct.A_dict_sis[ID_A]['length_birth'][cycle + dispA]/metadatastruct.x_avg,
                     metadatastruct.B_dict_sis[ID_B]
                     ['length_birth'][cycle + dispB]/metadatastruct.x_avg]) for ID_A, ID_B in
                zip(metadatastruct.A_dict_sis.keys(), metadatastruct.B_dict_sis.keys()) if
                min(len(metadatastruct.A_dict_sis[ID_A]), len(metadatastruct.B_dict_sis[ID_B])) > cycle + max(dispA, dispB)]

            n_cycle_sample = [
                np.array([metadatastruct.A_dict_non_sis[ID_A]['length_birth'][cycle + dispA]/metadatastruct.x_avg,
                          metadatastruct.B_dict_non_sis[ID_B]
                          ['length_birth'][cycle + dispB]/metadatastruct.x_avg]) for ID_A, ID_B in
                zip(metadatastruct.A_dict_non_sis.keys(), metadatastruct.B_dict_non_sis.keys()) if
                min(len(metadatastruct.A_dict_non_sis[ID_A]), len(metadatastruct.B_dict_non_sis[ID_B])) > cycle + max(dispA, dispB)]

            b_cycle_sample = [
                np.array([metadatastruct.A_dict_both[ID_A]['length_birth'][cycle + dispA]/metadatastruct.x_avg,
                          metadatastruct.B_dict_both[ID_B]
                          ['length_birth'][cycle + dispB]/metadatastruct.x_avg]) for ID_A, ID_B in
                zip(metadatastruct.A_dict_both.keys(), metadatastruct.B_dict_both.keys()) if
                min(len(metadatastruct.A_dict_both[ID_A]), len(metadatastruct.B_dict_both[ID_B])) > cycle + max(dispA, dispB)]
            # print(len(s_cycle_sample))
            beta_sis, phi_sis, r_value_sis, p_value_sis, std_err_sis = stats.linregress(
                np.array([s_cycle_sample[ID][0] for ID in range(len(s_cycle_sample))]),
                np.array([s_cycle_sample[ID][1] for ID in range(len(s_cycle_sample))]))
            beta_non, phi_non, r_value_non, p_value_non, std_err_non = stats.linregress(
                np.array([n_cycle_sample[ID][0] for ID in range(len(n_cycle_sample))]),
                np.array([n_cycle_sample[ID][1] for ID in range(len(n_cycle_sample))]))
            beta_both, phi_both, r_value_both, p_value_both, std_err_both = stats.linregress(
                np.array([b_cycle_sample[ID][0] for ID in range(len(b_cycle_sample))]),
                np.array([b_cycle_sample[ID][1] for ID in range(len(b_cycle_sample))]))

            if cycle == 0:
                beta_array = np.array([[beta_sis], [beta_non], [beta_both]])
                phi_array = np.array([[phi_sis], [phi_non], [phi_both]])
                cycle_array = np.append(cycle_array, cycle)
                pop_array = np.append(pop_array,
                                      np.mean([len(s_cycle_sample), len(n_cycle_sample), len(b_cycle_sample)]))

            else:
                pop_array = np.append(pop_array,
                                      np.mean([len(s_cycle_sample), len(n_cycle_sample), len(b_cycle_sample)]))

                beta_array = np.append(beta_array, np.array([[beta_sis], [beta_non], [beta_both]]), axis=1)
                phi_array = np.append(phi_array, np.array([[phi_sis], [phi_non], [phi_both]]), axis=1)
                cycle_array = np.append(cycle_array, cycle)

            if graph and cycle < 10:
                plt.scatter(np.array([s_cycle_sample[ID][0] for ID in range(len(s_cycle_sample))]),
                            np.array([s_cycle_sample[ID][1] for ID in range(len(s_cycle_sample))]), label="Sister")
                plt.scatter(np.array([n_cycle_sample[ID][0] for ID in range(len(n_cycle_sample))]),
                            np.array([n_cycle_sample[ID][1] for ID in range(len(n_cycle_sample))]), label="Non-Sister")
                plt.scatter(np.array([b_cycle_sample[ID][0] for ID in range(len(b_cycle_sample))]),
                            np.array([b_cycle_sample[ID][1] for ID in range(len(b_cycle_sample))]), label="Control")
                plt.xlabel(r'$ln\frac{x^A_n}{x^*}$')
                plt.ylabel(r'$ln\frac{x^B_n}{x^*}$')
                plt.legend()
                plt.title('Are the '+str(cycle)+' A and '+str(dispB+cycle)+'-th generation pair linearly correlated?')
                plt.savefig('scatter plot of '+str(cycle)+' cycle with B displacement of '+str(dispB))

        fig, ax1 = plt.subplots()
        ax1.plot(cycle_array, beta_array[0], label='Sister', marker='.')
        ax1.plot(cycle_array, beta_array[1], label='Non-Sister', marker='.')
        ax1.plot(cycle_array, beta_array[2], label='Control', marker='.')
        ax1.set_xlabel('cycle number')
        ax1.set_ylabel('beta value', color='b')
        ax1.tick_params('y', colors='b')
        ax1.legend()
        # IF WE WANT TO USE POINTS FROM ONLY THE CYCLE CYCLE
        plt.ylim([-.25,1])
        plt.xlim([0,20])

        ax2 = ax1.twinx()
        ax2.plot(cycle_array, pop_array, label='mean amount of samples', color='r', marker='.')
        ax2.set_ylabel('average number of samples of all data sets', color='r')
        ax2.tick_params('y', colors='r')
        ax2.legend()
        # IF WE WANT TO USE POINTS FROM ONLY THE CYCLE CYCLE
        plt.xlim([0,20])
        plt.savefig('displacement from first cell in Traj. B new/beta '+str(disp), dpi=300)
        plt.close()
        # plt.show()

        fig, ax1 = plt.subplots()
        ax1.plot(cycle_array, phi_array[0], label='Sister', marker='.')
        ax1.plot(cycle_array, phi_array[1], label='Non-Sister', marker='.')
        ax1.plot(cycle_array, phi_array[2], label='Control', marker='.')
        ax1.set_xlabel('cycle number')
        ax1.set_ylabel('phi value', color='b')
        ax1.tick_params('y', colors='b')
        ax1.legend()
        # IF WE WANT TO USE POINTS FROM ONLY THE CYCLE CYCLE
        plt.ylim([0, 4])
        plt.xlim([0,20])

        ax2 = ax1.twinx()
        ax2.plot(cycle_array, pop_array, label='mean amount of samples', color='r', marker='.')
        ax2.set_ylabel('average number of samples of all data sets', color='r')
        ax2.tick_params('y', colors='r')
        ax2.legend()
        # IF WE WANT TO USE POINTS FROM ONLY THE CYCLE CYCLE
        plt.xlim([0,20])
        plt.savefig('displacement from first cell in Traj. B new/phi '+str(disp), dpi=300)
        plt.close()
        # plt.show()



    # # IF WE WANT TO USE ALL THE POINTS OF THE CELL THAT HAS AT LEAST THESE NUMBER OF CYCLES
    # # CHOOSE WHAT CYCLE/GEN NUMBER YOU WANT
    # graph = True
    # for cycle in range(100):
    #
    #     # # IF WE WANT TO USE POINTS FROM ALL THE CYCLES
    #     # s_cycle_sample  = [np.array([metadatastruct.A_dict_sis[ID_A]['length_birth']/metadatastruct.x_avg, metadatastruct.B_dict_sis[ID_B]
    #     # ['length_birth']/metadatastruct.x_avg]) for ID_A, ID_B in zip(metadatastruct.A_dict_sis.keys(), metadatastruct.B_dict_sis.keys()) if
    #     # min(len(metadatastruct.A_dict_sis[ID_A]), len(metadatastruct.B_dict_sis[ID_B])) > cycle]
    #     #
    #     # n_cycle_sample = [np.array([metadatastruct.A_dict_non_sis[ID_A]['length_birth']/metadatastruct.x_avg, metadatastruct.B_dict_non_sis[ID_B]
    #     # ['length_birth']/metadatastruct.x_avg]) for ID_A, ID_B in zip(metadatastruct.A_dict_non_sis.keys(), metadatastruct.B_dict_non_sis.keys()) if
    #     #                   min(len(metadatastruct.A_dict_non_sis[ID_A]), len(metadatastruct.B_dict_non_sis[ID_B])) > cycle]
    #     #
    #     # b_cycle_sample = [np.array([metadatastruct.A_dict_both[ID_A]['length_birth']/metadatastruct.x_avg, metadatastruct.B_dict_both[ID_B]
    #     # ['length_birth']/metadatastruct.x_avg]) for ID_A, ID_B in zip(metadatastruct.A_dict_both.keys(), metadatastruct.B_dict_both.keys()) if
    #     #                   min(len(metadatastruct.A_dict_both[ID_A]), len(metadatastruct.B_dict_both[ID_B])) > cycle]
    #
    #     # IF WE WANT TO USE POINTS FROM ONLY THE CYCLE CYCLE
    #     s_cycle_sample = [
    #         np.array([metadatastruct.A_dict_sis[ID_A]['length_birth'][cycle]/metadatastruct.x_avg, metadatastruct.B_dict_sis[ID_B]
    #         ['length_birth'][cycle]/metadatastruct.x_avg]) for ID_A, ID_B in
    #         zip(metadatastruct.A_dict_sis.keys(), metadatastruct.B_dict_sis.keys()) if
    #         min(len(metadatastruct.A_dict_sis[ID_A]), len(metadatastruct.B_dict_sis[ID_B])) > cycle]
    #
    #     n_cycle_sample = [
    #         np.array([metadatastruct.A_dict_non_sis[ID_A]['length_birth'][cycle]/metadatastruct.x_avg, metadatastruct.B_dict_non_sis[ID_B]
    #         ['length_birth'][cycle]/metadatastruct.x_avg]) for ID_A, ID_B in
    #         zip(metadatastruct.A_dict_non_sis.keys(), metadatastruct.B_dict_non_sis.keys()) if
    #         min(len(metadatastruct.A_dict_non_sis[ID_A]), len(metadatastruct.B_dict_non_sis[ID_B])) > cycle]
    #
    #     b_cycle_sample = [
    #         np.array([metadatastruct.A_dict_both[ID_A]['length_birth'][cycle]/metadatastruct.x_avg, metadatastruct.B_dict_both[ID_B]
    #         ['length_birth'][cycle]/metadatastruct.x_avg]) for ID_A, ID_B in
    #         zip(metadatastruct.A_dict_both.keys(), metadatastruct.B_dict_both.keys()) if
    #         min(len(metadatastruct.A_dict_both[ID_A]), len(metadatastruct.B_dict_both[ID_B])) > cycle]
    #
    #
    #     # print(len(s_cycle_sample))
    #     beta_sis, phi_sis, r_value_sis, p_value_sis, std_err_sis = stats.linregress(np.array([s_cycle_sample[ID][0] for ID in range(len(s_cycle_sample))]), np.array([s_cycle_sample[ID][1] for ID in range(len(s_cycle_sample))]))
    #     beta_non, phi_non, r_value_non, p_value_non, std_err_non = stats.linregress(np.array([n_cycle_sample[ID][0] for ID in range(len(n_cycle_sample))]), np.array([n_cycle_sample[ID][1] for ID in range(len(n_cycle_sample))]))
    #     beta_both, phi_both, r_value_both, p_value_both, std_err_both = stats.linregress(np.array([b_cycle_sample[ID][0] for ID in range(len(b_cycle_sample))]), np.array([b_cycle_sample[ID][1] for ID in range(len(b_cycle_sample))]))
    #
    #     if cycle == 0:
    #         beta_array = np.array([[beta_sis], [beta_non], [beta_both]])
    #         phi_array = np.array([[phi_sis], [phi_non], [phi_both]])
    #         cycle_array = np.append(cycle_array, cycle)
    #         pop_array = np.append(pop_array, np.mean([len(s_cycle_sample), len(n_cycle_sample), len(b_cycle_sample)]))
    #
    #     else:
    #         pop_array = np.append(pop_array, np.mean([len(s_cycle_sample), len(n_cycle_sample), len(b_cycle_sample)]))
    #
    #         beta_array = np.append(beta_array, np.array([[beta_sis], [beta_non], [beta_both]]), axis=1)
    #         phi_array = np.append(phi_array, np.array([[phi_sis], [phi_non], [phi_both]]), axis=1)
    #         cycle_array = np.append(cycle_array, cycle)
    #
    #     if graph:
    #         plt.scatter(np.array([s_cycle_sample[ID][0] for ID in range(len(s_cycle_sample))]), np.array([s_cycle_sample[ID][1] for ID in range(len(s_cycle_sample))]), label = "Sister")
    #         plt.scatter(np.array([n_cycle_sample[ID][0] for ID in range(len(n_cycle_sample))]),
    #                     np.array([n_cycle_sample[ID][1] for ID in range(len(n_cycle_sample))]), label="Non-Sister")
    #         plt.scatter(np.array([b_cycle_sample[ID][0] for ID in range(len(b_cycle_sample))]),
    #                     np.array([b_cycle_sample[ID][1] for ID in range(len(b_cycle_sample))]), label="Control")
    #         plt.xlabel(r'$ln\frac{x^A_n}{x^*}$')
    #         plt.ylabel(r'$ln\frac{x^B_n}{x^*}$')
    #         plt.legend()
    #         plt.title('Are the '+str(cycle)+'-th generation pair linearly correlated?')
    #         plt.savefig(str(cycle)+' scatter plots')
    #
    #
    #
    # # print(beta_array)
    # fig, ax1 = plt.subplots()
    # ax1.plot(cycle_array, beta_array[0], label='Sister', marker='.')
    # ax1.plot(cycle_array, beta_array[1], label='Non-Sister', marker='.')
    # ax1.plot(cycle_array, beta_array[2], label='Control', marker='.')
    # ax1.set_xlabel('cycle number')
    # ax1.set_ylabel('beta value', color='b')
    # ax1.tick_params('y', colors='b')
    # ax1.legend()
    # # IF WE WANT TO USE POINTS FROM ONLY THE CYCLE CYCLE
    # # plt.ylim([-.25,1])
    # # plt.xlim([0,20])
    #
    # ax2 = ax1.twinx()
    # ax2.plot(cycle_array, pop_array, label='mean amount of samples', color='r', marker='.')
    # ax2.set_ylabel('average number of samples of all data sets', color='r')
    # ax2.tick_params('y', colors='r')
    # ax2.legend()
    # # IF WE WANT TO USE POINTS FROM ONLY THE CYCLE CYCLE
    # # plt.xlim([0,20])
    # plt.show()
    #
    #
    #
    # fig, ax1 = plt.subplots()
    # ax1.plot(cycle_array, phi_array[0], label='Sister', marker='.')
    # ax1.plot(cycle_array, phi_array[1], label='Non-Sister', marker='.')
    # ax1.plot(cycle_array, phi_array[2], label='Control', marker='.')
    # ax1.set_xlabel('cycle number')
    # ax1.set_ylabel('phi value', color='b')
    # ax1.tick_params('y', colors='b')
    # ax1.legend()
    # # IF WE WANT TO USE POINTS FROM ONLY THE CYCLE CYCLE
    # # plt.ylim([0, 4])
    # # plt.xlim([0,20])
    #
    #
    # ax2 = ax1.twinx()
    # ax2.plot(cycle_array, pop_array, label='mean amount of samples', color='r', marker='.')
    # ax2.set_ylabel('average number of samples of all data sets', color='r')
    # ax2.tick_params('y', colors='r')
    # ax2.legend()
    # # IF WE WANT TO USE POINTS FROM ONLY THE CYCLE CYCLE
    # # plt.xlim([0,20])
    # plt.show()
    

if __name__ == '__main__':
    main()
