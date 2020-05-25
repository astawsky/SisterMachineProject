

from __future__ import print_function

import numpy as np
import argparse
import sys,math
import glob
import matplotlib.pyplot as plt

import pickle

import sistercellclass as ssc

import CALCULATETHEBETAS
import os

def main():


    # Import the Refined Data
    pickle_in = open("metastructdata.pickle", "rb")
    metadatastruct = pickle.load(pickle_in)
    pickle_in.close()

    metadatastruct.GraphTrajCorrelations()

    # plt.plot(metadatastruct.non_sis_acfA_array[0]['length_birth'], label='A')
    # plt.plot(metadatastruct.non_sis_acfB_array[0]['length_birth'], label='B')
    # plt.xlabel('Division Generations')
    # plt.ylabel('correlation of {} correlation'.format("'length_birth'"))
    # plt.title('sister {}'.format(0))
    # plt.show()
    #
    # key = 'length_birth'
    # # my_dir = '/Users/alestawsky/PycharmProjects/untitled/Sisters_autocorrelations'
    # my_dir = '/Users/alestawsky/PycharmProjects/untitled/Nonsisters_autocorrelations'
    # for dataid in range(len(metadatastruct.non_sis_acfA_array)):
    #     plt.plot(metadatastruct.non_sis_acfA_array[dataid][key], label='A')
    #     plt.plot(metadatastruct.non_sis_acfB_array[dataid][key], label='B')
    # plt.xlabel('Division Generations')
    # plt.ylabel('correlation of {} correlation'.format(key))
    # plt.title('non-sister {}'.format(dataid))
    # # # plt.legend(handles=[ff, gg])
    # # plt.show()
    # plt.savefig(my_dir + '_non_sisters_' + key +  '.png', bbox_inches='tight')
    # plt.clf()
    #
    # for dataid in range(len(metadatastruct.non_sis_acfA_array)):
    #     plt.plot(metadatastruct.non_sis_acfA_array[dataid][key], label='A')
    #     plt.plot(metadatastruct.non_sis_acfB_array[dataid][key], label='B')
    #     plt.xlabel('Division Generations')
    #     plt.ylabel('correlation of {} correlation'.format(key))
    #     plt.title('non-sister {}'.format(dataid))
    #     # # plt.legend(handles=[ff, gg])
    #     # plt.show()
    #     plt.savefig(my_dir + str(dataid) + '_non_sisters_' + key +  '.png', bbox_inches='tight')
    #     plt.clf()
    #
    # # This is for the pooled data
    # pooled_dictA = 'columns_pooled_over_cycles_A_sis'
    # pooled_dictB = 'columns_pooled_over_cycles_B_sis'
    # key = 'length_birth'
    # # my_dir = '/Users/alestawsky/PycharmProjects/untitled/Sisters_autocorrelations'
    # my_dir = '/Users/alestawsky/PycharmProjects/untitled/Nonsisters_autocorrelations'
    # for cycle_number in range(len(metadatastruct.sis_pooled_data_dict[pooled_dictA][key])):
    #     plt.plot(metadatastruct.sis_pooled_data_dict[pooled_dictA][key][dataid], label='A')
    #     plt.plot(metadatastruct.non_sis_acfB_array[dataid][key], label='B')
    # plt.xlabel('Division Generations')
    # plt.ylabel('correlation of {} correlation'.format(key))
    # plt.title('non-sister {}'.format(dataid))
    # # # plt.legend(handles=[ff, gg])
    # # plt.show()
    # plt.savefig(my_dir + '_non_sisters_' + key + '.png', bbox_inches='tight')
    # plt.clf()
    #
    # for dataid in range(len(metadatastruct.non_sis_acfA_array)):
    #     plt.plot(metadatastruct.non_sis_acfA_array[dataid][key], label='A')
    #     plt.plot(metadatastruct.non_sis_acfB_array[dataid][key], label='B')
    #     plt.xlabel('Division Generations')
    #     plt.ylabel('correlation of {} correlation'.format(key))
    #     plt.title('non-sister {}'.format(dataid))
    #     # # plt.legend(handles=[ff, gg])
    #     # plt.show()
    #     plt.savefig(my_dir + str(dataid) + '_non_sisters_' + key + '.png', bbox_inches='tight')
    #     plt.clf()
    #
    # # # plot Betas of A vs betas of B
    # # plt.()


if __name__ == '__main__':
    main()
