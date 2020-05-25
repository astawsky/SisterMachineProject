
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

import scipy.stats as stats
from sklearn.linear_model import LinearRegression


def main():
    # Import the Refined Data
    pickle_in = open("metastructdata.pickle", "rb")
    metadatastruct = pickle.load(pickle_in)
    pickle_in.close()

    # PLOTTING THE CYCLE PARAMETERS OF A AND B IN THE SAME GRAPH (SISTERS)
    ind = 0
    for keyA, keyB in zip(list(metadatastruct.A_dict_sis.keys())[:87], list(metadatastruct.B_dict_sis.keys())[:87]):
        for param in metadatastruct.A_dict_sis[keyA].keys():
            plt.figure()
            plt.plot(metadatastruct.A_dict_sis[keyA][param], marker='.', label='A')
            plt.plot(metadatastruct.B_dict_sis[keyB][param], marker='.', label='B')
            plt.title(keyA + ' and ' + keyB)
            plt.xlabel('Generation (Cycle) Number')
            plt.ylabel(param)
            plt.legend()
            plt.savefig('/Users/alestawsky/PycharmProjects/untitled/Cycle_data_compared/Sisters/'
                + str(param) + '/' + str(ind), dpi=300)
            plt.close()
        ind = ind + 1

    # PLOTTING THE CYCLE PARAMETERS OF A AND B IN THE SAME GRAPH (NONSISTERS)
    ind = 0
    for keyA, keyB in zip(list(metadatastruct.A_dict_non_sis.keys())[:87], list(metadatastruct.B_dict_non_sis.keys())[:87]):
        for param in metadatastruct.A_dict_non_sis[keyA].keys():
            plt.figure()
            plt.plot(metadatastruct.A_dict_non_sis[keyA][param], marker='.', label='A')
            plt.plot(metadatastruct.B_dict_non_sis[keyB][param], marker='.', label='B')
            plt.title(keyA + ' and ' + keyB)
            plt.xlabel('Generation (Cycle) Number')
            plt.ylabel(param)
            plt.legend()
            plt.savefig('/Users/alestawsky/PycharmProjects/untitled/Cycle_data_compared/NonSisters/'
                        + str(param) + '/' + str(ind), dpi=300)
            plt.close()
        ind = ind + 1

    # PLOTTING THE CYCLE PARAMETERS OF A AND B IN THE SAME GRAPH (CONTROL)
    ind = 0
    for keyA, keyB in zip(list(metadatastruct.A_dict_both.keys())[:87],
                          list(metadatastruct.B_dict_both.keys())[:87]):
        for param in metadatastruct.A_dict_both[keyA].keys():
            plt.figure()
            plt.plot(metadatastruct.A_dict_both[keyA][param], marker='.', label='A')
            plt.plot(metadatastruct.B_dict_both[keyB][param], marker='.', label='B')
            plt.title(keyA + ' and ' + keyB)
            plt.xlabel('Generation (Cycle) Number')
            plt.ylabel(param)
            plt.legend()
            plt.savefig('/Users/alestawsky/PycharmProjects/untitled/Cycle_data_compared/Control/'
                        + str(param) + '/' + str(ind), dpi=300)
            plt.close()
        ind = ind + 1


if __name__ == '__main__':
    main()
