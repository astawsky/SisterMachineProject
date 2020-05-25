
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

    dataid=0
    for keyA, keyB in zip(list(metadatastruct.A_dict_sis.keys())[:87], list(metadatastruct.B_dict_sis.keys())[:87]):
        least_length = min(len(metadatastruct.A_dict_sis[keyA]), len(metadatastruct.B_dict_sis[keyB]))
        diff_df = np.abs(metadatastruct.A_dict_sis[keyA][:least_length]-metadatastruct.B_dict_sis[keyB][:least_length])
        diff_df = diff_df/np.mean(diff_df)
        plt.figure()
        for param in diff_df.keys():
            plt.plot(diff_df[param], marker='.', label=param)
        plt.title(keyA + ' and ' + keyB)
        plt.xlabel('Generation (Cycle) Number')
        plt.ylabel('absolute value of difference divided by mean of difference')
        plt.legend()
        plt.savefig('/Users/alestawsky/PycharmProjects/untitled/Differences of cycle params divided by their respective'
            +' means/Sisters/'+str(dataid), dpi=300)
        plt.close()
        dataid=dataid+1

    dataid = 0
    for keyA, keyB in zip(list(metadatastruct.A_dict_non_sis.keys())[:87], list(metadatastruct.B_dict_non_sis.keys())[:87]):
        least_length = min(len(metadatastruct.A_dict_non_sis[keyA]), len(metadatastruct.B_dict_non_sis[keyB]))
        diff_df = np.abs(
            metadatastruct.A_dict_non_sis[keyA][:least_length] - metadatastruct.B_dict_non_sis[keyB][:least_length])
        diff_df = diff_df / np.mean(diff_df)
        plt.figure()
        for param in diff_df.keys():
            plt.plot(diff_df[param], marker='.', label=param)
        plt.title(keyA + ' and ' + keyB)
        plt.xlabel('Generation (Cycle) Number')
        plt.ylabel('absolute value of difference divided by mean of difference')
        plt.legend()
        plt.savefig('/Users/alestawsky/PycharmProjects/untitled/Differences of cycle params divided by their respective'
                    + ' means/Nonsisters/' + str(dataid), dpi=300)
        plt.close()
        dataid = dataid + 1

    dataid = 0
    for keyA, keyB in zip(list(metadatastruct.A_dict_both.keys())[:87], list(metadatastruct.B_dict_both.keys())[:87]):
        least_length = min(len(metadatastruct.A_dict_both[keyA]), len(metadatastruct.B_dict_both[keyB]))
        diff_df = np.abs(
            metadatastruct.A_dict_both[keyA][:least_length] - metadatastruct.B_dict_both[keyB][:least_length])
        diff_df = diff_df / np.mean(diff_df)
        plt.figure()
        for param in diff_df.keys():
            plt.plot(diff_df[param], marker='.', label=param)
        plt.title(keyA + ' and ' + keyB)
        plt.xlabel('Generation (Cycle) Number')
        plt.ylabel('absolute value of difference divided by mean of difference')
        plt.legend()
        plt.savefig('/Users/alestawsky/PycharmProjects/untitled/Differences of cycle params divided by their respective'
                    + ' means/Control/' + str(dataid), dpi=300)
        plt.close()
        dataid = dataid + 1


if __name__ == '__main__':
    main()
