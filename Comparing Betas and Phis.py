
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

    # PLOT THE DIFFERENCE IN BETAS, CAN'T SEE ANY DIFFERENCE BETWEEN THE DISTRIBUTIONS
    plt.figure()
    y = np.log10(np.abs(np.array(metadatastruct.non_sis_beta_A_array) - np.array(metadatastruct.non_sis_beta_B_array)))
    res = plt.hist(y, weights=np.ones_like(y) / float(len(y)))
    y = np.log10(np.abs(np.array(metadatastruct.sis_beta_A_array) - np.array(metadatastruct.sis_beta_B_array)))
    res1 = plt.hist(y, weights=np.ones_like(y) / float(len(y)))
    y=np.log10(np.abs(np.array(metadatastruct.both_beta_A_array[:87]) - np.array(metadatastruct.both_beta_B_array[:87])))
    res2 = plt.hist(y, weights=np.ones_like(y) / float(len(y)))
    plt.close()
    plt.figure()
    x = [(res[1][i]+res[1][i+1])/np.abs(res[1][i]-res[1][i+1]) for i in range(len(res[1])-1)]
    plt.plot(x, res[0], label='Sisters', marker='.')
    x = [(res1[1][i]+res1[1][i+1])/np.abs(res1[1][i]-res1[1][i+1]) for i in range(len(res1[1]) - 1)]
    plt.plot(x, res1[0], label='Non-Sisters', marker='^', linestyle='--')
    x = [(res2[1][i]+res2[1][i+1])/np.abs(res2[1][i]-res2[1][i+1]) for i in range(len(res2[1]) - 1)]
    plt.plot(x, res2[0], label='Control', marker='*')
    plt.legend()
    plt.xlabel('log10 values of difference between betas')
    plt.ylabel('Normalized Histogram')
    plt.title('Difference between betas in cells')
    plt.show()

    # PLOT THE DIFFERENCE IN PHI-STAR, SEEMS AS IF THE SISTERS HAVE MORE OF A DIFFERENCE BETWEEN THEM
    # ALSO THERE IS ONE SAMPLE OF THE CONTROL PAIRS TO THE LEFT THAT APPEARS LIKE THEY ARE THE SAME
    plt.figure()
    y = np.log10(np.abs(np.array(metadatastruct.non_sis_phi_star_A_array) - np.array(metadatastruct.non_sis_phi_star_B_array)))
    res = plt.hist(y, weights=np.ones_like(y) / float(len(y)))
    y = np.log10(np.abs(np.array(metadatastruct.sis_phi_star_A_array) - np.array(metadatastruct.sis_phi_star_B_array)))
    res1 = plt.hist(y, weights=np.ones_like(y) / float(len(y)))
    y = np.log10(
        np.abs(np.array(metadatastruct.both_phi_star_A_array[:87]) - np.array(metadatastruct.both_phi_star_B_array[:87])))
    res2 = plt.hist(y, weights=np.ones_like(y) / float(len(y)))
    plt.close()
    plt.figure()
    x = [(res[1][i] + res[1][i + 1]) / np.abs(res[1][i] - res[1][i + 1]) for i in range(len(res[1]) - 1)]
    plt.plot(x, res[0], label='Sisters', marker='.')
    x = [(res1[1][i] + res1[1][i + 1]) / np.abs(res1[1][i] - res1[1][i + 1]) for i in range(len(res1[1]) - 1)]
    plt.plot(x, res1[0], label='Non-Sisters', marker='^', linestyle='--')
    x = [(res2[1][i] + res2[1][i + 1]) / np.abs(res2[1][i] - res2[1][i + 1]) for i in range(len(res2[1]) - 1)]
    plt.plot(x, res2[0], label='Control', marker='*')
    plt.legend()
    plt.xlabel('log10 values of difference between phi_stars')
    plt.ylabel('Normalized Histogram')
    plt.title('Difference between phi_stars in cells')
    plt.show()

    # SCATTER THE POINTS THAT HAVE THE BETA_A AND BETA_B AS THE X,Y-VALUES
    # points = [a + b for a,b in zip(metadatastruct.sis_beta_A_array[:87], metadatastruct.sis_beta_B_array[:87])]
    # reg = LinearRegression().fit(points, metadatastruct.sis_beta_B_array[:87])
    # print(reg.intercept_,reg.coef_, reg.get_params)
    sispear = stats.pearsonr(metadatastruct.sis_beta_A_array[:87], metadatastruct.sis_beta_B_array[:87])
    nonpear = stats.pearsonr(metadatastruct.non_sis_beta_A_array[:87], metadatastruct.non_sis_beta_B_array[:87])
    bothpear = stats.pearsonr(metadatastruct.both_beta_A_array[:87], metadatastruct.both_beta_B_array[:87])
    sislabel = r'Sister, corr.coeff.={:.2e}, p-value={:.2e}'.format(sispear[0], sispear[1])
    nonlabel = r'Non Sister, corr.coeff.={:.2e}, p-value={:.2e}'.format(nonpear[0], nonpear[1])
    bothlabel = r'Control, corr.coeff.={:.2e}, p-value={:.2e}'.format(bothpear[0], bothpear[1])
    plt.figure()
    plt.scatter(metadatastruct.sis_beta_A_array[:87], metadatastruct.sis_beta_B_array[:87], label = sislabel)
    plt.scatter(metadatastruct.non_sis_beta_A_array[:87], metadatastruct.non_sis_beta_B_array[:87], label=nonlabel)
    plt.scatter(metadatastruct.both_beta_A_array[:87], metadatastruct.both_beta_B_array[:87], label=bothlabel)
    plt.title('Are the pair betas correlated in some way?')
    plt.xlabel('Beta_A')
    plt.ylabel('Beta_B')
    plt.legend()
    plt.show()

    sispear = stats.pearsonr(metadatastruct.sis_phi_star_A_array[:87], metadatastruct.sis_phi_star_B_array[:87])
    nonpear = stats.pearsonr(metadatastruct.non_sis_phi_star_A_array[:87], metadatastruct.non_sis_phi_star_B_array[:87])
    bothpear = stats.pearsonr(metadatastruct.both_phi_star_A_array[:87], metadatastruct.both_phi_star_B_array[:87])
    sislabel = r'Sister, corr.coeff.={:.2e}, p-value={:.2e}'.format(sispear[0], sispear[1])
    nonlabel = r'Non Sister, corr.coeff.={:.2e}, p-value={:.2e}'.format(nonpear[0], nonpear[1])
    bothlabel = r'Control, corr.coeff.={:.2e}, p-value={:.2e}'.format(bothpear[0], bothpear[1])
    plt.figure()
    plt.scatter(metadatastruct.sis_phi_star_A_array[:87], metadatastruct.sis_phi_star_B_array[:87], label=sislabel)
    plt.scatter(metadatastruct.non_sis_phi_star_A_array[:87], metadatastruct.non_sis_phi_star_B_array[:87], label=nonlabel)
    plt.scatter(metadatastruct.both_phi_star_A_array[:87], metadatastruct.both_phi_star_B_array[:87], label=bothlabel)
    plt.title('Are the pair phi-stars correlated in some way?')
    plt.xlabel('phi_star_A')
    plt.ylabel('phi_star_B')
    plt.legend()
    plt.show()
    # plt.axvline



    # PLOT THE DIFFERENCE IN BETAS, CAN'T SEE ANY DIFFERENCE BETWEEN THE DISTRIBUTIONS
    plt.figure()
    y = np.log10(np.abs(np.array(metadatastruct.n_beta_A_array) - np.array(metadatastruct.n_beta_B_array)))
    res = plt.hist(y, weights=np.ones_like(y) / float(len(y)))
    y = np.log10(np.abs(np.array(metadatastruct.s_beta_A_array) - np.array(metadatastruct.s_beta_B_array)))
    res1 = plt.hist(y, weights=np.ones_like(y) / float(len(y)))
    y=np.log10(np.abs(np.array(metadatastruct.b_beta_A_array[:87]) - np.array(metadatastruct.b_beta_B_array[:87])))
    res2 = plt.hist(y, weights=np.ones_like(y) / float(len(y)))
    plt.close()
    plt.figure()
    x = [(res[1][i]+res[1][i+1])/np.abs(res[1][i]-res[1][i+1]) for i in range(len(res[1])-1)]
    plt.plot(x, res[0], label='Sisters', marker='.')
    x = [(res1[1][i]+res1[1][i+1])/np.abs(res1[1][i]-res1[1][i+1]) for i in range(len(res1[1]) - 1)]
    plt.plot(x, res1[0], label='Non-Sisters', marker='^', linestyle='--')
    x = [(res2[1][i]+res2[1][i+1])/np.abs(res2[1][i]-res2[1][i+1]) for i in range(len(res2[1]) - 1)]
    plt.plot(x, res2[0], label='Control', marker='*')
    plt.legend()
    plt.xlabel('log10 values of difference between betas')
    plt.ylabel('Normalized Histogram')
    plt.title('Difference between betas in cells (87 res[1])')
    plt.show()

    # PLOT THE DIFFERENCE IN PHI-STAR, SEEMS AS IF THE SISTERS HAVE MORE OF A DIFFERENCE BETWEEN THEM
    # ALSO THERE IS ONE SAMPLE OF THE CONTROL PAIRS TO THE LEFT THAT APPEARS LIKE THEY ARE THE SAME
    plt.figure()
    y = np.log10(np.abs(np.array(metadatastruct.n_phi_star_A_array) - np.array(metadatastruct.n_phi_star_B_array)))
    res = plt.hist(y, weights=np.ones_like(y) / float(len(y)))
    y = np.log10(np.abs(np.array(metadatastruct.s_phi_star_A_array) - np.array(metadatastruct.s_phi_star_B_array)))
    res1 = plt.hist(y, weights=np.ones_like(y) / float(len(y)))
    y = np.log10(
        np.abs(np.array(metadatastruct.b_phi_star_A_array[:87]) - np.array(metadatastruct.b_phi_star_B_array[:87])))
    res2 = plt.hist(y, weights=np.ones_like(y) / float(len(y)))
    plt.close()
    plt.figure()
    x = [(res[1][i] + res[1][i + 1]) / np.abs(res[1][i] - res[1][i + 1]) for i in range(len(res[1]) - 1)]
    plt.plot(x, res[0], label='Sisters', marker='.')
    x = [(res1[1][i] + res1[1][i + 1]) / np.abs(res1[1][i] - res1[1][i + 1]) for i in range(len(res1[1]) - 1)]
    plt.plot(x, res1[0], label='Non-Sisters', marker='^', linestyle='--')
    x = [(res2[1][i] + res2[1][i + 1]) / np.abs(res2[1][i] - res2[1][i + 1]) for i in range(len(res2[1]) - 1)]
    plt.plot(x, res2[0], label='Control', marker='*')
    plt.legend()
    plt.xlabel('log10 values of difference between phi_stars')
    plt.ylabel('Normalized Histogram')
    plt.title('Difference between phi_stars in cells (87 res[1])')
    plt.show()
    
    # SCATTER THE POINTS THAT HAVE THE BETA_A AND BETA_B AS THE X,Y-VALUES
    # points = [a + b for a,b in zip(metadatastruct.s_beta_A_array[:87], metadatastruct.s_beta_B_array[:87])]
    # reg = LinearRegression().fit(points, metadatastruct.s_beta_B_array[:87])
    # print(reg.intercept_,reg.coef_, reg.get_params)
    sispear = stats.pearsonr(metadatastruct.s_beta_A_array, metadatastruct.s_beta_B_array)
    nonpear = stats.pearsonr(metadatastruct.n_beta_A_array, metadatastruct.n_beta_B_array)
    bothpear = stats.pearsonr(metadatastruct.b_beta_A_array, metadatastruct.b_beta_B_array)
    print(len(metadatastruct.s_beta_A_array),len(metadatastruct.n_beta_A_array),len(metadatastruct.b_beta_A_array))
    sislabel = r'Sister, corr.coeff.={:.2e}, p-value={:.2e}'.format(sispear[0], sispear[1])
    nonlabel = r'Non Sister, corr.coeff.={:.2e}, p-value={:.2e}'.format(nonpear[0], nonpear[1])
    bothlabel = r'Control, corr.coeff.={:.2e}, p-value={:.2e}'.format(bothpear[0], bothpear[1])
    plt.figure()
    plt.scatter(metadatastruct.s_beta_A_array[:87], metadatastruct.s_beta_B_array[:87], label = sislabel)
    plt.scatter(metadatastruct.n_beta_A_array[:87], metadatastruct.n_beta_B_array[:87], label=nonlabel)
    plt.scatter(metadatastruct.b_beta_A_array[:87], metadatastruct.b_beta_B_array[:87], label=bothlabel)
    plt.title('Are the pair betas correlated in some way? (40 gens or more)')
    plt.xlabel('Beta_A')
    plt.ylabel('Beta_B')
    plt.legend()
    plt.show()

    sispear = stats.pearsonr(metadatastruct.s_phi_star_A_array[:87], metadatastruct.s_phi_star_B_array[:87])
    nonpear = stats.pearsonr(metadatastruct.n_phi_star_A_array[:87], metadatastruct.n_phi_star_B_array[:87])
    bothpear = stats.pearsonr(metadatastruct.b_phi_star_A_array[:87], metadatastruct.b_phi_star_B_array[:87])
    print(len(metadatastruct.s_beta_A_array),len(metadatastruct.n_beta_A_array),len(metadatastruct.b_beta_A_array))
    sislabel = r'Sister, corr.coeff.={:.2e}, p-value={:.2e}'.format(sispear[0], sispear[1])
    nonlabel = r'Non Sister, corr.coeff.={:.2e}, p-value={:.2e}'.format(nonpear[0], nonpear[1])
    bothlabel = r'Control, corr.coeff.={:.2e}, p-value={:.2e}'.format(bothpear[0], bothpear[1])
    plt.figure()
    plt.scatter(metadatastruct.s_phi_star_A_array[:87], metadatastruct.s_phi_star_B_array[:87], label=sislabel)
    plt.scatter(metadatastruct.n_phi_star_A_array[:87], metadatastruct.n_phi_star_B_array[:87], label=nonlabel)
    plt.scatter(metadatastruct.b_phi_star_A_array[:87], metadatastruct.b_phi_star_B_array[:87], label=bothlabel)
    plt.title('Are the pair phi-stars correlated in some way? (40 gens or more)')
    plt.xlabel('phi_star_A')
    plt.ylabel('phi_star_B')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()

