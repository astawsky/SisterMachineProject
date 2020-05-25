
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys, math
import glob
import pickle
import os
import scipy.stats as stats
from scipy import signal
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns
import itertools
import NewSisterCellClass as ssc


def concat_the_AB_traces_for_symmetry(df_A, df_B):
    print(len(df_A.copy()), len(df_B.copy()))
    both_A = pd.concat([df_A.copy(), df_B.copy()], axis=0)
    both_B = pd.concat([df_B.copy(), df_A.copy()], axis=0)
    print(len(both_A), len(both_B))
    return both_A, both_B


""" input the already centered factor and target dataframe """
def linear_regression_framework(factor_df, target_df, fit_intercept):

    # Make regression matrix in a dataframe representation
    df_matrix = pd.DataFrame(index=target_df.columns, columns=factor_df.columns)

    # arrays where we will keep the information
    scores = dict()
    intercepts = dict()
    predictions = dict()

    # loop over all the regressions we'll be doing
    for t_v in target_df.columns:
        # get the fit
        reg = LinearRegression(fit_intercept=fit_intercept).fit(factor_df, target_df[t_v])
        # save the information
        scores.update({t_v: round(reg.score(factor_df, target_df[t_v]), 3)})
        intercepts.update({t_v: round(reg.intercept_, 3)})
        predictions.update({t_v: reg.predict(factor_df)})

        coefficients = [round(coef, 3) for coef in reg.coef_]
        # put it in the matrix
        df_matrix.loc[t_v] = coefficients

    return df_matrix, scores, intercepts, predictions


def main():
    # Import the objects we want
    pickle_in = open("NewSisterCellClass_Population.pickle", "rb")
    Population = pickle.load(pickle_in)
    pickle_in.close()

    pickle_in = open("NewSisterCellClass_Sister.pickle", "rb")
    Sister = pickle.load(pickle_in)
    pickle_in.close()

    pickle_in = open("NewSisterCellClass_Nonsister.pickle", "rb")
    Nonsister = pickle.load(pickle_in)
    pickle_in.close()

    pickle_in = open("NewSisterCellClass_Control.pickle", "rb")
    Control = pickle.load(pickle_in)
    pickle_in.close()

    # trap averages
    A_sis, B_sis = concat_the_AB_traces_for_symmetry(df_A=Population.mother_dfs[0].copy(), df_B=Population.mother_dfs[0].copy())

    # A_sis, B_sis = concat_the_AB_traces_for_symmetry(df_A=Sister.A_intra_gen_bacteria[0].copy(), df_B=Sister.B_intra_gen_bacteria[0].copy())

    trap_sis = pd.DataFrame(index=['trap_avg_'+name for name in Population._variable_names],
                            columns=['trap_avg_'+name for name in Population._variable_names])

    for var1 in trap_sis.columns:
        for var2 in trap_sis.index:

            trap_sis[var1].loc[var2] = round(stats.pearsonr(A_sis[str(var1)], B_sis[str(var2)])[0], 3)

    print(trap_sis)

    print(dict(zip(trap_sis.index, Population._variable_symbols.loc['without subscript'])))

    trap_sis = trap_sis.rename(index=dict(zip(trap_sis.index, Population._variable_symbols.loc['without subscript'])),
                    columns=dict(zip(trap_sis.columns, Population._variable_symbols.loc['without subscript'])))

    trap_sis = trap_sis[trap_sis.columns].astype(float)


    # some = A_sis['trap_avg_generationtime']*B_sis['trap_avg_growth_rate']
    # sns.regplot(A_sis['trap_avg_generationtime'], B_sis['trap_avg_growth_rate'],
    #             label=round(stats.pearsonr(A_sis['trap_avg_generationtime'], B_sis['trap_avg_growth_rate'])[0], 3))
    # plt.legend()
    # plt.show()
    # plt.close()
    # sns.regplot(A_sis['log_trap_avg_generationtime'], B_sis['trap_avg_growth_rate'],
    #             label=round(stats.pearsonr(A_sis['trap_avg_generationtime'], B_sis['trap_avg_growth_rate'])[0], 3))
    # plt.legend()
    # plt.show()
    # plt.close()
    # plt.hist(some, label='{}, {}'.format(np.mean(some), np.std(some)))
    # plt.legend()
    # plt.show()
    # plt.close()
    sns.regplot(A_sis['trap_avg_fold_growth'], B_sis['log_trap_avg_division_ratio'],
                label=round(stats.pearsonr(A_sis['trap_avg_fold_growth'], B_sis['log_trap_avg_division_ratio'])[0], 3))
    plt.legend()
    plt.show()
    plt.close()
    other = A_sis['trap_avg_fold_growth'] + B_sis['log_trap_avg_division_ratio']
    plt.hist(other, label='{}, {}'.format(np.mean(other), np.std(other)))
    plt.legend()
    plt.show()
    plt.close()
    exit()
    some = A_sis['fold_growth'] * B_sis['division_ratio']
    plt.hist(some, label='{}, {}'.format(np.mean(some), np.std(some)))
    plt.legend()
    plt.show()
    exit()


    sns.heatmap(data=trap_sis, annot=True)
    plt.title('All data trap averages')
    plt.savefig('All data trap averages.png', dpi=300)
    # plt.show()
    plt.close()





    A_sis, B_sis = concat_the_AB_traces_for_symmetry(df_A=Sister.A_intra_gen_bacteria[0].copy(), df_B=Sister.B_intra_gen_bacteria[0].copy())

    traj_sis = pd.DataFrame(index=['traj_avg_' + name for name in Population._variable_names],
                            columns=['traj_avg_' + name for name in Population._variable_names])

    for var1 in traj_sis.columns:
        for var2 in traj_sis.index:
            traj_sis[var1].loc[var2] = round(stats.pearsonr(A_sis[str(var1)], B_sis[str(var2)])[0], 3)

    print(traj_sis)

    print(dict(zip(traj_sis.index, Population._variable_symbols.loc['without subscript'])))

    traj_sis = traj_sis.rename(index=dict(zip(traj_sis.index, Population._variable_symbols.loc['without subscript'])),
                               columns=dict(zip(traj_sis.columns, Population._variable_symbols.loc['without subscript'])))

    traj_sis = traj_sis[traj_sis.columns].astype(float)

    sns.heatmap(data=traj_sis, annot=True)
    plt.title('Sister trajectory averages')
    plt.savefig('Sister trajectory averages.png', dpi=300)
    # plt.show()
    plt.close()






    A_sis, B_sis = concat_the_AB_traces_for_symmetry(df_A=Nonsister.A_intra_gen_bacteria[0].copy(), df_B=Nonsister.B_intra_gen_bacteria[0].copy())

    traj_sis = pd.DataFrame(index=['traj_avg_' + name for name in Population._variable_names],
                            columns=['traj_avg_' + name for name in Population._variable_names])

    for var1 in traj_sis.columns:
        for var2 in traj_sis.index:
            traj_sis[var1].loc[var2] = round(stats.pearsonr(A_sis[str(var1)], B_sis[str(var2)])[0], 3)

    print(traj_sis)

    print(dict(zip(traj_sis.index, Population._variable_symbols.loc['without subscript'])))

    traj_sis = traj_sis.rename(index=dict(zip(traj_sis.index, Population._variable_symbols.loc['without subscript'])),
                               columns=dict(zip(traj_sis.columns, Population._variable_symbols.loc['without subscript'])))

    traj_sis = traj_sis[traj_sis.columns].astype(float)

    sns.heatmap(data=traj_sis, annot=True)
    plt.title('Nonsister trajectory averages')
    plt.savefig('Nonsister trajectory averages.png', dpi=300)
    # plt.show()
    plt.close()





    A_sis, B_sis = concat_the_AB_traces_for_symmetry(df_A=Control.A_intra_gen_bacteria[0].copy(), df_B=Control.B_intra_gen_bacteria[0].copy())

    traj_sis = pd.DataFrame(index=['traj_avg_' + name for name in Population._variable_names],
                            columns=['traj_avg_' + name for name in Population._variable_names])

    for var1 in traj_sis.columns:
        for var2 in traj_sis.index:
            traj_sis[var1].loc[var2] = round(stats.pearsonr(A_sis[str(var1)], B_sis[str(var2)])[0], 3)

    print(traj_sis)

    print(dict(zip(traj_sis.index, Population._variable_symbols.loc['without subscript'])))

    traj_sis = traj_sis.rename(index=dict(zip(traj_sis.index, Population._variable_symbols.loc['without subscript'])),
                               columns=dict(zip(traj_sis.columns, Population._variable_symbols.loc['without subscript'])))

    traj_sis = traj_sis[traj_sis.columns].astype(float)

    sns.heatmap(data=traj_sis, annot=True)
    plt.title('Control trajectory averages')
    plt.savefig('Control trajectory averages.png', dpi=300)
    # plt.show()
    plt.close()
    
    
    
    
    # print(A_sis[['trap_avg_'+name for name in Population._variable_names]])
    #
    # print(np.corrcoef(A_sis[['trap_avg_'+name for name in Population._variable_names]].T, B_sis[['trap_avg_'+name for name in Population._variable_names]].T))
    #
    # print(A_sis.columns)



if __name__ == '__main__':
    main()
