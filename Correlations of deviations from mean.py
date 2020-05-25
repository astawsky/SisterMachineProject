
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

    same_global = Population.mother_dfs[0].copy()
    same_trap = Population.mother_dfs[0].copy()

    print(same_trap[['trap_avg_'+name for name in Population._variable_names]])

    same_global = same_global[Population._variable_names] - same_global[Population._variable_names].mean()
    for col in Population._variable_names:
        same_trap[col] = same_trap[col] - same_trap['trap_avg_'+col]
    same_trap = same_trap[Population._variable_names]
    
    mom_global = Population.mother_dfs[0].copy()
    daughter_global = Population.daughter_dfs[0].copy()
    mom_trap = Population.mother_dfs[0].copy()
    daughter_trap = Population.daughter_dfs[0].copy()
    
    mom_global = mom_global[Population._variable_names] - mom_global[Population._variable_names].mean()
    for col in Population._variable_names:
        mom_trap[col] = mom_trap[col] - mom_trap['trap_avg_'+col]
    mom_trap = mom_trap[Population._variable_names]

    daughter_global = daughter_global[Population._variable_names] - daughter_global[Population._variable_names].mean()
    for col in Population._variable_names:
        daughter_trap[col] = daughter_trap[col] - daughter_trap['trap_avg_' + col]
    daughter_trap = daughter_trap[Population._variable_names]
    
    # same_trap = same_trap[Population._variable_names].copy() - same_trap[['trap_avg_'+name for name in Population._variable_names]]
    #
    # print(same_global.isnull(), same_trap.isnull())
    # exit()

    # for col in Population._variable_names:
    #     if col


    """ we have three different types of frameworks, the all logs, the all normals and the log-normal mixes """
    # # Same-cell for normal
    # Population.plot_same_cell_correlations(df=same_global, variables=Population._variable_names,
    #                                        labels=Population._variable_symbols.loc['without subscript, normalized lengths'])

    # Population.plot_same_cell_correlations(df=same_trap, variables=Population._variable_names,
    #                                        labels=Population._variable_symbols.loc['without subscript, normalized lengths'])

    # Mother-Daughter
    Population.plot_relationship_correlations(df_1=mom_global, df_2=daughter_global, df_1_variables=Population._variable_names,
                                              df_2_variables=Population._variable_names,
                                              x_labels=Population._variable_symbols.loc['_n, normalized lengths'],
                                              y_labels=Population._variable_symbols.loc['_{n+1}, normalized lengths'])

    Population.plot_relationship_correlations(df_1=mom_trap, df_2=daughter_trap, df_1_variables=Population._variable_names,
                                              df_2_variables=Population._variable_names,
                                              x_labels=Population._variable_symbols.loc['_n, normalized lengths'],
                                              y_labels=Population._variable_symbols.loc['_{n+1}, normalized lengths'])



if __name__ == '__main__':
    main()
