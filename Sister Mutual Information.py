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
import sklearn
from sklearn.model_selection import train_test_split
import seaborn as sns
import itertools
import NewSisterCellClass as ssc
import pymc3 as pm
from scipy.special import kn as bessel_function_of_second_kind
from scipy.special import comb as combinations


def subtract_traj_averages(df, columns_names, log_vars):
    df_new = pd.DataFrame(columns=columns_names)
    for col in columns_names:
        if col in log_vars:
            df_new[col] = np.log(df[col]) - df['log_traj_avg_' + col]
        else:
            df_new[col] = df[col] - df['traj_avg_' + col]

    return df_new


def subtract_trap_averages(df, columns_names, log_vars):
    df_new = pd.DataFrame(columns=columns_names)
    for col in columns_names:
        if col in log_vars:
            df_new[col] = np.log(df[col]) - df['log_trap_avg_' + col]
        else:
            df_new[col] = df[col] - df['trap_avg_' + col]

    return df_new


def subtract_global_averages(df, columns_names, log_vars):
    df_new = pd.DataFrame(columns=columns_names)
    for col in columns_names:
        if col in log_vars:
            df_new[col] = np.log(df[col]) - np.mean(np.log(df[col]))
        else:
            df_new[col] = df[col] - np.mean(df[col])

    return df_new


def normal_entropy(std):
    # NOOOOOOOT This one came from the Normal page on Wikipedia
    return np.log(std*np.sqrt(2*np.pi*np.exp(1)))


def multivariate_normal_KL_divergence(mean_0, cov_0, mean_1, cov_1):
    # Definition of Mutual Information! if N_0 is the multivariate distribution and N_1 is the product of all the marginal distributions

    # D_{KL}(N_0||N_1), which is not necessarily the same as D_{KL}(N_1||N_0)!

    # This one came from the Multivariate Normal page on Wikipedia

    # sanity check, doesn't make sense to compare distributions of different dimensions
    if cov_0.shape != cov_1.shape:
        print('cov_0.shape != cov_1.shape, ERROR!')
        exit()
    elif cov_0.shape[0] != cov_0.shape[1]:
        # if it is not a square covariance matrix (Doesn't make sense)
        print('The covariance matrices are not square!!! ERROR!')
        exit()

    k = cov_0.shape[0]
    inv_1 = np.linalg.inv(cov_1)
    diff_means = mean_1 - mean_0

    return .5 * (np.trace(np.dot(inv_1, cov_0)) + np.dot(diff_means.T, np.dot(inv_1, diff_means)) - k + np.log(np.linalg.det(cov_1) / np.linalg.det(cov_0)))


def multivariate_normal_diff_entropy(co_var_mat):
    # This one came from the Multivariate Normal page on Wikipedia
    print(co_var_mat.shape)
    exit()
    # this is the dimensionality of the matrix
    k = co_var_mat.shape[0]
    the_det = np.linalg.det(co_var_mat, 2)

    # same thing as .5 * np.log( ((2*np.pi*np.exp(1))**k) * np.linalg.det(co_var_mat) )
    return (k/2) + (k/2) * np.log(2*np.pi) + .5*np.log(the_det)


def MI_intra(sis_intra_A, sis_intra_B, variable):


    plus_plus = 0
    plus_minus_symmetric = 0
    minus_minus = 0
    count = 0
    plus_total = 0
    minus_total = 0

    for A, B in zip([sis_intra_A.iloc[num] for num in range(len(sis_intra_A))], [sis_intra_B.iloc[num] for num in range(len(sis_intra_B))]):
        plus_A = A[variable] >= A['trap_avg_'+variable]
        minus_A = A[variable] < A['trap_avg_'+variable]
        plus_B = B[variable] >= B['trap_avg_'+variable]
        minus_B = B[variable] < B['trap_avg_'+variable]
        if plus_A and plus_B:
            plus_plus += 1
            plus_total += 2
        elif (plus_A and minus_B) or (plus_B and minus_A):
            plus_minus_symmetric += 1
            plus_total += 1
            minus_total += 1
        elif minus_A and minus_B:
            minus_minus += 1
            minus_total += 2
        else:
            print('There is an error in the code')
            exit()
        count += 1

    p_p_joint = plus_plus / count
    m_m_joint = minus_minus / count
    p_m_joint = (plus_minus_symmetric/2) / count
    p_marg = plus_total / (count * 2)
    m_marg = minus_total / (count * 2)

    cond_entropy = - (p_p_joint*np.log(p_p_joint/p_marg)+p_m_joint*np.log(p_m_joint/p_marg)+p_m_joint*np.log(p_m_joint/m_marg)+m_m_joint*np.log(m_m_joint/m_marg))
    entropy = - (p_marg*np.log(p_marg)+m_marg*np.log(m_marg))
    mutual_info = p_p_joint*np.log(p_p_joint/(p_marg*p_marg))+p_m_joint*np.log(p_m_joint/(p_marg*m_marg))+p_m_joint*np.log(p_m_joint/(m_marg*p_marg))+m_m_joint*np.log(m_m_joint/(m_marg*m_marg))

    # print(plus_total+minus_total)
    # print(count)
    # print('++ / count =', plus_plus / count)
    # print('-- / count =', minus_minus / count)
    # print('+- / count =', plus_minus_symmetric / count)
    # print('minus_total =', minus_total / (count * 2))
    # print('plus_total =', plus_total / (count * 2))
    # print('cond of +|+ =', (plus_plus / count) / (plus_total / (count * 2)))
    # print('cond of +|- =', ((plus_minus_symmetric/2) / count) / (minus_total / (count * 2)))
    # print('cond of -|+ =', ((plus_minus_symmetric/2) / count) / (plus_total / (count * 2)))
    # print('cond of -|- =', (minus_minus / count) / (minus_total / (count * 2)))
    # print('cond_entropy', cond_entropy)
    # print('entropy', entropy)
    # print('mutual_info', mutual_info)
    # print('____________________')
    return entropy, cond_entropy, mutual_info


def entropies_for_one_trap(df_A, df_B, variable, trap_mean):

    # this is to get the joint and marginal probabilities easier by masking the dataframe
    together = pd.concat([df_A.rename(columns=dict(zip(df_A.keys(), [a + '_A' for a in df_A.keys()]))),
                          df_B.rename(columns=dict(zip(df_B.keys(), [a + '_B' for a in df_A.keys()])))], axis=1).reset_index(drop=True)

    # define the probabilities
    plus_plus_joint = len(together[(together[variable+'_A'] >= trap_mean[variable]) & (together[variable+'_B'] >= trap_mean[variable])]) / len(together)
    minus_minus_joint = len(together[(together[variable + '_A'] < trap_mean[variable]) & (together[variable + '_B'] < trap_mean[variable])]) / len(together)
    plus_minus_joint = len(together[(together[variable + '_A'] >= trap_mean[variable]) & (together[variable + '_B'] < trap_mean[variable])]) / len(together)
    minus_plus_joint = len(together[(together[variable + '_A'] < trap_mean[variable]) & (together[variable + '_B'] >= trap_mean[variable])]) / len(together)
    
    minus_marginal = (len(together[(together[variable + '_A'] < trap_mean[variable])])+len(together[(together[variable + '_B'] < trap_mean[variable])])) / (len(together)*2)
    plus_marginal = (len(together[(together[variable + '_A'] >= trap_mean[variable])])+len(together[(together[variable + '_B'] >= trap_mean[variable])])) / (len(together)*2)

    # to make the code for the following calculations easier to understand
    joint_prob_list = [plus_plus_joint, minus_minus_joint, plus_minus_joint, minus_plus_joint]
    A_marginal_probs = [plus_marginal, minus_marginal, minus_marginal, plus_marginal]
    B_marginal_probs = [plus_marginal, minus_marginal, plus_marginal, minus_marginal]

    # the entropy of the trap
    trap_entropy = np.array([- marg * np.log(marg) for marg in [plus_marginal, minus_marginal]]).sum()

    # conditioning the A trace based on the B trace
    A_conditioned_on_B_entropy = np.array([- joint * np.log(joint / marg) for joint, marg in zip(joint_prob_list, A_marginal_probs) if joint != 0 and marg != 0]).sum()
    B_conditioned_on_A_entropy = np.array([- joint * np.log(joint / marg) for joint, marg in zip(joint_prob_list, B_marginal_probs) if joint != 0 and marg != 0]).sum()

    # the mutual information between A and B trace for this variable
    mutual_info = np.array([joint * np.log(joint / (marg_A * marg_B)) for joint, marg_A, marg_B in zip(joint_prob_list, A_marginal_probs, B_marginal_probs) if joint != 0 and marg_A != 0 and marg_B != 0]).sum()
    
    return trap_entropy, A_conditioned_on_B_entropy, B_conditioned_on_A_entropy, mutual_info


def entropies_for_one_trap1(df_A, df_B, variable, trap_mean):
    # this is to get the joint and marginal probabilities easier by masking the dataframe
    together = pd.concat([df_A.rename(columns=dict(zip(df_A.keys(), [a + '_A' for a in df_A.keys()]))),
                          df_B.rename(columns=dict(zip(df_B.keys(), [a + '_B' for a in df_A.keys()])))], axis=1).reset_index(drop=True)

    # define the probabilities
    plus_plus_joint = len(together[(together[variable + '_A'] >= trap_mean[variable]) & (together[variable + '_B'] >= trap_mean[variable])]) / len(
        together)
    minus_minus_joint = len(together[(together[variable + '_A'] < trap_mean[variable]) & (together[variable + '_B'] < trap_mean[variable])]) / len(
        together)
    plus_minus_joint = len(together[(together[variable + '_A'] >= trap_mean[variable]) & (together[variable + '_B'] < trap_mean[variable])]) / len(
        together)
    minus_plus_joint = len(together[(together[variable + '_A'] < trap_mean[variable]) & (together[variable + '_B'] >= trap_mean[variable])]) / len(
        together)

    # If we consider each instance of the trace to come from its trace distribution and not the trap distribution
    A_minus_marginal = len(together[(together[variable + '_A'] < trap_mean[variable])]) / len(together)
    A_plus_marginal = len(together[(together[variable + '_A'] >= trap_mean[variable])]) / len(together)
    B_minus_marginal = len(together[(together[variable + '_B'] < trap_mean[variable])]) / len(together)
    B_plus_marginal = len(together[(together[variable + '_B'] >= trap_mean[variable])]) / len(together)

    # If we consider each instance of the trace to come from its trap distribution and not the trace distribution
    minus_marginal = (len(together[(together[variable + '_A'] < trap_mean[variable])]) + len(
        together[(together[variable + '_B'] < trap_mean[variable])])) / (len(together) * 2)
    plus_marginal = (len(together[(together[variable + '_A'] >= trap_mean[variable])]) + len(
        together[(together[variable + '_B'] >= trap_mean[variable])])) / (len(together) * 2)

    # to make the code for the following calculations easier to understand
    joint_prob_list = [plus_plus_joint, minus_minus_joint, plus_minus_joint, minus_plus_joint]
    A_marginal_probs = [plus_marginal, minus_marginal, minus_marginal, plus_marginal]
    B_marginal_probs = [plus_marginal, minus_marginal, plus_marginal, minus_marginal]
    A_trace_marginal_probs = [A_plus_marginal, A_minus_marginal, A_minus_marginal, A_plus_marginal]
    B_trace_marginal_probs = [B_plus_marginal, B_minus_marginal, B_plus_marginal, B_minus_marginal]

    # the entropy of the trap
    trap_entropy = np.array([- marg * np.log(marg) for marg in [plus_marginal, minus_marginal]]).sum()

    # conditioning the A trace based on the B trace
    A_conditioned_on_B_entropy = np.array(
        [- joint * np.log(joint / marg) for joint, marg in zip(joint_prob_list, A_marginal_probs) if joint != 0 and marg != 0]).sum()
    B_conditioned_on_A_entropy = np.array(
        [- joint * np.log(joint / marg) for joint, marg in zip(joint_prob_list, B_marginal_probs) if joint != 0 and marg != 0]).sum()

    # the mutual information between A and B trace for this variable thinking that marginal A and B came from the trap distribution
    mutual_info = np.array(
        [joint * np.log(joint / (marg_A * marg_B)) for joint, marg_A, marg_B in zip(joint_prob_list, A_marginal_probs, B_marginal_probs) if
         joint != 0 and marg_A != 0 and marg_B != 0]).sum()

    # the mutual information between A and B trace for this variable thinking that marginal A and B came from each trace distribution
    mutual_info_trace = np.array(
        [joint * np.log(joint / (marg_A * marg_B)) for joint, marg_A, marg_B in zip(joint_prob_list, A_trace_marginal_probs, B_trace_marginal_probs) if
         joint != 0 and marg_A != 0 and marg_B != 0]).sum()

    return trap_entropy, A_conditioned_on_B_entropy, B_conditioned_on_A_entropy, mutual_info_trace


def ents_for_one_trap(sis_A, sis_B, variable, trap_mean):
    plus_plus = 0
    plus_minus = 0
    minus_plus = 0
    minus_minus = 0
    count = 0
    plus_total = 0
    minus_total = 0

    together = pd.concat([sis_A.rename(columns=dict(zip(sis_A.keys(), [a+'_A' for a in sis_A.keys()]))),
                          sis_B.rename(columns=dict(zip(sis_B.keys(), [a+'_B' for a in sis_A.keys()])))], axis=1).reset_index(drop=True)
    exit()

    # print('______________')
    # print(sis_A)
    # print(sis_B)
    # print('______________')

    # plus_A = A[variable] >= trap_mean[variable]
    # minus_A = A[variable] < trap_mean[variable]
    # plus_B = B[variable] >= trap_mean[variable]
    # minus_B = B[variable] < trap_mean[variable]

    print(sis_A)
    print(sis_A[variable] >= trap_mean[variable])
    # print(sis_A[sis_A[variable] >= trap_mean[variable] & sis_B[variable] >= trap_mean[variable]])
    exit()





    for A, B in zip([sis_A.iloc[num] for num in range(len(sis_A))], [sis_B.iloc[num] for num in range(len(sis_B))]):
        # print('______________')
        # print(trap_mean)
        # print(A[variable])
        # print(B[variable])
        plus_A = A[variable] >= trap_mean[variable]
        minus_A = A[variable] < trap_mean[variable]
        plus_B = B[variable] >= trap_mean[variable]
        minus_B = B[variable] < trap_mean[variable]
        if plus_A and plus_B:
            plus_plus += 1
            plus_total += 2
        elif (plus_A and minus_B):
            plus_minus += 1
            plus_total += 1
            minus_total += 1
        elif (plus_B and minus_A):
            minus_plus += 1
            plus_total += 1
            minus_total += 1
        elif minus_A and minus_B:
            minus_minus += 1
            minus_total += 2
        else:
            print('There is an error in the code')
            exit()
        count += 1

    p_p_joint = plus_plus / count
    m_m_joint = minus_minus / count
    p_m_joint = plus_minus / count
    m_p_joint = minus_plus / count
    p_marg = plus_total / (count * 2)
    m_marg = minus_total / (count * 2)
    
    if p_p_joint == 0.0:
        print('p_p_joint is 0!')
        p_p_part = 0
        p_p_MI = 0
    else:
        p_p_part = p_p_joint * np.log(p_p_joint / p_marg)
        p_p_MI = p_p_joint * np.log(p_p_joint / (p_marg * p_marg))

    if p_m_joint == 0.0:
        print('p_m_joint is 0!')
        p_m_part = 0
        p_m_MI = 0
    else:
        p_m_part = p_m_joint * np.log(p_m_joint / p_marg)
        p_m_MI = p_m_joint * np.log(p_m_joint / (p_marg * m_marg))

    if m_p_joint == 0.0:
        print('m_p_joint is 0!')
        m_p_part = 0
        m_p_MI = 0
    else:
        m_p_part = m_p_joint * np.log(m_p_joint / m_marg)
        m_p_MI = m_p_joint * np.log(m_p_joint / (m_marg * p_marg))
        
    if m_m_joint == 0.0:
        print('m_m_joint is 0!')
        m_m_part = 0
        m_m_MI = 0
    else:
        m_m_part = m_m_joint * np.log(m_m_joint / m_marg)
        m_m_MI = m_m_joint * np.log(m_m_joint / (m_marg * m_marg))

    cond_entropy = - (p_p_part + p_m_part + m_p_part + m_m_part)
    entropy = - (p_marg * np.log(p_marg) + m_marg * np.log(m_marg))
    mutual_info = p_p_MI + p_m_MI + m_p_MI + m_m_MI

    if math.isnan(cond_entropy):
        print('THERE IS A NAN!!!!')
        print(p_p_joint)
        print(m_m_joint)
        print(p_m_joint)
        print(m_p_joint)
        print(p_marg)
        print(m_marg)
        print(cond_entropy)

    # print(plus_total+minus_total)
    # print(count)
    # print('++ / count =', plus_plus / count)
    # print('-- / count =', minus_minus / count)
    # print('+- / count =', plus_minus_symmetric / count)
    # print('minus_total =', minus_total / (count * 2))
    # print('plus_total =', plus_total / (count * 2))
    # print('cond of +|+ =', (plus_plus / count) / (plus_total / (count * 2)))
    # print('cond of +|- =', ((plus_minus_symmetric/2) / count) / (minus_total / (count * 2)))
    # print('cond of -|+ =', ((plus_minus_symmetric/2) / count) / (plus_total / (count * 2)))
    # print('cond of -|- =', (minus_minus / count) / (minus_total / (count * 2)))
    # print('cond_entropy', cond_entropy)
    # print('entropy', entropy)
    # print('mutual_info', mutual_info)

    # if p_marg > .6 or m_marg > .6:
    #     print(round(p_marg, 2), round(m_marg, 2))
    #     sns.kdeplot(pd.concat([sis_A, sis_B], axis=0)[variable])
    #     plt.axvline(trap_mean[variable], color='black')
    #     plt.axvline(np.mean(pd.concat([sis_A, sis_B], axis=0)[variable]), color='green')
    #     plt.show()
    #     plt.close()
    #     print(trap_mean)
    # print('____________________')
    return entropy, cond_entropy, mutual_info


def for_each_dset(dset_A, dset_B, variable, indeces, trap_mean=None):
    trap_entropy = []
    A_conditioned_on_B_entropy = []
    B_conditioned_on_A_entropy = []
    mutual_info = []
    for ind in range(88):

        df_A = dset_A[str(indeces[ind])].iloc[:min(len(dset_A[str(indeces[ind])][variable]), len(dset_B[str(indeces[ind])][variable]))]
        df_B = dset_B[str(indeces[ind])].iloc[:min(len(dset_A[str(indeces[ind])][variable]), len(dset_B[str(indeces[ind])][variable]))]
        if not isinstance(trap_mean, pd.Series):
            print('Trap Means')
            trap_mean = pd.concat([df_A, df_B], axis=0).mean()
        dset = entropies_for_one_trap(df_A, df_B, variable, trap_mean)
        trap_entropy.append(dset[0])
        A_conditioned_on_B_entropy.append(dset[1])
        B_conditioned_on_A_entropy.append(dset[2])
        mutual_info.append(dset[3])

    trap_entropy = np.array(trap_entropy)
    A_conditioned_on_B_entropy = np.array(A_conditioned_on_B_entropy)
    B_conditioned_on_A_entropy = np.array(B_conditioned_on_A_entropy)
    mutual_info = np.array(mutual_info)
        
    return trap_entropy, A_conditioned_on_B_entropy, B_conditioned_on_A_entropy, mutual_info


def for_each_dset2(dset_A, dset_B, variable, indeces, trap_mean=None):
    trap_entropy = []
    A_conditioned_on_B_entropy = []
    B_conditioned_on_A_entropy = []
    mutual_info = []
    for ind in range(88):

        df_A = dset_A[str(indeces[ind])].iloc[:min(len(dset_A[str(indeces[ind])][variable]), len(dset_B[str(indeces[ind])][variable]))]
        df_B = dset_B[str(indeces[ind])].iloc[:min(len(dset_A[str(indeces[ind])][variable]), len(dset_B[str(indeces[ind])][variable]))]
        if not isinstance(trap_mean, pd.Series):
            print('Trap Means')
            trap_mean = pd.concat([df_A, df_B], axis=0).mean()
        dset = entropies_for_one_trap1(df_A, df_B, variable, trap_mean)
        trap_entropy.append(dset[0])
        A_conditioned_on_B_entropy.append(dset[1])
        B_conditioned_on_A_entropy.append(dset[2])
        mutual_info.append(dset[3])

    trap_entropy = np.array(trap_entropy)
    A_conditioned_on_B_entropy = np.array(A_conditioned_on_B_entropy)
    B_conditioned_on_A_entropy = np.array(B_conditioned_on_A_entropy)
    mutual_info = np.array(mutual_info)

    return trap_entropy, A_conditioned_on_B_entropy, B_conditioned_on_A_entropy, mutual_info


def for_each_dset1(dset_A, dset_B, variable, indeces, trap_mean=None):
    for ind in range(88):

        df_A = dset_A[str(indeces[ind])].iloc[:min(len(dset_A[str(indeces[ind])][variable]), len(dset_B[str(indeces[ind])][variable]))]
        df_B = dset_B[str(indeces[ind])].iloc[:min(len(dset_A[str(indeces[ind])][variable]), len(dset_B[str(indeces[ind])][variable]))]
        if not isinstance(trap_mean, pd.Series):
            print('Trap Means')
            trap_mean = pd.concat([df_A, df_B], axis=0).mean()
        fitting_gaussian_to_trap_measurements(df_A, df_B, variable, trap_mean)


def fitting_gaussian_to_trap_measurements(df_A, df_B, variable, trap_mean):
    # this is to get the joint and marginal probabilities easier by masking the dataframe
    together = pd.concat([df_A, df_B], axis=0).reset_index(drop=True)

    mean, std = stats.norm.fit(together[variable])

    sns.kdeplot(np.random.normal(together[variable].mean(), together[variable].std(), 100), label='{} {}'.format(round(together[variable].mean(), 3), round(together[variable].std(), 3)))
    sns.kdeplot(together[variable], label='data hist')
    sns.kdeplot(np.random.normal(mean, std, 100), label='stats {}, {}'.format(round(mean, 3), round(std, 3)))
    plt.xlabel(variable)
    plt.legend()
    plt.show()
    plt.close()


def entropies_with_fitted_gaussian_per_trap(df_A, df_B, variable, trap_mean):
    # this is to get the joint and marginal probabilities easier by masking the dataframe
    together = pd.concat([df_A, df_B], axis=0).reset_index(drop=True)
    






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

    pickle_in = open("NewSisterCellClass_Env_Sister.pickle", "rb")
    Env_Sister = pickle.load(pickle_in)
    pickle_in.close()

    pickle_in = open("NewSisterCellClass_Env_Nonsister.pickle", "rb")
    Env_Nonsister = pickle.load(pickle_in)
    pickle_in.close()


    trap_mean = Population.mother_dfs[0].mean()

    # renaming
    sis_A, sis_B = Sister.A_dict.copy(), Sister.B_dict.copy()
    nonsis_A, nonsis_B = Nonsister.A_dict.copy(), Nonsister.B_dict.copy()
    control_A, control_B = Control.A_dict.copy(), Control.B_dict.copy()

    # renaming
    env_sis_intra_A, env_sis_intra_B = Env_Sister.A_intra_gen_bacteria.copy(), Env_Sister.B_intra_gen_bacteria.copy()
    sis_intra_A, sis_intra_B = Sister.A_intra_gen_bacteria.copy(), Sister.B_intra_gen_bacteria.copy()
    nonsis_intra_A, nonsis_intra_B = Nonsister.A_intra_gen_bacteria.copy(), Nonsister.B_intra_gen_bacteria.copy()
    env_nonsis_intra_A, env_nonsis_intra_B = Env_Nonsister.A_intra_gen_bacteria.copy(), Env_Nonsister.B_intra_gen_bacteria.copy()
    con_intra_A, con_intra_B = Control.A_intra_gen_bacteria.copy(), Control.B_intra_gen_bacteria.copy()

    np.random.seed(33)

    # this is so that we have as much sisters as nonsisters and control
    indeces_sis = np.random.choice(np.arange(130), size=88, replace=False)
    indeces = np.arange(88)

    for variable in Population._variable_names:

        # print('IN {}:'.format(variable))
        # print('IN SIS:')
        # for_each_dset1(sis_A, sis_B, variable,indeces_sis, trap_mean)
        # print('IN NONSIS:')
        # for_each_dset1(nonsis_A,nonsis_B,variable,indeces,trap_mean)
        # print('IN CONTROL:')
        # for_each_dset1(control_A, control_B,variable, indeces,trap_mean)


        print('IN {}:'.format(variable))
        print('IN SIS:')
        sis_trap_entropy, sis_A_conditioned_on_B_entropy, sis_B_conditioned_on_A_entropy, sis_mutual_info = for_each_dset(sis_A, sis_B, variable, indeces_sis, trap_mean)
        print('IN NONSIS:')
        nonsis_trap_entropy, nonsis_A_conditioned_on_B_entropy, nonsis_B_conditioned_on_A_entropy, nonsis_mutual_info = for_each_dset(nonsis_A, nonsis_B, variable, indeces, trap_mean)
        print('IN CONTROL:')
        con_trap_entropy, con_A_conditioned_on_B_entropy, con_B_conditioned_on_A_entropy, con_mutual_info = for_each_dset(control_A, control_B, variable, indeces, trap_mean)

        print('IN {}:'.format(variable))
        print('IN SIS:')
        _, _, _, sis_mutual_info1 = for_each_dset2(sis_A, sis_B, variable,
                                                                                                                          indeces_sis, trap_mean)
        print('IN NONSIS:')
        _, _, _, nonsis_mutual_info1 = for_each_dset2(nonsis_A,
                                                                                                                                      nonsis_B,
                                                                                                                                      variable,
                                                                                                                                      indeces, trap_mean)
        print('IN CONTROL:')
        _, _, _, con_mutual_info1 = for_each_dset2(control_A, control_B,
                                                                                                                          variable, indeces, trap_mean)

        # sis_MI = np.array([MI_intra(sis_intra_A[rel].iloc[indeces], sis_intra_B[rel].iloc[indeces], variable) for rel in range(len(sis_intra_A))])
        # nonsis_MI = np.array([MI_intra(nonsis_intra_A[rel], nonsis_intra_B[rel], variable) for rel in range(len(sis_intra_A))])
        # con_MI = np.array([MI_intra(con_intra_A[rel].iloc[indeces], con_intra_B[rel].iloc[indeces], variable) for rel in range(len(sis_intra_A))])
        # env_sis_MI = [MI_intra(env_sis_intra_A[rel], env_sis_intra_B[rel], variable) for rel in range(len(sis_intra_A))]
        # env_nonsis_MI = [MI_intra(env_nonsis_intra_A[rel], env_nonsis_intra_B[rel], variable) for rel in range(len(sis_intra_A))]
        #
        # plt.plot(np.arange(len(sis_intra_A)), sis_MI[:, 0], label='sis entropy', color='blue', linestyle='-', marker='.')
        # plt.plot(np.arange(len(sis_intra_A)), nonsis_MI[:, 0], label='nonsis entropy', color='orange', linestyle='-', marker='.')
        # plt.plot(np.arange(len(con_intra_A)), con_MI[:, 0], label='control entropy', color='green', linestyle='-', marker='.')
        # # plt.plot(np.arange(len(sis_intra_A)), sis_MI[:, 1], label='sis conditional entropy', color='blue', linestyle='--', marker='.')
        # # plt.plot(np.arange(len(sis_intra_A)), nonsis_MI[:, 1], label='nonsis conditional entropy', color='orange', linestyle='--', marker='.')
        # plt.plot(np.arange(len(sis_intra_A)), sis_MI[:, 2], label='sis MI', color='blue', linestyle=':', marker='.')
        # plt.plot(np.arange(len(sis_intra_A)), nonsis_MI[:, 2], label='nonsis MI', color='orange', linestyle=':', marker='.')
        # plt.plot(np.arange(len(con_intra_A)), con_MI[:, 2], label='con MI', color='green', linestyle=':', marker='.')
        # plt.legend()
        # plt.title(variable)
        # # sns.pointplot(x=np.arange(len(sis_intra_A)), y=env_sis_MI, dodge=True, label='env sis', color='red')
        # # sns.pointplot(x=np.arange(len(sis_intra_A)), y=env_nonsis_MI, dodge=True, label='env non sis', color='green')
        # plt.show()

        # normal entropies
        sns.kdeplot(sis_trap_entropy, color='blue', label='sis entropy')
        sns.kdeplot(nonsis_trap_entropy, color='orange', label='nonsis entropy')
        sns.kdeplot(con_trap_entropy, color='green', label='control entropy')
        plt.title(variable)
        plt.legend()
        plt.show()
        plt.close()

        # # the conditionals
        # sns.kdeplot(abs(sis_entropies - sis_cond_entropies), color='blue', label='sis cond. entropy')
        # sns.kdeplot(abs(nonsis_entropies - nonsis_cond_entropies), color='orange', label='nonsis cond. entropy')
        # sns.kdeplot(abs(control_entropies - control_cond_entropies), color='green', label='control cond. entropy')
        # # plt.axvline(np.mean(sis_entropies), color='blue', linestyle='--')
        # # plt.axvline(np.mean(nonsis_entropies), color='orange', linestyle='--')
        # # plt.axvline(np.mean(control_entropies), color='green', linestyle='--')
        # plt.title(variable)
        # plt.legend()
        # plt.show()
        # plt.close()

        # the mutual information
        sns.kdeplot(sis_mutual_info, color='blue', label='sis Mutual info')
        sns.kdeplot(nonsis_mutual_info, color='orange', label='nonsis Mutual info')
        sns.kdeplot(con_mutual_info, color='green', label='control Mutual info')
        plt.title(variable)
        plt.legend()
        plt.show()
        plt.close()


        sns.kdeplot(sis_mutual_info1, color='blue', label='sis Mutual info 1')
        sns.kdeplot(nonsis_mutual_info1, color='orange', label='nonsis Mutual info 1')
        sns.kdeplot(con_mutual_info1, color='green', label='control Mutual info 1')
        plt.title(variable)
        plt.legend()
        plt.show()
        plt.close()
    

    # NOW WE WILL DO THE ENTROPIES OF THE INTRA GENERATIONAL RELATIONSHIPS
    for variable in Population._variable_names:

        sis_MI = np.array([MI_intra(sis_intra_A[rel].iloc[indeces], sis_intra_B[rel].iloc[indeces], variable) for rel in range(len(sis_intra_A))])
        nonsis_MI = np.array([MI_intra(nonsis_intra_A[rel], nonsis_intra_B[rel], variable) for rel in range(len(sis_intra_A))])
        con_MI = np.array([MI_intra(con_intra_A[rel].iloc[indeces], con_intra_B[rel].iloc[indeces], variable) for rel in range(len(sis_intra_A))])
        env_sis_MI = [MI_intra(env_sis_intra_A[rel], env_sis_intra_B[rel], variable) for rel in range(len(sis_intra_A))]
        env_nonsis_MI = [MI_intra(env_nonsis_intra_A[rel], env_nonsis_intra_B[rel], variable) for rel in range(len(sis_intra_A))]

        plt.plot(np.arange(len(sis_intra_A)), sis_MI[:, 0], label='sis entropy', color='blue', linestyle='-', marker='.')
        plt.plot(np.arange(len(sis_intra_A)), nonsis_MI[:, 0], label='nonsis entropy', color='orange', linestyle='-', marker='.')
        plt.plot(np.arange(len(con_intra_A)), con_MI[:, 0], label='control entropy', color='green', linestyle='-', marker='.')
        # plt.plot(np.arange(len(sis_intra_A)), sis_MI[:, 1], label='sis conditional entropy', color='blue', linestyle='--', marker='.')
        # plt.plot(np.arange(len(sis_intra_A)), nonsis_MI[:, 1], label='nonsis conditional entropy', color='orange', linestyle='--', marker='.')
        plt.plot(np.arange(len(sis_intra_A)), sis_MI[:, 2], label='sis MI', color='blue', linestyle=':', marker='.')
        plt.plot(np.arange(len(sis_intra_A)), nonsis_MI[:, 2], label='nonsis MI', color='orange', linestyle=':', marker='.')
        plt.plot(np.arange(len(con_intra_A)), con_MI[:, 2], label='con MI', color='green', linestyle=':', marker='.')
        plt.legend()
        plt.title(variable)
        # sns.pointplot(x=np.arange(len(sis_intra_A)), y=env_sis_MI, dodge=True, label='env sis', color='red')
        # sns.pointplot(x=np.arange(len(sis_intra_A)), y=env_nonsis_MI, dodge=True, label='env non sis', color='green')
        plt.show()
    
    
    
    
    exit()

    MI_intra(sis_intra_A, sis_intra_B)
    MI_intra(nonsis_intra_A, nonsis_intra_B)
    MI_intra(env_sis_intra_A, env_sis_intra_B)
    MI_intra(env_nonsis_intra_A, env_nonsis_intra_B)

    exit()












    # renaming
    mom = Population.mother_dfs[0].copy()
    daug = Population.daughter_dfs[0].copy()

    # because it is a false cycle and has a negative growth rate which is impossible
    mom = mom.drop(index=1479)
    daug = daug.drop(index=1478)

    all_vars = ['generationtime', 'length_birth', 'length_final', 'growth_rate', 'fold_growth', 'division_ratio']

    # generationtime and growth rate are log normal
    mom_global = subtract_global_averages(df=mom, columns_names=Population._variable_names, log_vars=['generationtime', 'growth_rate', 'length_birth'])
    mom_trap = subtract_trap_averages(df=mom, columns_names=Population._variable_names, log_vars=['generationtime', 'growth_rate', 'length_birth'])
    mom_traj = subtract_traj_averages(df=mom, columns_names=Population._variable_names, log_vars=['generationtime', 'growth_rate', 'length_birth'])

    daug_global = subtract_global_averages(df=daug, columns_names=Population._variable_names, log_vars=['generationtime', 'growth_rate', 'length_birth'])
    daug_trap = subtract_trap_averages(df=daug, columns_names=Population._variable_names, log_vars=['generationtime', 'growth_rate', 'length_birth'])
    daug_traj = subtract_traj_averages(df=daug, columns_names=Population._variable_names, log_vars=['generationtime', 'growth_rate', 'length_birth'])











if __name__ == '__main__':
    main()
