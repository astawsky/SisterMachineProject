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
import scipy.spatial as ss
from scipy.special import digamma, gamma
from math import log, pi
import numpy.random as nr
from sklearn.feature_selection import mutual_info_regression



#####CONTINUOUS ESTIMATORS

def entropy(x,k=3,base=2):
  """ The classic K-L k-nearest neighbor continuous entropy estimator
      x should be a list of vectors, e.g. x = [[1.3],[3.7],[5.1],[2.4]]
      if x is a one-dimensional scalar and we have four samples
  """
  assert k <= len(x)-1, "Set k smaller than num. samples - 1"
  d = 1 #len(x[0])
  N = len(x)
  intens = 1e-10 #small noise to break degeneracy, see doc.
  x = [list(p + intens*nr.rand(d)) for p in x]
  tree = ss.cKDTree(x)
  nn = [tree.query(point,k+1,p=float('inf'))[0][k] for point in x]
  const = digamma(N)-digamma(k) + d*log(2)
  return (const + d*np.mean(np.log(nn)))/log(base)

def mi(x,y,k=3,base=2):
  """ Mutual information of x and y
      x,y should be a list of vectors, e.g. x = [[1.3],[3.7],[5.1],[2.4]]
      if x is a one-dimensional scalar and we have four samples
  """
  assert len(x)==len(y), "Lists should have same length"
  assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
  intens = 1e-10 #small noise to break degeneracy, see doc.
  x = [list(p + intens*nr.rand(len(x[0]))) for p in x]
  y = [list(p + intens*nr.rand(len(y[0]))) for p in y]
  points = zip2(x,y)
  #Find nearest neighbors in joint space, p=inf means max-norm
  tree = ss.cKDTree(points)
  dvec = [tree.query(point,k+1,p=float('inf'))[0][k] for point in points]
  a,b,c,d = avgdigamma(x,dvec), avgdigamma(y,dvec), digamma(k), digamma(len(x))
  return (-a-b+c+d)/log(base)

def cmi(x,y,z,k=3,base=2):
  """ Mutual information of x and y, conditioned on z
      x,y,z should be a list of vectors, e.g. x = [[1.3],[3.7],[5.1],[2.4]]
      if x is a one-dimensional scalar and we have four samples
  """
  assert len(x)==len(y), "Lists should have same length"
  assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
  intens = 1e-10 #small noise to break degeneracy, see doc.
  x = [list(p + intens*nr.rand(len(x[0]))) for p in x]
  y = [list(p + intens*nr.rand(len(y[0]))) for p in y]
  z = [list(p + intens*nr.rand(len(z[0]))) for p in z]
  points = zip2(x,y,z)
  #Find nearest neighbors in joint space, p=inf means max-norm
  tree = ss.cKDTree(points)
  dvec = [tree.query(point,k+1,p=float('inf'))[0][k] for point in points]
  a,b,c,d = avgdigamma(zip2(x,z),dvec),avgdigamma(zip2(y,z),dvec),avgdigamma(z,dvec), digamma(k)
  return (-a-b+c+d)/log(base)

def kldiv(x,xp,k=3,base=2):
  """ KL Divergence between p and q for x~p(x),xp~q(x)
      x,xp should be a list of vectors, e.g. x = [[1.3],[3.7],[5.1],[2.4]]
      if x is a one-dimensional scalar and we have four samples
  """
  assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
  assert k <= len(xp) - 1, "Set k smaller than num. samples - 1"
  assert len(x[0]) == len(xp[0]), "Two distributions must have same dim."
  d = len(x[0])
  n = len(x)
  m = len(xp)
  const = log(m) - log(n-1)
  tree = ss.cKDTree(x)
  treep = ss.cKDTree(xp)
  nn = [tree.query(point,k+1,p=float('inf'))[0][k] for point in x]
  nnp = [treep.query(point,k,p=float('inf'))[0][k-1] for point in x]
  return (const + d*np.mean(map(log,nnp))-d*np.mean(map(log,nn)))/log(base)

#####DISCRETE ESTIMATORS
def entropyd(sx,base=2):
  """ Discrete entropy estimator
      Given a list of samples which can be any hashable object
  """
  return entropyfromprobs(hist(sx),base=base)

def midd(x,y):
  """ Discrete mutual information estimator
      Given a list of samples which can be any hashable object
  """
  return -entropyd(list(zip(x,y)))+entropyd(x)+entropyd(y)

def cmidd(x,y,z):
  """ Discrete mutual information estimator
      Given a list of samples which can be any hashable object
  """
  return entropyd(zip(y,z))+entropyd(zip(x,z))-entropyd(zip(x,y,z))-entropyd(z)

def hist(sx):
  #Histogram from list of samples
  d = dict()
  for s in sx:
    d[s] = d.get(s,0) + 1
  # print(sx, type(sx), d, type(d))
  # exit()
  return map(lambda z:float(z)/len(sx),d.values())

def entropyfromprobs(probs,base=2):
#Turn a normalized list of probabilities of discrete outcomes into entropy (base 2)
  return -sum(map(elog,probs))/log(base)

def elog(x):
#for entropy, 0 log 0 = 0. but we get an error for putting log 0
  if x <= 0. or x>=1.:
    return 0
  else:
    return x*log(x)

#####MIXED ESTIMATORS
def micd(x,y,k=3,base=2,warning=True):
  """ If x is continuous and y is discrete, compute mutual information
  """
  overallentropy = entropy(x,k,base)

  n = len(y)
  word_dict = dict()
  for sample in y:
    word_dict[sample] = word_dict.get(sample,0) + 1./n
  yvals = list(set(word_dict.keys()))

  mi = overallentropy
  for yval in yvals:
    xgiveny = [x[i] for i in range(n) if y[i]==yval]
    if k <= len(xgiveny) - 1:
      mi -= word_dict[yval]*entropy(xgiveny,k,base)
    else:
      if warning:
        print ("Warning, after conditioning, on y=",yval," insufficient data. Assuming maximal entropy in this case.")
      mi -= word_dict[yval]*overallentropy
  return mi #units already applied

#####UTILITY FUNCTIONS
def vectorize(scalarlist):
  """ Turn a list of scalars into a list of one-d vectors
  """
  return [(x,) for x in scalarlist]

def shuffle_test(measure,x,y,z=False,ns=200,ci=0.95,**kwargs):
  """ Shuffle test
      Repeatedly shuffle the x-values and then estimate measure(x,y,[z]).
      Returns the mean and conf. interval ('ci=0.95' default) over 'ns' runs.
      'measure' could me mi,cmi, e.g. Keyword arguments can be passed.
      Mutual information and CMI should have a mean near zero.
  """
  xp = x[:] #A copy that we can shuffle
  outputs = []
  for i in range(ns):
    random.shuffle(xp)
    if z:
      outputs.append(measure(xp,y,z,**kwargs))
    else:
      outputs.append(measure(xp,y,**kwargs))
  outputs.sort()
  return np.mean(outputs),(outputs[int((1.-ci)/2*ns)],outputs[int((1.+ci)/2*ns)])

#####INTERNAL FUNCTIONS

def avgdigamma(points,dvec):
  #This part finds number of neighbors in some radius in the marginal space
  #returns expectation value of <psi(nx)>
  N = len(points)
  tree = ss.cKDTree(points)
  avg = 0.
  for i in range(N):
    dist = dvec[i]
    #subtlety, we don't include the boundary point,
    #but we are implicitly adding 1 to kraskov def bc center point is included
    num_points = len(tree.query_ball_point(points[i],dist-1e-15,p=float('inf')))
    avg += digamma(num_points)/N
  return avg

def zip2(*args):
  #zip2(x,y) takes the lists of vectors and makes it a list of vectors in a joint space
  #E.g. zip2([[1],[2],[3]],[[4],[5],[6]]) = [[1,4],[2,5],[3,6]]
  return [sum(sublist,[]) for sublist in zip(*args)]


def print_full_dataframe(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df)


def add_the_added_length(dictionary):
    new_dictionary = dictionary.copy()
    for key, val in dictionary.items():
        new_dictionary[key]['added_length'] = dictionary[key]['length_final'] - dictionary[key]['length_birth']
        if pd.Series(dictionary[key]['length_final'] < dictionary[key]['length_birth']).isna().any():
            print(dictionary[key]['length_final'].iloc[pd.Series(dictionary[key]['length_final'] < dictionary[key]['length_birth']).isna()],
                  dictionary[key]['length_birth'].iloc[pd.Series(dictionary[key]['length_final'] < dictionary[key]['length_birth']).isna()])
        if len(np.where(new_dictionary[key]['added_length'] < 0)) > 1:
            print("found a declining bacteria")
            print(np.where(new_dictionary[key]['added_length'] < 0))
            print(new_dictionary[key]['added_length'].iloc[np.where(new_dictionary[key]['added_length'] < 0)])
        new_dictionary[key] = new_dictionary[key].drop(index=np.where(new_dictionary[key]['added_length'] < 0)[0])

    return new_dictionary


def for_dataframe(dataframeA, dataframeB, variable_names):
    datasetmis = pd.DataFrame(columns=variable_names, index=variable_names)
    min_len = min(len(dataframeA), len(dataframeB))
    for var1 in variable_names:
        for var2 in variable_names:
            if var1 == 'generationtime' and var2 == 'generationtime':
                datasetmis[var1].loc[var2] = midd(x=dataframeA[var1].iloc[:min_len],
                                                  y=dataframeB[var2].iloc[:min_len])
            elif var1 != 'generationtime' and var2 == 'generationtime':
                datasetmis[var1].loc[var2] = micd(x=np.array(dataframeA[var1].iloc[:min_len]).reshape(-1, 1),
                                                  y=dataframeB[var2].iloc[:min_len], k=1, base=np.e, warning=True)
            elif var1 == 'generationtime' and var2 != 'generationtime':
                datasetmis[var1].loc[var2] = micd(x=list(dataframeB[var2].iloc[:min_len]),
                                                  y=dataframeA[var1].iloc[:min_len], k=1, base=np.e, warning=True)
            else:
                datasetmis[var1].loc[var2] = mi(x=np.array(dataframeA[var1].iloc[:min_len]).reshape(-1, 1),
                                                y=np.array(dataframeB[var2].iloc[:min_len]).reshape(-1, 1), k=1, base=np.e)

            if datasetmis[var1].loc[var2] < 0:
                print('a negative mutual info was found! {} and {}'.format(var1, var2))
                print(datasetmis[var1].loc[var2])

    return datasetmis


def for_dataframe1(dataframe, A_variable_symbol, B_variable_symbol, k, base):
    datasetmis = pd.DataFrame(columns=A_variable_symbol, index=B_variable_symbol, dtype=float)
    for var1 in A_variable_symbol:
        for var2 in B_variable_symbol:
            if var1 == 'generationtime' and var2 == 'generationtime':
                datasetmis[var1].loc[var2] = midd(x=dataframe[var1],
                                                  y=dataframe[var2])
            elif var1 != 'generationtime' and var2 == 'generationtime':
                datasetmis[var1].loc[var2] = micd(x=np.array(dataframe[var1]).reshape(-1, 1),
                                                  y=dataframe[var2], k=k, base=base, warning=True)
            elif var1 == 'generationtime' and var2 != 'generationtime':
                datasetmis[var1].loc[var2] = micd(x=list(dataframe[var2]),
                                                  y=dataframe[var1], k=k, base=base, warning=True)
            else:
                datasetmis[var1].loc[var2] = mi(x=np.array(dataframe[var1]).reshape(-1, 1),
                                                y=np.array(dataframe[var2]).reshape(-1, 1), k=k, base=base)

            if datasetmis[var1].loc[var2] < 0:
                print('a negative mutual info was found! {} and {}'.format(var1, var2))
                print(datasetmis[var1].loc[var2])

    return datasetmis


def for_dataset(datasetA, datasetB, variable_names):
    datasetmis_dict = dict()
    for keyA, keyB in zip(datasetA.keys(), datasetB.keys()):

        datasetmis = for_dataframe(datasetA[keyA], datasetB[keyB], variable_names)

        datasetmis_dict.update({str(keyA): datasetmis})

    return datasetmis_dict


def get_pooled_trap_df(A_df, B_df, variable_names, A_variable_symbol, B_variable_symbol):
    together_array = pd.DataFrame(columns=A_variable_symbol + B_variable_symbol, dtype=float)
    for val_A, val_B in zip(A_df.values(), B_df.values()):
        min_len = min(len(val_A), len(val_B))
        # append this variable's labels for all traps
        together_array = together_array.append(
            pd.concat([val_A.iloc[:min_len].rename(columns=dict(zip(variable_names, A_variable_symbol))),
                       val_B.iloc[:min_len].rename(columns=dict(zip(variable_names, B_variable_symbol)))],
                      axis=1), ignore_index=True)

    return together_array


def calculate_MI_and_save_heatmap_for_all_dsets_together(sis_MI, non_sis_MI, con_MI, type_mean, variable_names,
                                                         A_variable_symbol, B_variable_symbol):
    # mult_number = 100000
    #
    # sis_MI = sis_MI * mult_number
    # non_sis_MI = non_sis_MI * mult_number
    # con_MI = con_MI * mult_number

    vmin = np.min(np.array([np.min(sis_MI.min()), np.min(non_sis_MI.min()), np.min(con_MI.min())]))
    vmax = np.max(np.array([np.max(sis_MI.max()), np.max(non_sis_MI.max()), np.max(con_MI.max())]))

    fig, (ax_sis, ax_non_sis, ax_con) = plt.subplots(ncols=3, figsize=(12.7, 7.5))
    fig.subplots_adjust(wspace=0.01)

    sns.heatmap(data=sis_MI, annot=True, ax=ax_sis, cbar=False, vmin=vmin, vmax=vmax, yticklabels=True, fmt='.0f')
    ax_sis.set_title('Sister')
    sns.heatmap(data=non_sis_MI, annot=True, ax=ax_non_sis, cbar=False, vmin=vmin, vmax=vmax, yticklabels=False, fmt='.0f')
    ax_non_sis.set_title('Non-Sister')
    sns.heatmap(data=con_MI, annot=True, ax=ax_con, cbar=True, vmin=vmin, vmax=vmax, yticklabels=False, fmt='.0f')  # xticklabels=[]
    ax_con.set_title('Control')
    # fig.suptitle('Mutual Information x {} with {} bins on each side of the {} mean'.format(mult_number, bins_on_side, type_mean))
    # plt.tight_layout()
    # fig.colorbar(ax_con.collections[0], ax=ax_con, location="right", use_gridspec=False, pad=0.2)
    # plt.title('{} Mutual Information with {} bins on each side of the {} mean'.format(dataset, bins_on_side, type_mean))
    # plt.show()
    plt.savefig('NEW Mutual Information with {} mean'.format(type_mean), dpi=300)
    plt.close()


def my_ent(dataframe, k, eps_x, eps_y, variable_names_A, variable_names_B):
    # number of samples
    N = len(dataframe)
    # finding the n_x number of neighbors in
    for ind in dataframe.index:
        for var1 in variable_names_A:
            for var2 in variable_names_B:
                if var1 != variable_names_A[0] and var2 != variable_names_B[0]:
                    distance_of_k_th_neighbor = np.sort(np.unique(dataframe[var1] - dataframe[var2].iloc[ind]), axis=None)[k]

                else:
                    print(var1, var2)



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

    # renaming
    mom, daug = Population.mother_dfs[0].copy(), Population.daughter_dfs[0].copy()
    # mom = add_the_added_length(dictionary=mom)
    # daug = add_the_added_length(dictionary=daug)
    sis_A, sis_B = Sister.A_dict.copy(), Sister.B_dict.copy()
    sis_A = add_the_added_length(dictionary=sis_A)
    sis_B = add_the_added_length(dictionary=sis_B)
    non_sis_A, non_sis_B = Nonsister.A_dict.copy(), Nonsister.B_dict.copy()
    non_sis_A = add_the_added_length(dictionary=non_sis_A)
    non_sis_B = add_the_added_length(dictionary=non_sis_B)
    con_A, con_B = Control.A_dict.copy(), Control.B_dict.copy()
    con_A = add_the_added_length(dictionary=con_A)
    con_B = add_the_added_length(dictionary=con_B)
    con_ref_A, con_ref_B = Control.reference_A_dict.copy(), Control.reference_B_dict.copy()
    # env_sis_intra_A, env_sis_intra_B = Env_Sister.A_intra_gen_bacteria.copy(), Env_Sister.B_intra_gen_bacteria.copy()
    sis_intra_A, sis_intra_B = Sister.A_intra_gen_bacteria.copy(), Sister.B_intra_gen_bacteria.copy()
    sis_intra_A = [
        pd.concat([sis_intra_A[ind], pd.Series(sis_intra_A[ind]['length_final'] - sis_intra_A[ind]['length_birth'], name='added_length')], axis=1) for
        ind in range(len(sis_intra_A))]
    sis_intra_B = [
        pd.concat([sis_intra_B[ind], pd.Series(sis_intra_B[ind]['length_final'] - sis_intra_B[ind]['length_birth'], name='added_length')], axis=1) for
        ind in range(len(sis_intra_B))]
    non_sis_intra_A, non_sis_intra_B = Nonsister.A_intra_gen_bacteria.copy(), Nonsister.B_intra_gen_bacteria.copy()
    non_sis_intra_A = [
        pd.concat([non_sis_intra_A[ind], pd.Series(non_sis_intra_A[ind]['length_final'] - non_sis_intra_A[ind]['length_birth'], name='added_length')],
                  axis=1) for
        ind in range(len(non_sis_intra_A))]
    non_sis_intra_B = [
        pd.concat([non_sis_intra_B[ind], pd.Series(non_sis_intra_B[ind]['length_final'] - non_sis_intra_B[ind]['length_birth'], name='added_length')],
                  axis=1) for
        ind in range(len(non_sis_intra_B))]
    # env_nonsis_intra_A, env_nonsis_intra_B = Env_Nonsister.A_intra_gen_bacteria.copy(), Env_Nonsister.B_intra_gen_bacteria.copy()
    con_intra_A, con_intra_B = Control.A_intra_gen_bacteria.copy(), Control.B_intra_gen_bacteria.copy()
    con_intra_A = [
        pd.concat([con_intra_A[ind], pd.Series(con_intra_A[ind]['length_final'] - con_intra_A[ind]['length_birth'], name='added_length')], axis=1) for
        ind in range(len(con_intra_A))]
    con_intra_B = [
        pd.concat([con_intra_B[ind], pd.Series(con_intra_B[ind]['length_final'] - con_intra_B[ind]['length_birth'], name='added_length')], axis=1) for
        ind in range(len(con_intra_B))]

    # getting the all bacteria dataframe
    all_bacteria = pd.concat([mom, daug], axis=0).reset_index(drop=True).drop_duplicates(inplace=False, keep='first', subset=mom.columns)
    all_bacteria['added_length'] = pd.Series(all_bacteria['length_final'] - all_bacteria['length_birth'], name='added_length')
    all_bacteria = all_bacteria.drop(index=np.where(all_bacteria['added_length'] < 0)[0])
    # all_bacteria = add_the_added_length(dictionary=all_bacteria)

    log_vars = ['generationtime', 'length_birth', 'length_final', 'growth_rate', 'fold_growth', 'added_length']
    variable_names = ['generationtime', 'length_birth', 'length_final', 'growth_rate', 'fold_growth', 'division_ratio', 'added_length']

    # # the Histograms to check if variables are normal or lognormal
    # for var in variable_names:
    #     print(var)
    #     normal = np.random.normal(np.mean(all_bacteria[var]), np.std(all_bacteria[var]), 500)
    #     lognormal = np.random.lognormal(np.mean(np.log(all_bacteria[var])), np.std(np.log(all_bacteria[var])), 500)
    #     normal_da_p = stats.mstats.normaltest(all_bacteria[var])
    #     log_normal_da_p = stats.mstats.normaltest(np.log(all_bacteria[var]))
    #     print('test for normality:', normal_da_p[0], normal_da_p[1])
    #     print('test for log-normality:', log_normal_da_p[0], log_normal_da_p[1])
    #     # sns.kdeplot(all_bacteria[var], label=var)
    #     # sns.kdeplot(normal, label='normal')
    #     # sns.kdeplot(lognormal, label='lognormal')
    #     # plt.legend()
    #     # plt.show()
    #     # plt.close()
    # exit()

    #### It is the log of delta addded length!

    # so that every dataset has 88 sets
    np.random.seed(42)
    sis_keys = np.random.choice(list(sis_A.keys()), size=88, replace=False)
    new_sis_A = dict([(key, sis_A[key]) for key in sis_keys])
    new_sis_B = dict([(key, sis_B[key]) for key in sis_keys])

    A_variable_symbol = [r'$\ln(\tau)_A$', r'$\ln(x(0))_A$', r'$\ln(x(\tau))_A$', r'$\ln(\alpha)_A$', r'$\ln(\phi)_A$', r'$f_A$', r'$\Delta_A$']
    B_variable_symbol = [r'$\ln(\tau)_B$', r'$\ln(x(0))_B$', r'$\ln(x(\tau))_B$', r'$\ln(\alpha)_B$', r'$\ln(\phi)_B$', r'$f_B$', r'$\Delta_B$']
    
    pooled_sis = get_pooled_trap_df(A_df=new_sis_A, B_df=new_sis_B, variable_names=variable_names, A_variable_symbol=A_variable_symbol, B_variable_symbol=B_variable_symbol)
    pooled_non_sis = get_pooled_trap_df(A_df=non_sis_A, B_df=non_sis_B, variable_names=variable_names, A_variable_symbol=A_variable_symbol, B_variable_symbol=B_variable_symbol)
    pooled_con = get_pooled_trap_df(A_df=con_A, B_df=con_B, variable_names=variable_names, A_variable_symbol=A_variable_symbol, B_variable_symbol=B_variable_symbol)

    # sis_heatmap = pd.DataFrame(columns=A_variable_symbol, index=B_variable_symbol)

    print(len(pooled_sis))
    step = int(np.ceil((len(pooled_sis)-2)/100))
    print(step)
    # k_array = np.arange(1, len(pooled_sis), step)
    k_array = np.arange(1, 16)
    # exit()

    # k_array = np.arange(1, len(pooled_sis), step)
    # array = []
    # for k in k_array:
    #     for varB in B_variable_symbol:
    #         print(mutual_info_regression(X=np.array(pooled_sis[A_variable_symbol]).reshape(-1, 7), y=np.array(pooled_sis[varB]), discrete_features='auto', n_neighbors=k, copy=True, random_state=42))
    #         array.append(mutual_info_regression(X=np.array(pooled_sis[A_variable_symbol]).reshape(-1, 7), y=np.array(pooled_sis[varB]), discrete_features='auto', n_neighbors=k, copy=True, random_state=42)[0])
    #
    # array = np.array(array)
    # for ind, varB in zip(range(array.shape[1]), B_variable_symbol):
    #     print(varB)
    #     print(dict(zip(A_variable_symbol, array[:, ind])))
    # exit()

    # for var1 in variable_names:
    #     for var2 in variable_names:
    #         if var1 != 'generationtime' and var2 != 'generationtime':
    #             print(var1, var2)
    #             array=[]
    #             array1 = []
    #             p, sig1, sig2 = .8, 1, 1
    #             simulated = np.random.multivariate_normal([0, 0], [[sig1 ** 2, sig1 * sig2 * p], [sig1 * sig2 * p, sig2 ** 2]], size=len(all_bacteria))
    #             sim_x, sim_y = simulated[:, 0], simulated[:, 1]
    #             for k in k_array:
    #                 # print(varA, varB, k, mutual_info_regression(X=np.array(pooled_sis[varA]).reshape(-1, 1), y=np.array(pooled_sis[varB]).reshape(-1, 1), discrete_features=False, n_neighbors=3, copy=True, random_state=42)[0])
    #                 array.append(
    #                     mutual_info_regression(X=np.array(all_bacteria[var1]).reshape(-1, 1), y=np.array(all_bacteria[var2]), discrete_features=False,
    #                                            n_neighbors=k, copy=True, random_state=42)[0])
    #                 array1.append(
    #                     mutual_info_regression(X=sim_x.reshape(-1, 1), y=sim_y, discrete_features=False,
    #                                            n_neighbors=k, copy=True, random_state=42)[0])
    #                 # exit()
    #             plt.plot(k_array, array, marker='.', color='blue')
    #             plt.plot(k_array, array1, marker='.', color='purple')
    #             plt.axhline(-.5 * (np.log(1 - (stats.pearsonr(np.array(all_bacteria[var1]), np.array(all_bacteria[var2]))[0] ** 2))),
    #                         label='exact relationship to Gaussian', color='blue', ls='--')
    #             plt.axhline(-.5 * (np.log(1 - (stats.pearsonr(sim_x, sim_y)[0] ** 2))),
    #                         label='exact relationship to Gaussian simulated', color='purple', ls='--')
    #             print(stats.pearsonr(np.array(all_bacteria[var1]), np.array(all_bacteria[var2]))[0])
    #             print(-.5 * (np.log(1 - (stats.pearsonr(sim_x, sim_y)[0] ** 2))))
    #             print(-.5 * (np.log(1 - (p ** 2))))
    #             plt.title(r'{} {}'.format(var1, var2))
    #             plt.legend()
    #             plt.xlabel('k')
    #             plt.ylabel('MI')
    #             plt.show()
    #             plt.close()


    for varA in A_variable_symbol:
        for varB in B_variable_symbol:
            if varA != A_variable_symbol[0] and varB != B_variable_symbol[0]:
                array = []
                array1 = []
                p, sig1, sig2 = .8, 1, 1
                simulated = np.random.multivariate_normal([0, 0], [[sig1**2, sig1*sig2*p], [sig1*sig2*p, sig2**2]], size=len(pooled_sis))
                sim_x, sim_y = simulated[:, 0], simulated[:, 1]
                for k in k_array:
                    # print(varA, varB, k, mutual_info_regression(X=np.array(pooled_sis[varA]).reshape(-1, 1), y=np.array(pooled_sis[varB]).reshape(-1, 1), discrete_features=False, n_neighbors=3, copy=True, random_state=42)[0])
                    array.append(mutual_info_regression(X=np.array(pooled_sis[varA]).reshape(-1, 1), y=np.array(pooled_sis[varB]), discrete_features=False, n_neighbors=k, copy=True, random_state=42)[0])
                    array1.append(
                        mutual_info_regression(X=sim_x.reshape(-1, 1), y=sim_y, discrete_features=False,
                                               n_neighbors=k, copy=True, random_state=42)[0])
                    # exit()
                plt.plot(k_array, array, marker='.', color='blue')
                plt.plot(k_array, array1, marker='.', color='purple')
                plt.axhline(-.5 * (np.log(1 - (stats.pearsonr(np.array(pooled_sis[varA]), np.array(pooled_sis[varB]))[0] ** 2))),
                            label='exact relationship to Gaussian', color='blue', ls='--')
                plt.axhline(-.5 * (np.log(1 - (stats.pearsonr(sim_x, sim_y)[0] ** 2))),
                            label='exact relationship to Gaussian simulated', color='purple', ls='--')
                print(stats.pearsonr(np.array(pooled_sis[varA]), np.array(pooled_sis[varB]))[0])
                print(stats.pearsonr(sim_x, sim_y)[0], p)
                print(-.5 * (np.log(1 - (stats.pearsonr(sim_x, sim_y)[0] ** 2))))
                print(-.5 * (np.log(1 - (p ** 2))))
                print('--------')
                plt.title(r'{} {}'.format(varA, varB))
                plt.legend()
                plt.xlabel('k')
                plt.ylabel('MI')
                plt.show()
                plt.close()
            else:
                print(varA, varB)


    exit()
    sis_heatmap = for_dataframe1(dataframe=pooled_sis, A_variable_symbol=A_variable_symbol, B_variable_symbol=B_variable_symbol, k=3, base=2)
    sns.heatmap(sis_heatmap)
    plt.show()
    exit()

    calculate_MI_and_save_heatmap_for_all_dsets_together(pooled_sis, pooled_non_sis, pooled_con, 'Pooled traps', variable_names,
                                                         A_variable_symbol, B_variable_symbol)

    for var1 in variable_names:
        for var2 in variable_names:
            sns.distplot(np.array([new_sis_mis_dict_not_pooled[key][var1].loc[var2] for key in new_sis_mis_dict_not_pooled.keys()]), label='Sisters')
            sns.distplot(np.array([non_sis_mis_dict_not_pooled[key][var1].loc[var2] for key in non_sis_mis_dict_not_pooled.keys()]), label='Non-Sisters')
            sns.distplot(np.array([con_mis_dict_not_pooled[key][var1].loc[var2] for key in con_mis_dict_not_pooled.keys()]), label='Control')
            plt.title('Mutual Information')
            plt.legend()
            plt.xlabel(var1+'_A and '+var2+'_B')
            plt.show()
            plt.close()

    exit()

    new_sis_mis_dict_not_pooled = for_dataset(datasetA=new_sis_A, datasetB=new_sis_B, variable_names=variable_names)
    non_sis_mis_dict_not_pooled = for_dataset(datasetA=non_sis_A, datasetB=non_sis_B, variable_names=variable_names)
    con_mis_dict_not_pooled = for_dataset(datasetA=con_A, datasetB=con_B, variable_names=variable_names)

    for var1 in variable_names:
        for var2 in variable_names:
            sns.distplot(np.array([new_sis_mis_dict_not_pooled[key][var1].loc[var2] for key in new_sis_mis_dict_not_pooled.keys()]), label='Sisters')
            sns.distplot(np.array([non_sis_mis_dict_not_pooled[key][var1].loc[var2] for key in non_sis_mis_dict_not_pooled.keys()]), label='Non-Sisters')
            sns.distplot(np.array([con_mis_dict_not_pooled[key][var1].loc[var2] for key in con_mis_dict_not_pooled.keys()]), label='Control')
            plt.title('Mutual Information')
            plt.legend()
            plt.xlabel(var1+'_A and '+var2+'_B')
            plt.show()
            plt.close()
    
    
        































if __name__ == '__main__':
    main()
