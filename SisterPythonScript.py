

from __future__ import print_function

import numpy as np
import argparse
import sys,math
import glob
import matplotlib.pyplot as plt

import sistercellclass as ssc


def xstar(Ensemble, discretize_by, traj, both):
    # Sisters or Nonsisters are the inputs most likely
    # discretize_by is either length, flourescence or cellarea
    # traj is either 'A' or 'B'
    # if ensemble is to take from both, make both = True
    if both:
        return np.mean(np.sum([Ensemble[dataid][discretize_by + 'A'][0] for dataid in range(len(Ensemble))],
                              [Ensemble[dataid][discretize_by + 'B'][0] for dataid in range(len(Ensemble))], axis=0)/2.)
    else:
        return np.mean([Ensemble[dataid][discretize_by + traj][0] for dataid in range(len(Ensemble))])


def xstar1(all_dict):
    # Sisters or Nonsisters are the inputs most likely
    # discretize_by is either length, flourescence or cellarea
    # traj is either 'A' or 'B'
    # if ensemble is to take from both, make both = True
    return np.mean([np.mean(all_dict[key]['length_birth']) for key in all_dict.keys()])


def lsqm(x,y,cov):
    n = len(x)
    sx = np.sum(x)
    sy = np.sum(y)
    sxx = np.dot(x, x)
    sxy = np.dot(x, y)
    syy = np.dot(y, y)

    # estimate parameters
    denom = (n * sxx - sx * sx)
    b = (n * sxy - sx * sy) / denom
    a = (sy - b * sx) / n
    estimate = np.array([a, b], dtype=np.float)

    if cov:
        # estimate covariance matrix of estimated parameters
        sigma2 = syy + n * a * a + b * b * sxx + 2 * a * b * sx - 2 * a * sy - 2 * b * sxy  # variance of deviations from linear line
        covmatrix = sigma2 / denom * np.array([[sxx, -sx], [-sx, n]], dtype=np.float)

        return estimate, covmatrix
    else:
        return estimate


def CheckDivTimes(traj, Data, dataid, dictionary, mydir):
    # This function plots the trajectories of the lengths and their division times to check if they are correct

    # traj='A' or 'B'
    # Data is the raw data, ie. Sisters, Nonsisters or some mixture of the two
    # dictionary is the dictionary where we keep all the processed data
    # mydir is the directory you would like to save the files to
    plt.figure(figsize=(20, 20))
    plt.plot(Data[dataid]['time'+traj], Data[dataid]['length'+traj], color='b', linewidth=4)
    div_time_array = Data[dataid]['time'+traj][0] + np.cumsum(np.array([dictionary[k]['generationtime'] for
                            k in dictionary.keys() if k == 'Non-sister_Trace_' + traj + '_' + str(dataid)]))
    print('divetimearray',div_time_array)
    print('dataid', dataid)
    # plot the vertical lines
    for div_time in div_time_array:
        plt.axvline(x=div_time, color='r', linewidth=2)

    plt.savefig(mydir + str(dataid) + '_non_sisters_' + str(traj) + '.png', bbox_inches='tight')
    plt.clf()
    # del div_time_array


def GetBetas(dict_of_trajs_df):
    for dfA, dfB in zip(A_dict_sis.values(), B_dict_sis.values()):
        phi_star_A, beta_A = lsqm(x=np.log(np.divide(dfA['length_birth'], x_avg)), y=np.multiply(dfA['generationtime'
                                                                                                 ],
                                                                                                 dfA['growth_length']),
                                  cov=False)
        phi_star_B, beta_B = lsqm(x=np.log(np.divide(dfB['length_birth'], x_avg)), y=np.multiply(dfB['generationtime'
                                                                                                 ],
                                                                                                 dfB['growth_length']),
                                  cov=False)
        # plt.scatter(dfA['length_birth'])
        t = np.arange(-2.5, 2.5)
        s = phi_star_A + beta_A * t
        s1 = phi_star_B + beta_B * t
        ff = plt.plot(s, color='r', label='Sister Cells')
        gg = plt.plot(s1, color='b', label='Non-Sister Cells')
    plt.title(r'Individual $\beta$ for Sister/Non-Sister Cells Trajectories')
    plt.xlabel(r'$ln(\frac{x_n^{(k)}}{x^*})$', fontsize=25)
    plt.ylabel(r'$\phi^{(k)}_n$', fontsize=25)
    plt.legend(handles=[ff[0], gg[0]])
    plt.show()


# For Mac
infiles_sisters = glob.glob(r'/Users/alestawsky/PycharmProjects/untitled/SISTERS/*.xls')
infiles_nonsisters = glob.glob(r'/Users/alestawsky/PycharmProjects/untitled/SISTERS/*.xls')

# # For windows
# infiles_sisters = glob.glob(r"C:\Users\alejandros\Downloads\cleaned_datafiles\cleaned_datafiles\SISTERS-NONSISTERS\SISTERS\*.xls")
# infiles_nonsisters = glob.glob(r"C:\Users\alejandros\Downloads\cleaned_datafiles\cleaned_datafiles\SISTERS-NONSISTERS\NONSISTERS\*.xls")

Sisters = ssc.SisterCellData(infiles = infiles_sisters, debugmode = True)
Nonsisters = ssc.SisterCellData(infiles = infiles_nonsisters, debugmode = True)

meth = 'threshold'

# CREATES THE DICT_WITH_ALL_SISTER_TRACES AND NEXT ONE NON_SISTERS
meth = 'threshold'
dict_with_all_sister_traces = {'none' : 1}
for dataID in range(0, len(infiles_sisters), 1):

    dict_with_all_sister_traces = Sisters.DictionaryOfSisterTraces(dataID, meth, discretize_by = 'length',
        dictionary = dict_with_all_sister_traces, sis=True)
    if dataID == 0:
        del dict_with_all_sister_traces['none']

meth = 'threshold'
dict_with_all_non_sister_traces = {'none' : 1}
for dataID in range(0, len(infiles_nonsisters) - 1, 1):
    dict_with_all_non_sister_traces = Nonsisters.DictionaryOfSisterTraces(dataID, meth, discretize_by='length',
                                                                dictionary=dict_with_all_non_sister_traces, sis=False)
    if dataID == 0:
        del dict_with_all_non_sister_traces['none']


########################################################################################################################
# FORMAT NAME: 'Sister_Trace_A_' + str(dataID) OR 'Non-sister_Trace_A_' + str(dataID) ##################################
########################################################################################################################


dict_all = dict()
dict_all.update(dict_with_all_sister_traces)
dict_all.update(dict_with_all_non_sister_traces)

x_avg = xstar1(dict_all)
A_dict_sis = dict(zip([k for k,v in dict_with_all_sister_traces.items() if 'Sister_Trace_A_' in k],
                      [v for k,v in dict_with_all_sister_traces.items() if 'Sister_Trace_A_' in k]))
# x_avg = xstar(Nonsisters, discretize_by='length', traj='A', both=False)
A_dict_non_sis = dict(zip([k for k,v in dict_with_all_non_sister_traces.items() if 'Non-sister_Trace_A_' in k],
                      [v for k,v in dict_with_all_non_sister_traces.items() if 'Non-sister_Trace_A_' in k]))

# x_avg = xstar(Nonsisters, discretize_by='lengthA'discretize_by='length', traj='A', both=False)
B_dict_sis = dict(zip([k for k,v in dict_with_all_sister_traces.items() if 'Sister_Trace_B_' in k],
                      [v for k,v in dict_with_all_sister_traces.items() if 'Sister_Trace_B_' in k]))
B_dict_non_sis = dict(zip([k for k,v in dict_with_all_non_sister_traces.items() if 'Non-sister_Trace_B_' in k],
                      [v for k,v in dict_with_all_non_sister_traces.items() if 'Non-sister_Trace_B_' in k]))

meta_data = dict(mapping = [str([Sisters, Nonsisters, dict_with_all_sister_traces, dict_with_all_non_sister_traces, dict_all, x_avg, A_dict_sis,\
       A_dict_non_sis, B_dict_sis, B_dict_non_sis]),[Sisters, Nonsisters, dict_with_all_sister_traces, dict_with_all_non_sister_traces, dict_all, x_avg, A_dict_sis,\
       A_dict_non_sis, B_dict_sis, B_dict_non_sis]])

print('SisterPythonScript.py has been executed')
print('locals: ', len(locals()), 'now globals: ', len(globals()))
# loc = locals()
# print('loc', loc)
# print('length of integrated globals: ',  len(globals()))


def main():

    # Checks if it works
    all_data_dictionary = dict_with_all_sister_traces
    print(len(all_data_dictionary.keys()))
    all_data_dictionary.update(dict_with_all_non_sister_traces)
    print(len(all_data_dictionary.keys()))
    print(all_data_dictionary['Sister_Trace_A_21'])


if __name__ == '__main__':
    main()


