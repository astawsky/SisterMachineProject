from __future__ import print_function

import numpy as np
import argparse
import sys, math
import glob
import matplotlib.pyplot as plt

import pickle
import matplotlib.patches as mpatches

import sistercellclass as ssc

import CALCULATETHEBETAS
import os

import scipy.stats as stats
import random


def pooling(number, cycle, A_dict, B_dict, dict_all, x_avg):
    # CHECKING THE RIGHT POOLING OPTION ACCORDING TO THE WHITEBOARD
    if number not in np.array([1, 2, 3]):
        IOError('wrong pooling option')

    # IMPLEMENTING THE POOLING OPTIONS for ln(x^A/x*) vs. ln(x^B/x*)
    if number == 0:
        # Here we use the minimum of the max of A and B cycles for the IDs that contain at least "cycle" amount of cycles

        param = 'length_birth'

        # Since either dispA or dispB is equal to 0, cycle + dispA + dispB = cycle + disp
        A_arrays = np.array([np.array([np.log(A_dict[idA][param][cyc] / x_avg) for cyc in
                                       range(min(len(A_dict[idA][param]), len(B_dict[idB][param])))])
                             for idA, idB in zip(A_dict.keys(), B_dict.keys())
                             if min(len(A_dict[idA]), len(B_dict[idB])) > cycle])

        B_arrays = np.array([np.array([np.log(B_dict[idB][param][cyc] / x_avg) for cyc in
                                       range(min(len(A_dict[idA][param]), len(B_dict[idB][param])))])
                             for idA, idB in zip(A_dict.keys(), B_dict.keys())
                             if min(len(A_dict[idA]), len(B_dict[idB])) > cycle])

        B_trace = np.array([])
        for l in range(len(B_arrays)):
            B_trace = np.concatenate((B_trace, B_arrays[l]))

        A_trace = np.array([])
        for l in range(len(A_arrays)):
            A_trace = np.concatenate((A_trace, A_arrays[l]))

        # WE NEED SAME SIZES FOR THE X,Y-PLANE
        if len(A_trace) != len(B_trace):
            IOError("trace sizes don't match!")

        number_of_points = len(A_trace)  # We could've used B_trace instead... Doesn't matter

        return A_trace, B_trace, number_of_points

    # FOR THIS OPTION WE HAVE THE FOLDER
    elif number == 1:

        # Here we use "cycle" amount of cycles from A and B for the IDs that contain at least "cycle" amount of cycles

        param = 'length_birth'

        # Since either dispA or dispB is equal to 0, cycle + dispA + dispB = cycle + disp
        
        # Has first dimension as id and second as wanted value for "cyc" cycle
        x_trace_A = np.array([np.array([np.log(A_dict[idA][param][cyc] / x_avg) for cyc in range(cycle)])
                             for idA, idB in zip(A_dict.keys(), B_dict.keys())
                             if min(len(A_dict[idA]), len(B_dict[idB])) > cycle])

        y_phi_A = np.array([np.array([A_dict[idA]['growth_length'][cyc]*A_dict[idA]['generationtime'][cyc]
                                       for cyc in range(cycle)])
                             for idA, idB in zip(A_dict.keys(), B_dict.keys())
                             if min(len(A_dict[idA]), len(B_dict[idB])) > cycle])

        x_trace_B = np.array([np.array([np.log(B_dict[idB][param][cyc] / x_avg) for cyc in range(cycle)])
                              for idA, idB in zip(A_dict.keys(), B_dict.keys())
                              if min(len(A_dict[idA]), len(B_dict[idB])) > cycle])

        y_phi_B = np.array([np.array([B_dict[idB]['growth_length'][cyc] * B_dict[idB]['generationtime'][cyc]
                                      for cyc in range(cycle)])
                            for idA, idB in zip(A_dict.keys(), B_dict.keys())
                            if min(len(A_dict[idA]), len(B_dict[idB])) > cycle])

        return x_trace_A, y_phi_A, x_trace_B, y_phi_B

    # FOR THIS OPTION WE HAVE THE FOLDER "Scatter, GenPlots, and BetaPhi plots of all cells in dset that have cycle
    # generations and use "cycle" number of points for each trace"
    elif number == 2: # SEPARATES JUST BY DSET

        # Here we use cycle amount of cycles from A and B for the IDs that contain at least "cycle" amount of cycles

        param = 'length_birth'

        x_trace = np.array([np.array([np.log(dict_all[ID][param][cyc] / x_avg) for cyc in range(cycle)])
                              for ID in dict_all.keys() if len(dict_all[ID]) > cycle])

        y_phi = np.array([np.array([dict_all[ID]['growth_length'][cyc] * dict_all[ID]['generationtime'][cyc]
                            for cyc in range(cycle)]) for ID in dict_all.keys() if len(dict_all[ID]) > cycle])

        trace = np.array([])
        for l in range(len(x_trace)):
            trace = np.concatenate((trace, x_trace[l]))

        phi = np.array([])
        for l in range(len(y_phi)):
            phi = np.concatenate((phi, y_phi[l]))

        # WE NEED SAME SIZES FOR THE X,Y-PLANE
        if len(phi) != len(trace):
            IOError("trace sizes don't match!")

        number_of_points = len(phi)  # We could've used trace instead... Doesn't matter

        return trace, phi, number_of_points

    # FOR THIS OPTION WE HAVE THE FOLDER Generation Specific phi and ln(x/x*) scatter and generationplot [NO RESULTS]
    elif number == 3:

        # Here we use 1 cycle ("cycle" cycle) from A and B for the IDs that contain at least "cycle" amount of cycles

        param = 'length_birth'

        # trace is the log(x_n/x*) and phi is the fold-growth
        x_trace = np.array([np.log(dict_all[ID][param][cycle] / x_avg) for ID in dict_all.keys() if len(dict_all[ID]) > cycle])

        y_phi = np.array([dict_all[ID]['growth_length'][cycle] * dict_all[ID]['generationtime'][cycle]
                          for ID in dict_all.keys() if len(dict_all[ID]) > cycle])

        # WE NEED SAME SIZES FOR THE X,Y-PLANE
        if len(x_trace) != len(y_phi):
            IOError("trace sizes don't match!")

        number_of_points = len(y_phi)

        return x_trace, y_phi, number_of_points


# Don't use it anymore!
def createDirectories():
    # MAKING THE DIRECTORIES TO PUT THE PLOTS, THE NAMES WILL BE THE GENERATIONAL DISPLACEMENT UTILISED

    # SCATTER PLOTS OF THE TWO DICTIONARIES INPUTTED
    os.mkdir('WhiteBoard Analysis Plots ln(x) vs phi_n/Scatter Plots')

    os.mkdir('WhiteBoard Analysis Plots ln(x) vs phi_n/Scatter Plots/A Displacement')
    os.mkdir(
        'WhiteBoard Analysis Plots ln(x) vs phi_n/Scatter Plots/A Displacement/all cycles pooling')  # number 1 on White Board
    os.mkdir(
        'WhiteBoard Analysis Plots ln(x) vs phi_n/Scatter Plots/A Displacement/number of cycles is number of cycles pooled')  # number 2 on White Board
    os.mkdir(
        'WhiteBoard Analysis Plots ln(x) vs phi_n/Scatter Plots/A Displacement/single cycle pooling')  # number 3 on White Board

    os.mkdir('WhiteBoard Analysis Plots ln(x) vs phi_n/Scatter Plots/B Displacement')
    os.mkdir(
        'WhiteBoard Analysis Plots ln(x) vs phi_n/Scatter Plots/B Displacement/all cycles pooling')  # number 1 on White Board
    os.mkdir(
        'WhiteBoard Analysis Plots ln(x) vs phi_n/Scatter Plots/B Displacement/number of cycles is number of cycles pooled')  # number 2 on White Board
    os.mkdir(
        'WhiteBoard Analysis Plots ln(x) vs phi_n/Scatter Plots/B Displacement/single cycle pooling')  # number 3 on White Board


    # PEARSON AND LEAST-SQUARES SLOPE AND INTERCEPT OF THE TWO DICTIONARIES INPUTTED (NUMBER OF POINTS USED
    #                                                                               WILL ALSO BE ON THE BACKGROUND)
    os.mkdir('WhiteBoard Analysis Plots ln(x) vs phi_n/Pearson, Slope and Intercept Plots')

    os.mkdir('WhiteBoard Analysis Plots ln(x) vs phi_n/Pearson, Slope and Intercept Plots/A Displacement')
    os.mkdir(
        'WhiteBoard Analysis Plots ln(x) vs phi_n/Pearson, Slope and Intercept Plots/A Displacement/all cycles pooling')  # number 1 on White Board
    os.mkdir(
        'WhiteBoard Analysis Plots ln(x) vs phi_n/Pearson, Slope and Intercept Plots/A Displacement/number of cycles is number of cycles pooled')  # number 2 on White Board
    os.mkdir(
        'WhiteBoard Analysis Plots ln(x) vs phi_n/Pearson, Slope and Intercept Plots/A Displacement/single cycle pooling')  # number 3 on White Board

    os.mkdir('WhiteBoard Analysis Plots ln(x) vs phi_n/Pearson, Slope and Intercept Plots/B Displacement')
    os.mkdir(
        'WhiteBoard Analysis Plots ln(x) vs phi_n/Pearson, Slope and Intercept Plots/B Displacement/all cycles pooling')  # number 1 on White Board
    os.mkdir(
        'WhiteBoard Analysis Plots ln(x) vs phi_n/Pearson, Slope and Intercept Plots/B Displacement/number of cycles is number of cycles pooled')  # number 2 on White Board
    os.mkdir(
        'WhiteBoard Analysis Plots ln(x) vs phi_n/Pearson, Slope and Intercept Plots/B Displacement/single cycle pooling')  # number 3 on White Board


def scatterPlot(x_trace, y_phi, xlabel, ylabel, PoolID, cycle, dset):
    # HERE WE SAVE THE SCATTER PLOT

    # Decide where to save the graph

    if PoolID == 1:
        pool = 'all cycles pooling/'
    elif PoolID == 2:
        pool = 'number of cycles is number of cycles pooled/'
    elif PoolID == 3:
        pool = 'single cycle pooling/'

    plt.scatter(x_trace, y_phi, label='number of points: ' + str(len(y_phi)))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim([.3, 1.2])
    plt.xlim([-.6, .6])
    plt.legend()
    plt.savefig('Scatter Plot for '+str(dset)+' of the cycle '+str(cycle))
    plt.close()
    # plt.show()


def GenerationPlot(length_GenPlotArray, phi_GenPlotArray, xlabel, PoolID, dset):
    if len(length_GenPlotArray) != len(phi_GenPlotArray):
        IOError('not same sizes!')

    pcoeffArray = np.array([])
    slopeArray = np.array([])
    interceptArray = np.array([])

    for ind in range(len(length_GenPlotArray)):
        slope, intercept, r_value, p_value, std_err = stats.linregress(length_GenPlotArray[ind], phi_GenPlotArray[ind])
        pcoeff, pval = stats.pearsonr(length_GenPlotArray[ind], phi_GenPlotArray[ind])
        pcoeffArray = np.append(pcoeffArray, pcoeff)
        slopeArray = np.append(slopeArray, slope)
        interceptArray = np.append(interceptArray, intercept)

    # Decide where to save the graph
    
    if PoolID == 1:
        pool = 'all cycles pooling/'
    elif PoolID == 2:
        pool = 'number of cycles is number of cycles pooled/'
    elif PoolID == 3:
        pool = 'single cycle pooling/'

    # plot the pearson coefficient over generation relations
    plt.plot(pcoeffArray, marker='.')
    plt.xlabel(xlabel)  # GENERATION NUMBER
    plt.ylabel('Pearson Correlation Coefficient')  # PEARSON COEFF
    plt.ylim([-1, 1])
    plt.savefig('Pearson Correlation across generations for '+str(dset))
    plt.close()
    # plt.show()

    # plot the slope and intercept over generation relations
    plt.plot(slopeArray, label=r'$\beta$', marker='.')
    plt.plot(interceptArray, label=r'$\phi^*$ intercept', marker='.')
    plt.xlabel(xlabel)  # GENERATION NUMBER
    plt.ylabel('slope and intercept value')  # slope and intercept
    plt.ylim([-1, 1])
    plt.legend()
    plt.savefig('Slope and Intercept across generations for ' + str(dset))
    plt.close()
    # plt.show()

def Distribution(diffs_sis, diffs_non_sis, diffs_both):
    # PoolID == 1
    sis_label = 'Sister ' + r'$\mu=$' + '{:.2e}'.format(np.mean(diffs_sis)) + r', $\sigma=$' + '{:.2e}'.format(
        np.var(diffs_sis))
    non_label = 'Non Sister ' + r'$\mu=$' + '{:.2e}'.format(
        np.mean(diffs_non_sis)) + r', $\sigma=$' + '{:.2e}'.format(np.var(diffs_non_sis))
    both_label = 'Control ' + r'$\mu=$' + '{:.2e}'.format(np.mean(diffs_both)) + r', $\sigma=$' + '{:.2e}'.format(
        np.var(diffs_both))
    
    arr_sis = plt.hist(x=diffs_sis, label=sis_label, weights=np.ones_like(diffs_sis) / float(len(diffs_sis)))
    arr_non_sis = plt.hist(x=diffs_non_sis, label=non_label, weights=np.ones_like(diffs_non_sis) / float(len(diffs_non_sis)))
    arr_both = plt.hist(x=diffs_both, label=both_label, weights=np.ones_like(diffs_both) / float(len(diffs_both)))
    plt.close()

    print('arr_sis[0]:', arr_sis[0])
    print('arr_sis[1]:', arr_sis[1])
    plt.plot(np.array([(arr_sis[1][l]+arr_sis[1][l+1])/2. for l in range(len(arr_sis)-1)]), arr_sis[0], label=sis_label, marker='.')
    plt.plot(arr_non_sis[1][1:], arr_non_sis[0], label=non_label, marker='.')
    plt.plot(arr_both[1][1:], arr_both[0], label=both_label, marker='.')
    plt.xlabel('value of the difference in mean')
    plt.ylabel('PDF (Weighted Histogram)')
    plt.legend()
    plt.show()


def main():
    # Import the Refined Data
    pickle_in = open("metastructdata.pickle", "rb")
    metadatastruct = pickle.load(pickle_in)
    pickle_in.close()

    # createDirectories()

    # FOR THE GENERATION PLOT
    length_GenPlotArray_sis = []
    length_GenPlotArray_non_sis = []
    length_GenPlotArray_both = []
    phi_GenPlotArray_sis = []
    phi_GenPlotArray_non_sis = []
    phi_GenPlotArray_both = []

    # FOR THE GENERATION PLOT AND POOLID == 1
    length_GenPlotArray_A_sis = []
    length_GenPlotArray_A_non_sis = []
    length_GenPlotArray_A_both = []
    phi_GenPlotArray_A_sis = []
    phi_GenPlotArray_A_non_sis = []
    phi_GenPlotArray_A_both = []

    length_GenPlotArray_B_sis = []
    length_GenPlotArray_B_non_sis = []
    length_GenPlotArray_B_both = []
    phi_GenPlotArray_B_sis = []
    phi_GenPlotArray_B_non_sis = []
    phi_GenPlotArray_B_both = []

    # CHOOSES HOW MANY CYCLES TO DO THIS FOR
    how_many_cycles = 80
    
    # CAN BE 1,2,3
    PoolID = 1

    # FOR THE POOLING FUNCTION, MAINLY TO GET THE SCATTER PLOT
    for cycle in range(10,how_many_cycles): # NO MINIMUM STARTING NUMBER FOR POOLID == 1

        if PoolID == 1: # splits by A/B trace
            # SISTERS
            x_trace_A_sis, y_phi_A_sis, x_trace_B_sis, y_phi_B_sis = pooling(number=PoolID, cycle=cycle,
                                                                   A_dict=metadatastruct.A_dict_sis,
                                                                   B_dict=metadatastruct.B_dict_sis,
                                                                   dict_all=metadatastruct.dict_with_all_sister_traces,
                                                                   x_avg=metadatastruct.x_avg)

            scatterPlot(x_trace=x_trace_A_sis, y_phi=y_phi_A_sis, xlabel=r'$ln(\frac{x^A_n}{x^*})$',
                        ylabel=r'$\phi^A_n$', PoolID=PoolID, cycle=cycle, dset="Sisters")
            
            scatterPlot(x_trace=x_trace_B_sis, y_phi=y_phi_B_sis, xlabel=r'$ln(\frac{x^B_n}{x^*})$',
                        ylabel=r'$\phi^B_n$', PoolID=PoolID, cycle=cycle, dset="Sisters")

            # NON-SISTERS
            x_trace_A_non_sis, y_phi_A_non_sis, x_trace_B_non_sis, y_phi_B_non_sis = pooling(number=PoolID, cycle=cycle,
                                                                               A_dict=metadatastruct.A_dict_non_sis,
                                                                               B_dict=metadatastruct.B_dict_non_sis,
                                                                               dict_all=metadatastruct.dict_with_all_non_sister_traces,
                                                                               x_avg=metadatastruct.x_avg)

            scatterPlot(x_trace=x_trace_A_non_sis, y_phi=y_phi_A_non_sis, xlabel=r'$ln(\frac{x^A_n}{x^*})$',
                        ylabel=r'$\phi^A_n$', PoolID=PoolID, cycle=cycle, dset="Non")

            scatterPlot(x_trace=x_trace_B_non_sis, y_phi=y_phi_B_non_sis, xlabel=r'$ln(\frac{x^B_n}{x^*})$',
                        ylabel=r'$\phi^B_n$', PoolID=PoolID, cycle=cycle, dset="Non")

            # CONTROL
            x_trace_A_both, y_phi_A_both, x_trace_B_both, y_phi_B_both = pooling(number=PoolID, cycle=cycle,
                                                                      A_dict=metadatastruct.A_dict_both,
                                                                      B_dict=metadatastruct.B_dict_both,
                                                                      dict_all=metadatastruct.dict_with_all_both_traces,
                                                                      x_avg=metadatastruct.x_avg)

            scatterPlot(x_trace=x_trace_A_both, y_phi=y_phi_A_both, xlabel=r'$ln(\frac{x^A_n}{x^*})$',
                        ylabel=r'$\phi^A_n$', PoolID=PoolID, cycle=cycle, dset="Control")

            scatterPlot(x_trace=x_trace_B_both, y_phi=y_phi_B_both, xlabel=r'$ln(\frac{x^B_n}{x^*})$',
                        ylabel=r'$\phi^B_n$', PoolID=PoolID, cycle=cycle, dset="Control")
            
            diffs_sis = x_trace_A_sis - x_trace_B_sis
            diffs_non_sis = x_trace_A_non_sis - x_trace_B_non_sis
            diffs_both = x_trace_A_both - x_trace_B_both
            
            Distribution(diffs_sis=diffs_sis, diffs_non_sis=diffs_non_sis, diffs_both=diffs_both)

        else:
            # SISTERS
            x_trace_sis, y_phi_sis, number_of_points_sis = pooling(number=PoolID, cycle=cycle, A_dict=metadatastruct.A_dict_sis,
                          B_dict=metadatastruct.B_dict_sis, dict_all=metadatastruct.dict_with_all_sister_traces,
                          x_avg=metadatastruct.x_avg)

            scatterPlot(x_trace=x_trace_sis, y_phi=y_phi_sis, xlabel=r'$ln(\frac{x_n}{x^*})$',
                        ylabel=r'$\phi_n$', PoolID=PoolID, cycle=cycle, dset="Sisters")

            # NON-SISTERS
            x_trace_non_sis, y_phi_non_sis, number_of_points_non_sis = pooling(number=PoolID, cycle=cycle,
                                                                   A_dict=metadatastruct.A_dict_non_sis,
                                                                   B_dict=metadatastruct.B_dict_non_sis,
                                                                   dict_all=metadatastruct.dict_with_all_non_sister_traces,
                                                                   x_avg=metadatastruct.x_avg)

            scatterPlot(x_trace=x_trace_non_sis, y_phi=y_phi_non_sis, xlabel=r'$ln(\frac{x_n}{x^*})$',
                        ylabel=r'$\phi_n$', PoolID=PoolID, cycle=cycle, dset="Non")

            # CONTROL
            x_trace_both, y_phi_both, number_of_points_both = pooling(number=PoolID, cycle=cycle,
                                                                   A_dict=metadatastruct.A_dict_both,
                                                                   B_dict=metadatastruct.B_dict_both,
                                                                   dict_all=metadatastruct.dict_with_all_both_traces,
                                                                   x_avg=metadatastruct.x_avg)

            scatterPlot(x_trace=x_trace_both, y_phi=y_phi_both, xlabel=r'$ln(\frac{x_n}{x^*})$',
                        ylabel=r'$\phi_n$', PoolID=PoolID, cycle=cycle, dset="Control")

        if PoolID == 1:
            # FOR THE GENPLOT
            length_GenPlotArray_A_sis.append(x_trace_A_sis)
            length_GenPlotArray_A_non_sis.append(x_trace_A_non_sis)
            length_GenPlotArray_A_both.append(x_trace_A_both)

            phi_GenPlotArray_A_sis.append(y_phi_A_sis)
            phi_GenPlotArray_A_non_sis.append(y_phi_A_non_sis)
            phi_GenPlotArray_A_both.append(y_phi_A_both)

            length_GenPlotArray_B_sis.append(x_trace_B_sis)
            length_GenPlotArray_B_non_sis.append(x_trace_B_non_sis)
            length_GenPlotArray_B_both.append(x_trace_B_both)

            phi_GenPlotArray_B_sis.append(y_phi_B_sis)
            phi_GenPlotArray_B_non_sis.append(y_phi_B_non_sis)
            phi_GenPlotArray_B_both.append(y_phi_B_both)

        # FOR THE GENPLOT
        length_GenPlotArray_sis.append(x_trace_sis)
        length_GenPlotArray_non_sis.append(x_trace_non_sis)
        length_GenPlotArray_both.append(x_trace_both)

        phi_GenPlotArray_sis.append(y_phi_sis)
        phi_GenPlotArray_non_sis.append(y_phi_non_sis)
        phi_GenPlotArray_both.append(y_phi_both)

    # RUN AND SAVE THE GENERATION PLOT
    GenerationPlot(length_GenPlotArray=length_GenPlotArray_sis, phi_GenPlotArray=phi_GenPlotArray_sis,
                   xlabel='Generation', PoolID=3, dset="Sisters")
    GenerationPlot(length_GenPlotArray=length_GenPlotArray_non_sis, phi_GenPlotArray=phi_GenPlotArray_non_sis,
                   xlabel='Generation', PoolID=3, dset="Non")
    GenerationPlot(length_GenPlotArray=length_GenPlotArray_both, phi_GenPlotArray=phi_GenPlotArray_both,
                   xlabel='Generation', PoolID=3, dset="Control")


    # FOR NOW WE ARE ONLY USING POOLID == 2,3 AND NOT 1 BECAUSE IT IS BETTER TO COMPARE THEM WITH THE PEARSON CORRELATIONS
    # LIKE WITH ln(x^A_n/x*) vs. ln(x^B_n/x*)


if __name__ == '__main__':
    main()
