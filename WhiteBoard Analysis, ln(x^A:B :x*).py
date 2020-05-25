
from __future__ import print_function

import numpy as np
import argparse
import sys,math
import glob
import matplotlib.pyplot as plt

import pickle
import matplotlib.patches as mpatches

import sistercellclass as ssc

import CALCULATETHEBETAS
import os

import scipy.stats as stats
import random

def pooling(number, cycle, trace, A_dict, B_dict, x_avg, disp):

    # CHECKING THE RIGHT POOLING OPTION ACCORDING TO THE WHITEBOARD
    if number not in np.array([1,2,3]):
        IOError('wrong pooling option')

    # A DISPLACEMENT CASE (COUSINS, AUNTS, NIECES, ETC...)
    if trace == 'A':
        dispA = disp
        dispB = 0

    # B DISPLACEMENT CASE (COUSINS, AUNTS, NIECES, ETC...)
    elif trace == 'B':
        dispA = 0
        dispB = disp

    # ITSELF DISPLACEMENT CASE (MOM, GRANDMA, DAUGHTER, ETC...)
    elif trace == 'Itself':

        # DOESN'T MATTER WHICH WE CHOOSE SINCE IN THIS CASE TRACE A AND B ARE THE SAME
        dispA = disp
        dispB = 0

    # IMPLEMENTING THE POOLING OPTIONS for ln(x^A/x*) vs. ln(x^B/x*)
    if number == 1:
        # Here we use the minimum of the max of A and B cycles for the IDs that contain at least "cycle" amount of cycles

        param = 'length_birth'

        # Since either dispA or dispB is equal to 0, cycle + dispA + dispB = cycle + disp
        A_arrays = np.array([np.array([np.log(A_dict[idA][param][cyc + dispA] / x_avg) for cyc in
                         range(min(len(A_dict[idA][param]) - dispA, len(B_dict[idB][param]) - dispB))]) for idA, idB in zip(A_dict.keys(), B_dict.keys())
                                       if min(len(A_dict[idA]), len(B_dict[idB])) > cycle + dispA + dispB])

        B_arrays = np.array([np.array([np.log(B_dict[idB][param][cyc + dispB] / x_avg) for cyc in
                         range(min(len(A_dict[idA][param]) - dispA, len(B_dict[idB][param]) - dispB))]) for idA, idB in zip(A_dict.keys(), B_dict.keys())
                         if min(len(A_dict[idA]), len(B_dict[idB])) > cycle + dispA + dispB])

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

    elif number == 2:

        # Here we use "cycle" amount of cycles from A and B for the IDs that contain at least "cycle" amount of cycles

        param = 'length_birth'

        # Since either dispA or dispB is equal to 0, cycle + dispA + dispB = cycle + disp
        A_arrays = np.array([np.array([np.log(A_dict[idA][param][cyc + dispA] / x_avg) for cyc in range(cycle)])
                             for idA, idB in zip(A_dict.keys(), B_dict.keys())
                             if min(len(A_dict[idA]), len(B_dict[idB])) > cycle + dispA + dispB])

        B_arrays = np.array([np.array([np.log(B_dict[idB][param][cyc + dispB] / x_avg) for cyc in range(cycle)])
                             for idA, idB in zip(A_dict.keys(), B_dict.keys())
                             if min(len(A_dict[idA]), len(B_dict[idB])) > cycle + dispA + dispB])

        B_trace = np.array([])
        for l in range(len(B_arrays)):
            B_trace = np.concatenate((B_trace, B_arrays[l]))

        A_trace = np.array([])
        for l in range(len(A_arrays)):
            A_trace = np.concatenate((A_trace, A_arrays[l]))

        # WE NEED SAME SIZES FOR THE X,Y-PLANE
        if len(A_trace) != len(B_trace):
            IOError("trace sizes don't match!")

        number_of_points = len(A_trace) # We could've used B_trace instead... Doesn't matter

        return A_trace, B_trace, number_of_points

    elif number == 3:

        # Here we use 1 cycle ("cycle" cycle) from A and B for the IDs that contain at least "cycle" amount of cycles

        param = 'length_birth'

        # Since either dispA or dispB is equal to 0, cycle + dispA + dispB = cycle + disp
        A_trace = np.array([np.log(A_dict[idA][param][cycle + dispA] / x_avg)
                             for idA, idB in zip(A_dict.keys(), B_dict.keys())
                             if min(len(A_dict[idA]), len(B_dict[idB])) > cycle + dispA + dispB])

        B_trace = np.array([np.log(B_dict[idB][param][cycle + dispB] / x_avg)
                             for idA, idB in zip(A_dict.keys(), B_dict.keys())
                             if min(len(A_dict[idA]), len(B_dict[idB])) > cycle + dispA + dispB])

        # WE NEED SAME SIZES FOR THE X,Y-PLANE
        if len(A_trace) != len(B_trace):
            IOError("trace sizes don't match!")

        number_of_points = len(A_trace)  # We could've used B_trace instead... Doesn't matter

        return A_trace, B_trace, number_of_points

# Don't use it anymore!
def createDirectories():

    # MAKING THE DIRECTORIES TO PUT THE PLOTS, THE NAMES WILL BE THE GENERATIONAL DISPLACEMENT UTILISED

    # SCATTER PLOTS OF THE TWO DICTIONARIES INPUTTED
    os.mkdir('WhiteBoard Analysis Plots/Scatter Plots')

    os.mkdir('WhiteBoard Analysis Plots/Scatter Plots/A Displacement')
    os.mkdir(
        'WhiteBoard Analysis Plots/Scatter Plots/A Displacement/all cycles pooling')  # number 1 on White Board
    os.mkdir(
        'WhiteBoard Analysis Plots/Scatter Plots/A Displacement/number of cycles is number of cycles pooled')  # number 2 on White Board
    os.mkdir(
        'WhiteBoard Analysis Plots/Scatter Plots/A Displacement/single cycle pooling')  # number 3 on White Board

    os.mkdir('WhiteBoard Analysis Plots/Scatter Plots/B Displacement')
    os.mkdir(
        'WhiteBoard Analysis Plots/Scatter Plots/B Displacement/all cycles pooling')  # number 1 on White Board
    os.mkdir(
        'WhiteBoard Analysis Plots/Scatter Plots/B Displacement/number of cycles is number of cycles pooled')  # number 2 on White Board
    os.mkdir(
        'WhiteBoard Analysis Plots/Scatter Plots/B Displacement/single cycle pooling')  # number 3 on White Board

    os.mkdir('WhiteBoard Analysis Plots/Scatter Plots/A itself')
    os.mkdir('WhiteBoard Analysis Plots/Scatter Plots/A itself/all cycles pooling')  # number 1 on White Board
    os.mkdir(
        'WhiteBoard Analysis Plots/Scatter Plots/A itself/number of cycles is number of cycles pooled')  # number 2 on White Board
    os.mkdir('WhiteBoard Analysis Plots/Scatter Plots/A itself/single cycle pooling')  # number 3 on White Board

    os.mkdir('WhiteBoard Analysis Plots/Scatter Plots/B itself')
    os.mkdir('WhiteBoard Analysis Plots/Scatter Plots/B itself/all cycles pooling')  # number 1 on White Board
    os.mkdir(
        'WhiteBoard Analysis Plots/Scatter Plots/B itself/number of cycles is number of cycles pooled')  # number 2 on White Board
    os.mkdir('WhiteBoard Analysis Plots/Scatter Plots/B itself/single cycle pooling')  # number 3 on White Board

    # PEARSON AND LEAST-SQUARES SLOPE AND INTERCEPT OF THE TWO DICTIONARIES INPUTTED (NUMBER OF POINTS USED
    #                                                                               WILL ALSO BE ON THE BACKGROUND)
    os.mkdir('WhiteBoard Analysis Plots/Pearson, Slope and Intercept Plots')

    os.mkdir('WhiteBoard Analysis Plots/Pearson, Slope and Intercept Plots/A Displacement')
    os.mkdir(
        'WhiteBoard Analysis Plots/Pearson, Slope and Intercept Plots/A Displacement/all cycles pooling')  # number 1 on White Board
    os.mkdir(
        'WhiteBoard Analysis Plots/Pearson, Slope and Intercept Plots/A Displacement/number of cycles is number of cycles pooled')  # number 2 on White Board
    os.mkdir(
        'WhiteBoard Analysis Plots/Pearson, Slope and Intercept Plots/A Displacement/single cycle pooling')  # number 3 on White Board

    os.mkdir('WhiteBoard Analysis Plots/Pearson, Slope and Intercept Plots/B Displacement')
    os.mkdir(
        'WhiteBoard Analysis Plots/Pearson, Slope and Intercept Plots/B Displacement/all cycles pooling')  # number 1 on White Board
    os.mkdir(
        'WhiteBoard Analysis Plots/Pearson, Slope and Intercept Plots/B Displacement/number of cycles is number of cycles pooled')  # number 2 on White Board
    os.mkdir(
        'WhiteBoard Analysis Plots/Pearson, Slope and Intercept Plots/B Displacement/single cycle pooling')  # number 3 on White Board

    os.mkdir('WhiteBoard Analysis Plots/Pearson, Slope and Intercept Plots/A itself')
    os.mkdir(
        'WhiteBoard Analysis Plots/Pearson, Slope and Intercept Plots/A itself/all cycles pooling')  # number 1 on White Board
    os.mkdir(
        'WhiteBoard Analysis Plots/Pearson, Slope and Intercept Plots/A itself/number of cycles is number of cycles pooled')  # number 2 on White Board
    os.mkdir(
        'WhiteBoard Analysis Plots/Pearson, Slope and Intercept Plots/A itself/single cycle pooling')  # number 3 on White Board

    os.mkdir('WhiteBoard Analysis Plots/Pearson, Slope and Intercept Plots/B itself')
    os.mkdir(
        'WhiteBoard Analysis Plots/Pearson, Slope and Intercept Plots/B itself/all cycles pooling')  # number 1 on White Board
    os.mkdir(
        'WhiteBoard Analysis Plots/Pearson, Slope and Intercept Plots/B itself/number of cycles is number of cycles pooled')  # number 2 on White Board
    os.mkdir(
        'WhiteBoard Analysis Plots/Pearson, Slope and Intercept Plots/B itself/single cycle pooling')  # number 3 on White Board

def scatterPlot(A_trace, B_trace, xlabel, ylabel, trace, PoolID, cycle, disp, dset):

    # HERE WE SAVE THE SCATTER PLOT

    # Decide where to save the graph

    if trace == 'A':
        direct = 'A Displacement/'
    elif trace == 'B':
        direct = 'B Displacement/'
    if trace == 'Itself':
        direct = 'Itself/'

    if PoolID == 1:
        pool = 'all cycles pooling/'
    elif PoolID == 2:
        pool = 'number of cycles is number of cycles pooled/'
    elif PoolID == 3:
        pool = 'single cycle pooling/'


    plt.scatter(A_trace, B_trace, label='number of points: '+str(len(A_trace)))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim([-.6, .6])
    plt.xlim([-.7, .7])
    plt.legend()
    plt.savefig('WhiteBoard Analysis Plots/'+dset+'/Scatter Plots/'+direct+pool+'cycle: '+str(cycle)+', disp: '+str(disp))
    plt.close()

def GenerationPlot(A_GenPlotArray, B_GenPlotArray, xlabel, trace, PoolID, disp, dset):

    if len(A_GenPlotArray) != len(B_GenPlotArray):
        IOError('not same sizes!')

    pcoeffArray = np.array([])
    slopeArray = np.array([])
    interceptArray = np.array([])

    for ind in range(len(A_GenPlotArray)):
        slope, intercept, r_value, p_value, std_err = stats.linregress(A_GenPlotArray[ind], B_GenPlotArray[ind])
        pcoeff, pval = stats.pearsonr(A_GenPlotArray[ind], B_GenPlotArray[ind])
        pcoeffArray = np.append(pcoeffArray, pcoeff)
        slopeArray = np.append(slopeArray, slope)
        interceptArray = np.append(interceptArray, intercept)

    # Decide where to save the graph

    if trace == 'A':
        direct = 'A Displacement/'
    elif trace == 'B':
        direct = 'B Displacement/'
    if trace == 'Itself':
        direct = 'Itself/'

    if PoolID == 1:
        pool = 'all cycles pooling/'
    elif PoolID == 2:
        pool = 'number of cycles is number of cycles pooled/'
    elif PoolID == 3:
        pool = 'single cycle pooling/'

    # plot the pearson coefficient over generation relations
    plt.plot(pcoeffArray, marker='.')
    plt.xlabel(xlabel) #GENERATION NUMBER
    plt.ylabel('Pearson Correlation Coefficient') # PEARSON COEFF
    plt.ylim([-1, 1])
    plt.savefig('WhiteBoard Analysis Plots/' + dset + '/Pearson, Slope and Intercept Plots/' + direct + pool +
                'Pearson Coeff ' + ', disp = '+str(disp))
    plt.close()

    # plot the slope and intercept over generation relations
    plt.plot(slopeArray, label = 'slope', marker='.')
    plt.plot(interceptArray, label='intercept', marker='.')
    plt.xlabel(xlabel)  # GENERATION NUMBER
    plt.ylabel('slope and intercept value')  # slope and intercept
    plt.ylim([-1, 1])
    plt.savefig('WhiteBoard Analysis Plots/' + dset + '/Pearson, Slope and Intercept Plots/' + direct + pool +
                'Slope and Intercept ' + ', disp = ' + str(disp))
    plt.close()

def main():
    # Import the Refined Data
    pickle_in = open("metastructdata.pickle", "rb")
    metadatastruct = pickle.load(pickle_in)
    pickle_in.close()

    # beta_array = np.array([])
    # phi_array = np.array([])
    cycle_array = np.array([])
    pop_array = np.array([])

    # createDirectories()

    # CHOOSES HOW MANY CYCLES TO DO THIS FOR
    how_many_cycles = 20

    # DECIDES THE GENERATIONAL DISPLACEMENT OF THE CELLS
    for disp in range(5): # range(10)

        # FIRST WE PLOT TO B DISPLACEMENT FOLDERS, THEN TO THE A DISPLACEMENT  AND FINALLY TO ITSELF FOLDERS
        for trace in ['A', 'B', 'Itself']:

            GenPlotArray_sis_A = []
            GenPlotArray_non_sis_A = []
            GenPlotArray_both_A = []

            GenPlotArray_sis_B = []
            GenPlotArray_non_sis_B = []
            GenPlotArray_both_B = []
            
            # FOR THE POOLING FUNCTION, MAINLY TO GET THE SCATTER PLOT
            for cycle in range(how_many_cycles):

                # USING ALL THE POOLING OPTIONS
                for PoolID in [3]: # [1,2,3]
                    # SISTERS
                    A_trace_sis, B_trace_sis, number_of_points = pooling(number=PoolID, cycle=cycle, trace=trace, A_dict=metadatastruct.A_dict_sis
                                            , B_dict=metadatastruct.B_dict_sis, x_avg=metadatastruct.x_avg, disp=disp)

                    # scatterPlot(A_trace=A_trace_sis, B_trace=B_trace_sis, xlabel=r'$ln(\frac{x^A}{x^*})$',
                    #             ylabel=r'$ln(\frac{x^B}{x^*})$', trace=trace, PoolID=PoolID, cycle=cycle, disp=disp, dset="Sisters")

                    # NON-SISTERS
                    A_trace_non_sis, B_trace_non_sis, number_of_points = pooling(number=PoolID, cycle=cycle, trace=trace,
                                                                         A_dict=metadatastruct.A_dict_non_sis
                                                                         , B_dict=metadatastruct.B_dict_non_sis,
                                                                         x_avg=metadatastruct.x_avg, disp=disp)

                    # scatterPlot(A_trace=A_trace_non_sis, B_trace=B_trace_non_sis, xlabel=r'$ln(\frac{x^A}{x^*})$',
                    #             ylabel=r'$ln(\frac{x^B}{x^*})$', trace=trace, PoolID=PoolID, cycle=cycle, disp=disp, dset="Non")

                    # CONTROL
                    A_trace_both, B_trace_both, number_of_points = pooling(number=PoolID, cycle=cycle,
                                                                                 trace=trace,
                                                                                 A_dict=metadatastruct.A_dict_both
                                                                                 , B_dict=metadatastruct.B_dict_both,
                                                                                 x_avg=metadatastruct.x_avg, disp=disp)

                    # scatterPlot(A_trace=A_trace_both, B_trace=B_trace_both, xlabel=r'$ln(\frac{x^A}{x^*})$',
                    #             ylabel=r'$ln(\frac{x^B}{x^*})$', trace=trace, PoolID=PoolID, cycle=cycle, disp=disp, dset="Control")


                    # ONLY MAKES SENSE TO USE THIS IF DISPLACEMENT IS NOT ZERO
                    if trace == 'Itself': # IN THEORY WE CAN CHOOSE A OR B TRAJECTORY, MAKES NO DIFFERENCE, ALL WE DO IS MAKE THE A_DICT THE SAME AS B_DICT
                        # SISTERS
                        A_trace_sis, B_trace_sis, number_of_points = pooling(number=PoolID, cycle=cycle, trace=trace,
                                                                             A_dict=metadatastruct.A_dict_sis
                                                                             , B_dict=metadatastruct.A_dict_sis,
                                                                             x_avg=metadatastruct.x_avg, disp=disp)

                        # scatterPlot(A_trace=A_trace_sis, B_trace=B_trace_sis, xlabel=r'$ln(\frac{x^A}{x^*})$',
                        #             ylabel=r'$ln(\frac{x^B}{x^*})$', trace=trace, PoolID=PoolID, cycle=cycle, disp=disp, dset="Sisters")

                        # NON-SISTERS
                        A_trace_non_sis, B_trace_non_sis, number_of_points = pooling(number=PoolID, cycle=cycle,
                                                                                     trace=trace,
                                                                                     A_dict=metadatastruct.A_dict_non_sis
                                                                                     ,
                                                                                     B_dict=metadatastruct.A_dict_non_sis,
                                                                                     x_avg=metadatastruct.x_avg,
                                                                                     disp=disp)

                        # scatterPlot(A_trace=A_trace_non_sis, B_trace=B_trace_non_sis, xlabel=r'$ln(\frac{x^A}{x^*})$',
                        #             ylabel=r'$ln(\frac{x^B}{x^*})$', trace=trace, PoolID=PoolID, cycle=cycle, disp=disp, dset="Non")

                        # CONTROL
                        A_trace_both, B_trace_both, number_of_points = pooling(number=PoolID, cycle=cycle,
                                                                               trace=trace,
                                                                               A_dict=metadatastruct.A_dict_both
                                                                               , B_dict=metadatastruct.A_dict_both,
                                                                               x_avg=metadatastruct.x_avg, disp=disp)

                        # scatterPlot(A_trace=A_trace_both, B_trace=B_trace_both, xlabel=r'$ln(\frac{x^A}{x^*})$',
                        #             ylabel=r'$ln(\frac{x^B}{x^*})$', trace=trace, PoolID=PoolID, cycle=cycle, disp=disp, dset="Control")

                if PoolID == 3:
                    # print(A_trace_sis)
                    GenPlotArray_sis_A.append(A_trace_sis)
                    GenPlotArray_non_sis_A.append(A_trace_non_sis)
                    GenPlotArray_both_A.append(A_trace_both)

                    GenPlotArray_sis_B.append(B_trace_sis)
                    GenPlotArray_non_sis_B.append(B_trace_non_sis)
                    GenPlotArray_both_B.append(B_trace_both)

            if PoolID == 3:
                GenerationPlot(A_GenPlotArray=GenPlotArray_sis_A, B_GenPlotArray=GenPlotArray_sis_B, xlabel='Generation'
                               , trace=trace, PoolID=3, disp=disp, dset="Sisters")
                GenerationPlot(A_GenPlotArray=GenPlotArray_non_sis_A, B_GenPlotArray=GenPlotArray_non_sis_B, xlabel='Generation'
                               , trace=trace, PoolID=3, disp=disp, dset="Non")
                GenerationPlot(A_GenPlotArray=GenPlotArray_both_A, B_GenPlotArray=GenPlotArray_both_B, xlabel='Generation'
                               , trace=trace, PoolID=3, disp=disp, dset="Control")


if __name__ == '__main__':
    main()
