#!/usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys, math
import glob
import pickle
import os
import scipy.stats as stats
import random


class SisterCellData(object):
    def __init__(self, **kwargs):
        self.__infiles = kwargs.get('infiles', [])
        self.__debugmode = kwargs.get('debugmode', False)  # return only single trajectory in iteration
        self.__sisterdata = kwargs.get('sisterdata', True)
        self.updated = kwargs.get('updated', [])

        # lists to store data internally
        self.__data = list()
        self.__dataorigin = list()
        self.__keylist = list()

        # load first sheet of each Excel-File, fill internal data structure
        for filename in self.__infiles:
            try:
                tmpdata = pd.read_excel(filename)
            except:
                continue
            self.__data.append(tmpdata)
            self.__dataorigin.append(filename)
            for k in tmpdata.keys():
                if not str(k) in self.__keylist:
                    self.__keylist.append(str(k))

        # there's no point in not having data ...
        # ... or something went wrong. rather stop here
        if not len(self) > 0:
            raise IOError('no data loaded')

    def LMSQ(self, x, y, cov=True):
        """
        Least Mean Squares estimator
        for a linear interpolation between x,y: y ~ a + b x
        Returns also covariance matrix for the estimated parameters a,b if 'cov' is True

        Algorithm:
        A = ( 1  ... 1  )
            ( x1 ... xn )

        y = ( y1 ... yn ).T

        p = (a b).T

        E[p]     = inv(A.T * A) * A.T * y
        Cov[p,p] = sigma2 * inv(A.T * A)
        sigma2   = E[ ( y - A*E[p] )^2 ]
        """

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

    def otsu(self, x):
        """
        Otsu's method
        First described in IEEE TRANSACTIONS ON SYSTEMS, MAN, AND CYBERNETICS (1979)
        Usually used to binarize photos into black/white
        Here modified to use bins with single entries for each measurement
        """
        sx = np.sort(x)
        lx = len(x)

        w = np.arange(lx, dtype=np.float) / lx
        m = np.array([np.mean(sx[:k]) * w[k] if k > 0 else 0 for k in range(lx)])
        mT = np.mean(sx)

        sB = np.array(
            [(mT * w[k] - m[k]) ** 2 / (w[k] * (1. - w[k])) if w[k] * (1. - w[k]) > 0 else 0 for k in range(lx)])
        idx = np.argmax(sB)

        return sx[idx]

    def CellDivisionTrajectory(self, dataID, meth, discretize_by='length', sisterdata=None, additional_columns=[]):
        """
        Transform measured time series data into data for each generation:
        (1) Find cell division events as a large enough drop in the measured value given by 'discretize_by'
        (2) Compute various observables from these generations of cells
        (3) Returns two pandas dataframes for each of the discretized trajectories
        """

        if not discretize_by in self.keylist_stripped:  raise KeyError('key not found')
        # not sure if this is needed
        if sisterdata is None:  sisterdata = self.__sisterdata
        if sisterdata:
            keysuffix = ['A', 'B']
        else:
            keysuffix = ['']

        # return two pandas-dataframes for the two trajectories usually contained in one file as two elements of a list
        # as the two sisters do not need to have the same number of cell divisions, a single dataframe might cause problems with variably lengthed trajectories
        ret = list()

        alldiffdata = np.concatenate([np.diff(self.__data[dataID][discretize_by + ks]) for ks in keysuffix])

        for ks in keysuffix:
            # use 'discretize_by' as the column name that should serve as measurement that indicates the division
            # in our data, both 'length' and 'cellsize' should be OK.
            # get list of changes between measurements. Usually this distribution is bimodal:
            #  * scattering around small values for normal data-points
            #  * scattering around 'large' negative values for cell divisions
            diffdata = np.diff(self.__data[dataID][discretize_by + ks])

            if meth == 'otsu':
                # estimate a threshold from the data between the two peaks in the bimodal distribution with Otsu's method, then get indices of these transitions
                index_div = np.where(diffdata < self.otsu(alldiffdata))[0].flatten()
                # INPUTTING MY OWN CODE! :/
                if len(index_div) < 2:
                    ret_ks = []
                    ret.append(pd.DataFrame(ret_ks))
                    continue
            elif meth == 'threshold':
                index_div = np.where(diffdata < -1)[0].flatten()
                for ind1, ind2 in zip(index_div[:-1], index_div[1:]):
                    if ind2 - ind1 <= 2:
                        index_div = np.delete(index_div, np.where(index_div == ind2))
                # WE DISCARD THE FIRST AND LAST CYCLE BECAUSE THEY ARE MOST LIKELY NOT COMPLETE
                # THESE ARE INDICES AS WELL!
                start_times = [x + 1 for x in index_div]
                end_times = [x - 1 for x in index_div]
                start_times.append(0)
                start_times.sort()
                del start_times[-1]
                end_times.sort()
                if self.__data[dataID][discretize_by + ks].index[-1] in start_times:
                    start_times = start_times.remove(self.__data[dataID][discretize_by + ks].index[-1])
                if 0 in end_times:
                    end_times = end_times.remove(0)
                for start, end in zip(start_times, end_times):
                    if start >= end:
                        print('start', start, 'end', end, "didn't work")

            # 'double jump': division spans two (or more) time points
            # todo: need to implement some lines of code that detect if a division spans multiple time points, and corrects the algorithm below properly
            # WHAT ABOUT WHEN THE DIFFERENCE IS NOT 1 BUT MORE OR LESS THAN 1 AND WHAT IS THE THRESHOLD FOR DECIDING IF IT JUMPED????
            dj = sum(np.where(np.diff(index_div) == 1)[0].flatten())

            # timepoint of division is assumed to be the average before and after the drop in signal
            time_div = np.concatenate([[1.5 * self.__data[dataID]['time' + ks][0] - 0.5 * self.__data[dataID]
            ['time' + ks][1]], 0.5 * np.array(self.__data[dataID]['time' + ks][index_div + 1]) + 0.5 * np.array
                                       (self.__data[dataID]['time' + ks][index_div])])

            # store results in dictionary, which can be easier transformed into pandas-dataframe
            # values computed for each cell cycle are
            #  * generation time
            #  * size at birth
            #  * size at division
            #  * Least Mean Squares Estimator for the (exponential) growth rate over the full division cycle
            ret_ks = dict()
            # ret_ks['divisiontimes']           = np.array(time_div)
            ret_ks['generationtime'] = np.diff(time_div)
            ret_ks[discretize_by + '_birth'] = np.concatenate([[self.__data[dataID][discretize_by + ks][0]], np.array(
                self.__data[dataID][discretize_by + ks][index_div + 1])[:-1]])
            ret_ks[discretize_by + '_final'] = np.array(self.__data[dataID][discretize_by + ks][index_div])

            # GROWTH LENGTH IS THE INDIVIDUAL ALPHA
            ret_ks['growth_' + discretize_by] = np.concatenate([
                [self.LMSQ(self.__data[dataID]['time' + ks][:index_div[0]],
                           np.log(self.__data[dataID][discretize_by + ks][:index_div[0]]), cov=False)[1]],
                # first generation
                np.array([self.LMSQ(
                    self.__data[dataID]['time' + ks][index_div[i] + 1:index_div[i + 1] + 1],  # x-values
                    np.log(self.__data[dataID][discretize_by + ks][index_div[i] + 1:index_div[i + 1] + 1]),  # y-values
                    cov=False)[1] for i in range(len(index_div) - 1)])])


            # if additional data is requested, return values from birth and division
            if len(additional_columns) > 0:
                for ac in additional_columns:
                    if ac in self.keylist_stripped:
                        ret_ks[ac + '_birth'] = np.concatenate([[self.__data[dataID][ac + ks][0]],
                                                                np.array(self.__data[dataID][ac + ks][index_div + 1])[
                                                                :-1]])
                        ret_ks[ac + '_final'] = np.array(self.__data[dataID][ac + ks][index_div])

                        print('additional data was required OLD')

            # GET THE DIVISION RATIOS
            div_rats = []
            for final, next_beg in zip(ret_ks[discretize_by + '_final'][0:-1], ret_ks[discretize_by + '_birth'][1:]):
                div_rats.append(next_beg / final)
            div_rats.append(self.__data[dataID][discretize_by + ks][index_div[-1] + 1]/ret_ks[discretize_by + '_final'][-1])
            ret_ks['division_ratios__f_n'] = div_rats

            # we have everything, now make a dataframe
            ret.append(pd.DataFrame(ret_ks))
        return ret




    ''' Added on October 22, 2019 because of the inconsistencies we saw with the growth rate and the birth length '''
    def CellDivisionTrajectoryUPDATED(self, dataID, meth, discretize_by='length', sisterdata=None, additional_columns=[]):
        """
        Transform measured time series data into data for each generation:
        (1) Find cell division events as a large enough drop in the measured value given by 'discretize_by'
        (2) Compute various observables from these generations of cells
        (3) Returns two pandas dataframes for each of the discretized trajectories
        """

        if not discretize_by in self.keylist_stripped:  raise KeyError('key not found')
        # not sure if this is needed
        if sisterdata is None:  sisterdata = self.__sisterdata
        if sisterdata:
            keysuffix = ['A', 'B']
        else:
            keysuffix = ['']

        # return two pandas-dataframes for the two trajectories usually contained in one file as two elements of a list
        # as the two sisters do not need to have the same number of cell divisions, a single dataframe might cause problems with variably lengthed trajectories
        ret = list()

        alldiffdata = np.concatenate([np.diff(self.__data[dataID][discretize_by + ks]) for ks in keysuffix])

        for ks in keysuffix:
            # use 'discretize_by' as the column name that should serve as measurement that indicates the division
            # in our data, both 'length' and 'cellsize' should be OK.
            # get list of changes between measurements. Usually this distribution is bimodal:
            #  * scattering around small values for normal data-points
            #  * scattering around 'large' negative values for cell divisions
            diffdata = np.diff(self.__data[dataID][discretize_by + ks])

            if meth == 'otsu':
                # estimate a threshold from the data between the two peaks in the bimodal distribution with Otsu's method, then get indices of these transitions
                index_div = np.where(diffdata < self.otsu(alldiffdata))[0].flatten()
                # INPUTTING MY OWN CODE! :/
                if len(index_div) < 2:
                    ret_ks = []
                    ret.append(pd.DataFrame(ret_ks))
                    continue
            elif meth == 'threshold':
                index_div = np.where(diffdata < -1)[0].flatten()
                for ind1, ind2 in zip(index_div[:-1], index_div[1:]):
                    if ind2 - ind1 <= 2:
                        index_div = np.delete(index_div, np.where(index_div == ind2))
                # WE DISCARD THE FIRST AND LAST CYCLE BECAUSE THEY ARE MOST LIKELY NOT COMPLETE
                # THESE ARE INDICES AS WELL!
                start_times = [x + 1 for x in index_div]
                end_times = [x - 1 for x in index_div]
                start_times.append(0)
                start_times.sort()
                del start_times[-1]
                end_times.sort()
                if self.__data[dataID][discretize_by + ks].index[-1] in start_times:
                    start_times = start_times.remove(self.__data[dataID][discretize_by + ks].index[-1])
                if 0 in end_times:
                    end_times = end_times.remove(0)
                for start, end in zip(start_times, end_times):
                    if start >= end:
                        print('start', start, 'end', end, "didn't work")

            ''' this is the part that we changed and added, basically we do a least squares to get the length birth and growth length '''

            # Now we figure out where the division times occur, also found in SisterMachineDataProcessing, ie. the "source code"
            diffdata = np.diff(self.__data[dataID]['length' + ks])

            index_div = np.where(diffdata < -1)[0].flatten()
            for ind1, ind2 in zip(index_div[:-1], index_div[1:]):
                if ind2 - ind1 <= 2:
                    index_div = np.delete(index_div, np.where(index_div == ind2))
            # WE DISCARD THE LAST CYCLE BECAUSE IT MOST LIKELY WON'T BE COMPLETE
            # THESE ARE INDICES AS WELL!
            start_times = [x + 1 for x in index_div]
            end_times = [x - 1 for x in index_div]
            start_times.append(0)
            start_times.sort()
            del start_times[-1]
            end_times.sort()
            if self.__data[dataID]['length'+ks].index[-1] in start_times:
                start_times = start_times.remove(self.__data[dataID]['length'+ks].index[-1])
            if 0 in end_times:
                end_times = end_times.remove(0)
            for start, end in zip(start_times, end_times):
                if start >= end:
                    print('start', start, 'end', end, "didn't work")

            dj = sum(np.where(np.diff(index_div) == 1)[0].flatten())

            # timepoint of division is assumed to be the average before and after the drop in signal
            time_div = np.concatenate([[1.5 * self.__data[dataID]['time'+ks][0] - 0.5 * self.__data[dataID]
            ['time'+ks][1]], 0.5 * np.array(self.__data[dataID]['time'+ks][index_div + 1]) + 0.5 * np.array
                                       (self.__data[dataID]['time'+ks][index_div])])

            # store results in dictionary, which can be easier transformed into pandas-dataframe
            # values computed for each cell cycle are
            #  * generation time
            #  * size at birth
            #  * size at division
            #  * Least Mean Squares Estimator for the (exponential) growth rate over the full division cycle
            ret_ks = dict()
            # ret_ks['divisiontimes']           = np.array(time_div)
            ret_ks['generationtime'] = np.diff(time_div) - .05

            
            domain = np.linspace(0, (len(self.__data[dataID]['time'+ks][:index_div[0]]) - 1) * .05, num=len(self.__data[dataID]['time'+ks][:index_div[
                0]]))
            first = np.linalg.lstsq(np.concatenate([domain[:, np.newaxis], np.ones_like(domain)[:, np.newaxis]], axis=1),
                                np.log(self.__data[dataID][discretize_by + ks][:index_div[0]]))[0]
            rest = np.array([np.linalg.lstsq(np.concatenate([
                np.linspace(0, (len(self.__data[dataID]['time' + ks][index_div[i] + 1:index_div[i + 1] + 1]) - 1) * .05,
                            num=len(self.__data[dataID]['time' + ks][index_div[i] + 1:index_div[i + 1] + 1]))[:, np.newaxis], np.ones_like(
                    np.linspace(0, (len(self.__data[dataID]['time' + ks][index_div[i] + 1:index_div[i + 1] + 1]) - 1) * .05,
                                num=len(self.__data[dataID]['time' + ks][index_div[i] + 1:index_div[i + 1] + 1]))
                )[:, np.newaxis]], axis=1),  # x-values
                np.log(self.__data[dataID][discretize_by + ks][index_div[i] + 1:index_div[i + 1] + 1])  # y-values
            )[0] for i in range(len(index_div) - 1)])

            line_details = np.concatenate([np.array(first, ndmin=2), rest], axis=0)
            
            ret_ks[discretize_by + '_birth'] = np.exp(line_details[:,1])

            ret_ks[discretize_by + '_final'] = np.array(self.__data[dataID][discretize_by + ks][index_div])

            ret_ks['growth_' + discretize_by] = line_details[:, 0]


            # if additional data is requested, return values from birth and division
            if len(additional_columns) > 0:
                for ac in additional_columns:
                    if ac in self.keylist_stripped:
                        ret_ks[ac + '_birth'] = np.concatenate([[self.__data[dataID][ac + ks][0]],
                                                                np.array(self.__data[dataID][ac + ks][index_div + 1])[
                                                                :-1]])
                        ret_ks[ac + '_final'] = np.array(self.__data[dataID][ac + ks][index_div])

                        print('additional data was required')

            # GET THE DIVISION RATIOS
            div_rats = []
            for final, next_beg in zip(ret_ks[discretize_by + '_final'][0:-1], ret_ks[discretize_by + '_birth'][1:]):
                div_rats.append(next_beg / final)
            div_rats.append(self.__data[dataID][discretize_by + ks][index_div[-1] + 1]/ret_ks[discretize_by + '_final'][-1])
            ret_ks['division_ratios__f_n'] = div_rats

            # we have everything, now make a dataframe
            ret.append(pd.DataFrame(ret_ks))
        return ret




    # PUT IN BY ALE...
    def DictionaryOfSisterTraces(self, dataID, meth, discretize_by, dictionary, sis):
        if self.updated:
            TrajA, TrajB = self.CellDivisionTrajectoryUPDATED(dataID, meth, discretize_by=discretize_by, sisterdata=None,
                                                              additional_columns=[])
            print('self_self.updated is true', dataID, sis)
        else:
            TrajA, TrajB = self.CellDivisionTrajectory(dataID, meth, discretize_by=discretize_by, sisterdata=None,
                                                              additional_columns=[])
            print('self_self.updated is NOT true', dataID, sis)

        if sis == 'Yes':
            keyA = 'Sister_Trace_A_' + str(dataID)
            keyB = 'Sister_Trace_B_' + str(dataID)
        elif sis == 'No':
            keyA = 'Non-sister_Trace_A_' + str(dataID)
            keyB = 'Non-sister_Trace_B_' + str(dataID)
        elif sis == 'Both':
            keyA = 'Both_Trace_A_' + str(dataID)
            keyB = 'Both_Trace_B_' + str(dataID)
        dictionary.update({keyA: TrajA, keyB: TrajB})
        # print('length of dataframe A:',len(TrajA),'length of dataframe B:', len(TrajB))
        return dictionary

    def CorrelateParamsFFT(self, trajA, trajB, normalize=False, maxlen=None, enforcelen=False):
        acf_array1 = dict()
        for key in trajA.keys():
            ml = np.nanmin([len(trajB[key]), len(trajA[key])])
            xp = trajA[key][:ml] - np.mean(trajB[key][:ml])
            f = np.fft.fft(xp)
            p = np.array([np.real(v) ** 2 + np.imag(v) ** 2 for v in f])
            pi = np.fft.ifft(p)
            acf1 = np.real(pi)[:int(ml / 2)]
            if normalize: acf1 *= 1. / np.sum(xp ** 2)
            if enforcelen and not maxlen is None:
                if len(acf1) < maxlen:
                    acf1 = np.concatenate([acf1, np.zeros(maxlen - len(acf1))])
            acf_array1[key] = acf1

        acf_array2 = dict()
        for key in trajA.keys():
            ml = np.nanmin([len(trajB[key]), len(trajA[key])])
            xp = trajB[key][:ml] - np.mean(trajA[key][:ml])
            f = np.fft.fft(xp)
            p = np.array([np.real(v) ** 2 + np.imag(v) ** 2 for v in f])
            pi = np.fft.ifft(p)
            acf2 = np.real(pi)[:int(ml / 2)]
            if normalize: acf2 *= 1. / np.sum(xp ** 2)
            if enforcelen and not maxlen is None:
                if len(acf2) < maxlen:
                    acf2 = np.concatenate([acf2, np.zeros(maxlen - len(acf2))])
            acf_array2[key] = acf2

        return acf_array1, acf_array2

    def AutocorrelationFFT(self, dataID, normalize=False, maxlen=None, enforcelen=False):
        """
        Compute the autocorrelation of the signal, based on the properties of the
        power spectral density of the signal.
        """
        if self.updated:
            trajA, trajB = self.CellDivisionTrajectoryUPDATED(dataID, meth='threshold', discretize_by='length', sisterdata=None,
                                                              additional_columns=[])
        else:
            trajA, trajB = self.CellDivisionTrajectory(dataID, meth='threshold', discretize_by='length', sisterdata=None, additional_columns=[])

        acfA = dict()
        for key in trajA.keys():
            ml = np.nanmin([maxlen, len(trajA[key])])
            xp = trajA[key][:ml] - np.mean(trajA[key][:ml])
            f = np.fft.fft(xp)
            p = np.array([np.real(v) ** 2 + np.imag(v) ** 2 for v in f])
            pi = np.fft.ifft(p)
            acf = np.real(pi)[:int(ml / 2)]
            if normalize: acf *= 1. / np.sum(xp ** 2)
            if enforcelen and not maxlen is None:
                if len(acf) < maxlen:
                    acf = np.concatenate([acf, np.zeros(maxlen - len(acf))])
            acfA[key] = acf

        acfB = dict()
        for key in trajB.keys():
            ml = np.min([maxlen, len(trajB[key])])
            xp = trajB[key][:ml] - np.mean(trajB[key][:ml])
            f = np.fft.fft(xp)
            p = np.array([np.real(v) ** 2 + np.imag(v) ** 2 for v in f])
            pi = np.fft.ifft(p)
            acf = np.real(pi)[:int(ml / 2)]
            if normalize: acf *= 1. / np.sum(xp ** 2)
            if enforcelen and not maxlen is None:
                if len(acf) < maxlen:
                    acf = np.concatenate([acf, np.zeros(maxlen - len(acf))])
            acfB[key] = acf

        return acfA, acfB

    def AutocorrelationFFT1(self, trajA, trajB, normalize=False, maxlen=None, enforcelen=False):
        """
        Compute the autocorrelation of the signal, based on the properties of the
        power spectral density of the signal.
        """

        acfA = dict()
        for key in trajA.keys():
            ml = np.nanmin([maxlen, len(trajA[key])])
            xp = trajA[key][:ml] - np.mean(trajA[key][:ml])
            f = np.fft.fft(xp)
            p = np.array([np.real(v) ** 2 + np.imag(v) ** 2 for v in f])
            pi = np.fft.ifft(p)
            acf = np.real(pi)[:int(ml / 2)]
            if normalize: acf *= 1. / np.sum(xp ** 2)
            if enforcelen and not maxlen is None:
                if len(acf) < maxlen:
                    acf = np.concatenate([acf, np.zeros(maxlen - len(acf))])
            acfA[key] = acf

        acfB = dict()
        for key in trajB.keys():
            ml = np.min([maxlen, len(trajB[key])])
            xp = trajB[key][:ml] - np.mean(trajB[key][:ml])
            f = np.fft.fft(xp)
            p = np.array([np.real(v) ** 2 + np.imag(v) ** 2 for v in f])
            pi = np.fft.ifft(p)
            acf = np.real(pi)[:int(ml / 2)]
            if normalize: acf *= 1. / np.sum(xp ** 2)
            if enforcelen and not maxlen is None:
                if len(acf) < maxlen:
                    acf = np.concatenate([acf, np.zeros(maxlen - len(acf))])
            acfB[key] = acf

        return acfA, acfB

    def LineageCorrelation1(self, trajA, trajB, maxlen=20):
        """
        Compute the correlation between the two lineages A and B of sistercells,
        specifically <X(A,t) X(B,t)> - <X(A,t)> <X(B,t)>,
        where X indicates the data in the discretized trajectory
        """
        corrmatrix_sumAB = dict()
        corrmatrix_sumA = dict()
        corrmatrix_sumB = dict()
        corrmatrix_count = dict()

        for corrkey in trajA.keys():
            if not corrkey in corrmatrix_sumAB.keys():
                corrmatrix_sumAB[corrkey] = np.zeros((maxlen, maxlen), dtype=np.float)
                corrmatrix_sumA[corrkey] = np.zeros((maxlen, maxlen), dtype=np.float)
                corrmatrix_sumB[corrkey] = np.zeros((maxlen, maxlen), dtype=np.float)
                corrmatrix_count[corrkey] = np.zeros((maxlen, maxlen), dtype=np.float)

            mlA = min(maxlen, len(trajA))
            mlB = min(maxlen, len(trajB))

            corrmatrix_sumAB[corrkey][:mlA, :mlB] += np.outer(trajA[corrkey][:mlA], trajB[corrkey][:mlB])
            corrmatrix_sumB[corrkey][:mlA, :mlB] += np.repeat([trajB[corrkey][:mlB]], mlA, axis=0)
            corrmatrix_sumA[corrkey][:mlA, :mlB] += np.repeat([trajA[corrkey][:mlA]], mlB, axis=0).T
            corrmatrix_count[corrkey][:mlA, :mlB] += np.ones((mlA, mlB))

        return corrmatrix_sumAB, corrmatrix_sumA, corrmatrix_sumB, corrmatrix_count

    # access single dataframe by its ID
    def __getitem__(self, key):
        return self.__data[key]

    # should not allow accessing internal variables in other ways than funneling through this here
    def __getattr__(self, key):
        if key == "filenames":
            return self.__dataorigin
        elif key == "keylist":
            return self.__keylist
        elif key == "keylist_stripped":
            return list(set([s.strip('AB ') for s in self.__keylist]))
        elif key == 'timestep':
            ts = self[0]['timeA'][1] - self[0]['timeA'][0]
            for dataID in range(len(self)):
                dt = np.diff(self.__data[dataID]['timeA'])
                if ts > np.min(dt): ts = np.min(dt)
            return ts

    # convenience
    def __len__(self):
        return len(self.__data)

    # data will be processes as loop over the class instance
    # 'debugmode' only returns a single item (the first one)
    def __iter__(self):
        dataIDs = np.arange(len(self), dtype=int)
        if self.__debugmode:
            # yield only first element in debugmode to check analysis on this trajectory
            yield 0, self.__dataorigin[0], self.__data[0]
        else:
            for dataID, origin, data in zip(dataIDs, self.__dataorigin, self.__data):
                yield dataID, origin, data

    def __getstate__(self):
        return vars(self)

    def __setstate__(self, state):
        return vars(self).update(state)


class CycleData(SisterCellData):
    def __init__(self, **kwargs):
        infiles_sisters = kwargs.get('infiles_sisters', [])
        infiles_nonsisters = kwargs.get('infiles_nonsisters', [])
        discretize_by = kwargs.get('discretize_by', [])
        updated = kwargs.get('updated', [])

        def xstar1(all_dict, discretize_by):
            # Sisters or Nonsisters are the inputs most likely
            # discretize_by is either length, flourescence or cellarea
            # traj is either 'A' or 'B'
            # if ensemble is to take from both, make both = True
            if discretize_by == 'length':
                return np.mean([np.mean(all_dict[key]['length_birth']) for key in all_dict.keys()])
            elif discretize_by == 'cellarea':
                return np.mean([np.mean(all_dict[key]['length_birth']) for key in all_dict.keys()])
            elif discretize_by == 'flourescence':
                return np.mean([np.mean(all_dict[key]['length_birth']) for key in all_dict.keys()])

        Sisters1 = SisterCellData(infiles=infiles_sisters, debugmode=False, updated=updated)
        Nonsisters1 = SisterCellData(infiles=infiles_nonsisters, debugmode=False, updated=updated)

        self.Sisters = Sisters1
        self.Nonsisters = Nonsisters1
        # self.Both = Both1

        # CREATES THE DICT_WITH_ALL_SISTER_TRACES AND NEXT ONE NON_SISTERS
        meth = 'threshold'
        self.dict_with_all_sister_traces = {'none': 1}
        for dataID in range(0, len(infiles_sisters), 1):

            self.dict_with_all_sister_traces = self.Sisters.DictionaryOfSisterTraces(dataID, meth,
                                                                                     discretize_by=discretize_by,
                                                                                     dictionary=self.dict_with_all_sister_traces,
                                                                                     sis='Yes')
            if dataID == 0:
                del self.dict_with_all_sister_traces['none']

        self.dict_with_all_non_sister_traces = {'none': 1}
        for dataID in range(0, len(infiles_nonsisters) - 1, 1):
            self.dict_with_all_non_sister_traces = self.Nonsisters.DictionaryOfSisterTraces(dataID, meth,
                                                                                            discretize_by=discretize_by,
                                                                                            dictionary=self.dict_with_all_non_sister_traces,
                                                                                            sis='No')
            if dataID == 0:
                del self.dict_with_all_non_sister_traces['none']

        #######################################################################################
        # FORMAT NAME: 'Sister_Trace_A_' + str(dataID) OR 'Non-sister_Trace_A_' + str(dataID) #
        #######################################################################################

        self.dict_all = dict()
        self.dict_all.update(self.dict_with_all_sister_traces)
        self.dict_all.update(self.dict_with_all_non_sister_traces)

        self.x_avg = xstar1(self.dict_all, discretize_by)

        self.A_dict_sis = dict(zip([k for k, v in self.dict_with_all_sister_traces.items() if 'Sister_Trace_A_' in k],
                                   [v for k, v in self.dict_with_all_sister_traces.items() if 'Sister_Trace_A_' in k]))
        self.A_dict_non_sis = dict(
            zip([k for k, v in self.dict_with_all_non_sister_traces.items() if 'Non-sister_Trace_A_' in k],
                [v for k, v in self.dict_with_all_non_sister_traces.items() if
                 'Non-sister_Trace_A_' in k]))
        self.B_dict_sis = dict(zip([k for k, v in self.dict_with_all_sister_traces.items() if 'Sister_Trace_B_' in k],
                                   [v for k, v in self.dict_with_all_sister_traces.items() if 'Sister_Trace_B_' in k]))
        self.B_dict_non_sis = dict(
            zip([k for k, v in self.dict_with_all_non_sister_traces.items() if 'Non-sister_Trace_B_' in k],
                [v for k, v in self.dict_with_all_non_sister_traces.items() if
                 'Non-sister_Trace_B_' in k]))

        random.seed(55)
        # DECIDE WHAT ARE THE IDS LEFTOVER FROM 40 GENS.
        dsetA = self.A_dict_sis
        dsetB = self.B_dict_sis
        dset1 = self.A_dict_non_sis
        dset2 = self.B_dict_non_sis
        track_A = []
        track_B = []
        for how_many_gens in range(min(max([len(dsetA[keyA]) for keyA in dsetA.keys()]),
                                       max([len(dsetB[keyA]) for keyA in dsetB.keys()]),
                                       max([len(dset1[keyA]) for keyA in dset1.keys()]),
                                       max([len(dset2[keyA]) for keyA in dset2.keys()])
                                       ), 0, -1):
            print('how_many_gens: ', how_many_gens)
            pop_IDs = np.array([[keyA, keyB] for keyA, keyB in zip(dsetA.keys(), dsetB.keys()) if
                                min(len(dsetA[keyA]['generationtime']),
                                    len(dsetB[keyB]['generationtime'])) > how_many_gens])
            pop_IDs1 = np.array([[keyA, keyB] for keyA, keyB in zip(dset1.keys(), dset2.keys()) if
                                 min(len(dset1[keyA]['generationtime']),
                                     len(dset2[keyB]['generationtime'])) > how_many_gens])
            pull_out = min(len([pop_IDs[num][0] for num in range(len(pop_IDs)) if pop_IDs[num][0] not in track_A]),
                           len([pop_IDs1[num][0] for num in range(len(pop_IDs1)) if pop_IDs1[num][0] not in track_B]))
            print('pop_IDs: ', pop_IDs)
            print('pop_IDs1: ', pop_IDs1)
            if [pop_IDs[num][0] for num in range(len(pop_IDs)) if pop_IDs[num][0] not in track_A] != []:
                A_samples_sis = random.sample([pop_IDs[num][0] for num in range(len(pop_IDs)) if pop_IDs[num][0] not in
                                               track_A], k=pull_out)
                track_A.extend(A_samples_sis)
                print('track_A: ', track_A)
            if [pop_IDs1[num][0] for num in range(len(pop_IDs1)) if pop_IDs1[num][0] not in track_B] != []:
                A_samples_non_sis = random.sample([pop_IDs1[num][0] for num in range(len(pop_IDs1)) if pop_IDs1[num][0]
                                                   not in track_B], k=pull_out)
                track_B.extend(A_samples_non_sis)
                print('track_B: ', track_B)

        self.A_dict_both = dict(zip(track_A, [self.dict_all[ky] for ky in track_A]))
        self.B_dict_both = dict(zip(track_B, [self.dict_all[ky] for ky in track_B]))

        self.dict_with_all_both_traces = dict()

        self.dict_with_all_both_traces.update(self.A_dict_both)
        self.dict_with_all_both_traces.update(self.B_dict_both)

        sis_A_control = list(self.A_dict_both.keys())
        non_sis_B_control = list(self.B_dict_both.keys())
        list_of_Control_IDs_1 = list()
        for id in range(len(sis_A_control)):
            if sis_A_control[id][-3] == '_':
                list_of_Control_IDs_1.append(int(sis_A_control[id][-2:]))
            elif sis_A_control[id][-2] == '_':
                list_of_Control_IDs_1.append(int(sis_A_control[id][-1:]))
            else:
                list_of_Control_IDs_1.append(int(sis_A_control[id][-3:]))
        list_of_Control_IDs_2 = list()
        for id in range(len(non_sis_B_control)):
            if non_sis_B_control[id][-3] == '_':
                list_of_Control_IDs_2.append(int(non_sis_B_control[id][-2:]))
            elif non_sis_B_control[id][-2] == '_':
                list_of_Control_IDs_2.append(int(non_sis_B_control[id][-1:]))
            elif non_sis_B_control[id][-4] == '_':
                list_of_Control_IDs_2.append(int(non_sis_B_control[id][-3:]))
        print(list_of_Control_IDs_1, list_of_Control_IDs_2)

        if len(list_of_Control_IDs_1) != len(list_of_Control_IDs_2):
            IOError("Not the same number of samples in both A and B both trajectories")
        IDs_for_Control = range(len(list_of_Control_IDs_1))

        # ALWAYS REMEMBER THAT THE A TRACES ARE TAKEN FROM SISTERS A and B TRACES FROM NON-SISTER B
        self.Control = [list_of_Control_IDs_1, list_of_Control_IDs_2]

        # self.Control.update(zip(IDs_for_Control, [pd.concat([self.Sisters[id1].iloc[:,:5],
        #                 self.Nonsisters[id2].iloc[:, :5].rename(columns={"timeA": "timeB", "lengthA":"lengthB",
        #                 "cellareaA": "cellareaB", 'meanfluorescenceA': 'meanfluorescenceB','fluorescenceA':
        #                 'fluorescenceB'})], ignore_index=False, axis=1) for id1, id2 in zip(list_of_Control_IDs_1,
        #                                                                                   list_of_Control_IDs_2)]))
        #
        # print(len(self.Control[0]), len(self.Sisters[list_of_Control_IDs_1[0]].iloc[:,:5]), len(self.Nonsisters[list_of_Control_IDs_2[0]].iloc[:, :5].rename(index=str, columns={"timeA": "timeB", "lengthA":"lengthB",
        #                 "cellareaA": "cellareaB", 'meanfluorescenceA': 'meanfluorescenceB','fluorescenceA':
        #                 'fluorescenceB'})))
        # print(self.Control[0])

        """
        for cycle in range(min(max([len(self.A_dict_sis[keyA]['generationtime']) for keyA in self.A_dict_sis.keys()]),
                          max([len(self.B_dict_sis[keyB]['generationtime']) for keyB in self.B_dict_sis.keys()]))):
            # NUMBER OF PAIRS THAT SURVIVED cycle CYCLES
            sis_pop_dict_cycle = [[keyA,keyB] for keyA, keyB in zip(self.A_dict_sis.keys(), self.B_dict_sis) if
             min(len(self.A_dict_sis[keyA]['generationtime'], len(self.B_dict_sis[keyB]['generationtime']))) > cycle]
            self.sis_pop_dict.update({str(cycle) : sis_pop_dict_cycle})
        """

    # TRANSLATE, FIGURE OUT WHAT FOR
    def PearsonStats(self):

        sis_and_non_sis_pooled_data_correlations = dict()
        for key in self.sis_pooled_data_dict['columns_pooled_over_cycles_A_sis'].keys():
            sis_and_non_sis_pooled_data_correlations \
                .update({'Pearson_corr_coeff_sisVSnonsis_' + key: [np.corrcoef(
                np.array(self.sis_pooled_data_dict['columns_pooled_over_cycles_A_sis'][key][cycle_num]),
                np.array(self.non_sis_pooled_data_dict[key][cycle_num])) for cycle_num in
                range(min(len(self.sis_pooled_data_dict['columns_pooled_over_cycles_A_sis'][key]),
                          len(self.non_sis_pooled_data_dict['columns_pooled_over_cycles_A_non_sis'][key])))]})

        return sis_and_non_sis_pooled_data_correlations

    # EASY, TRANSLATE
    def IndParamNorm(self, trajA, trajB):
        minlen = min(len(trajA), len(trajB))
        diff = np.abs([trajA[param][:minlen] - trajB[param][:minlen] for param in trajA.keys()])
        l1 = [np.sum([diff[param][ind] for param in np.arange(len(diff))]) for ind in np.arange(len(diff[0]))]
        l2 = [np.linalg.norm([diff[param][ind] for param in np.arange(len(diff))]) for ind in np.arange(len(diff[0]))]
        # frobenius norm, 2-norm

        return l1, l2

    # HOLY SHIT TRANSLATE WELL, MAYBE DIVIDE INTO DIFFERENT ONES
    def PearsonOverIndividual(self, separatebysister=True, significance_thresh=None, comparelineages=None,
                              dsets=None, graph=None, seperatebydsets_only=None, cloud=None, time_graph=None):

        """
        pvals_s/n/b_pairs is a dictionary that contains the keys taken from the A/B trajectory keys separated
        by a space and a value of that pair's individual 2-tailed pearson coefficient and 2-tailed p-value
        for each param, ie. pvals_s/n/b_pairs[param] --> [pcoeff, 2-tailed p-value]
        """

        if significance_thresh == None:
            significance_thresh = 0.05

        # Separate the points onto which we take the Pearson Coefficient
        if separatebysister == True and seperatebydsets_only == False:

            if 's' in dsets:

                pval_sis = dict()
                # Create the dictionaries with the pval stats
                for keyA, keyB in zip(self.A_dict_sis.keys(), self.B_dict_sis.keys()):
                    p_val_param = dict()
                    for param in self.A_dict_sis[keyA].keys():
                        sample1 = self.A_dict_sis[keyA][param]
                        sample2 = self.B_dict_sis[keyB][param]
                        minlen = min(len(sample1), len(sample2))
                        p_val = stats.pearsonr(x=sample1[:minlen], y=sample2[:minlen])
                        p_val_param.update({param: p_val})
                    pval_sis.update({str(keyA) + ' ' + str(keyB): p_val_param})

                if graph == True:

                    # Plot the pearson correlations
                    for param in pval_sis[list(pval_sis.keys())[0]]:
                        plt.figure()
                        dist_of_pairs = [pval_sis[key][param][0] for key in pval_sis.keys()]

                        plt.hist(x=dist_of_pairs, range=[-.6, 1],
                                 label=r'$\mu=$' + '{:.2e}'.format(np.mean(dist_of_pairs))
                                       + r', $\sigma=$' + '{:.2e}'.format(
                                     np.std(dist_of_pairs)),
                                 weights=np.ones_like(dist_of_pairs) / float(len(dist_of_pairs)))
                        plt.xlabel(' pearson coeff value')
                        plt.ylabel('PDF (normed hist)')
                        plt.title('Sister for ' + param)
                        plt.legend()
                        plt.savefig(
                            '/Users/alestawsky/PycharmProjects/untitled/pearsonoverindividual/Sisters/' + str(param))
                        plt.close()

                    # Plot the p-values
                    for param in pval_sis[list(pval_sis.keys())[0]]:
                        plt.figure()
                        dist_of_pairs = [pval_sis[key][param][1] for key in pval_sis.keys()]

                        plt.hist(x=dist_of_pairs, label=r'$\mu=$' + '{:.2e}'.format(np.mean(dist_of_pairs))
                                                        + r', $\sigma=$' + '{:.2e}'.format(np.std(dist_of_pairs)),
                                 weights=np.ones_like(dist_of_pairs) / float(len(dist_of_pairs)))
                        plt.xlabel(' p-values of pearson coeff')
                        plt.ylabel('PDF (normed hist)')
                        plt.title('Sister for ' + param)
                        plt.legend()
                        plt.savefig(
                            '/Users/alestawsky/PycharmProjects/untitled/pearsonoverindividual/Sisters/' + 'pval, ' + str(
                                param))
                        plt.close()

                if cloud == True:
                    keyA = random.sample(list(self.A_dict_sis.keys()), k=1)[0]
                    keyB = random.sample(list(self.B_dict_sis.keys()), k=1)[0]

                    for param in self.A_dict_sis[keyA].keys():
                        minlen = min(len(self.A_dict_sis[keyA][param]), len(self.B_dict_sis[keyB][param]))
                        plt.figure()
                        plt.scatter(self.A_dict_sis[keyA][param][:minlen], self.B_dict_sis[keyB][param][:minlen])
                        plt.xlabel(str(param) + " of the Sister A")
                        plt.ylabel(str(param) + " of the Sister B")
                        plt.title("are they linearly correlated? Pearson Coeff...")
                        plt.show()

            if 'n' in dsets:

                pval_non = dict()
                # Create the dictionaries with the pval stats
                for keyA, keyB in zip(self.A_dict_non_sis.keys(), self.B_dict_non_sis.keys()):
                    p_val_param = dict()
                    for param in self.A_dict_non_sis[keyA].keys():
                        sample1 = self.A_dict_non_sis[keyA][param]
                        sample2 = self.B_dict_non_sis[keyB][param]
                        minlen = min(len(sample1), len(sample2))
                        p_val = stats.pearsonr(x=sample1[:minlen], y=sample2[:minlen])
                        p_val_param.update({param: p_val})
                    pval_non.update({str(keyA) + ' ' + str(keyB): p_val_param})

                if graph == True:

                    # Plot the pearson correlations
                    for param in pval_non[list(pval_non.keys())[0]]:
                        plt.figure()
                        dist_of_pairs = [pval_non[key][param][0] for key in pval_non.keys()]

                        plt.hist(x=dist_of_pairs, range=[-.6, 1],
                                 label=r'$\mu=$' + '{:.2e}'.format(np.mean(dist_of_pairs))
                                       + r', $\sigma=$' + '{:.2e}'.format(
                                     np.std(dist_of_pairs)),
                                 weights=np.ones_like(dist_of_pairs) / float(len(dist_of_pairs)))
                        plt.xlabel(' pearson coeff value')
                        plt.ylabel('PDF (normed hist)')
                        plt.title('Non-Sister for ' + param)
                        plt.legend()
                        plt.savefig(
                            '/Users/alestawsky/PycharmProjects/untitled/pearsonoverindividual/Non/' + str(param))
                        plt.close()

                    # Plot the p-values
                    for param in pval_non[list(pval_non.keys())[0]]:
                        plt.figure()
                        dist_of_pairs = [pval_non[key][param][1] for key in pval_non.keys()]

                        plt.hist(x=dist_of_pairs, label=r'$\mu=$' + '{:.2e}'.format(np.mean(dist_of_pairs))
                                                        + r', $\sigma=$' + '{:.2e}'.format(np.std(dist_of_pairs)),
                                 weights=np.ones_like(dist_of_pairs) / float(len(dist_of_pairs)))
                        plt.xlabel(' p-values of pearson coeff')
                        plt.ylabel('PDF (normed hist)')
                        plt.title('Non-Sister for ' + param)
                        plt.legend()
                        plt.savefig(
                            '/Users/alestawsky/PycharmProjects/untitled/pearsonoverindividual/Non/' + 'pval, ' + str(
                                param))
                        plt.close()

                if cloud == True:
                    keyA = random.sample(list(self.A_dict_non_sis.keys()), k=1)[0]
                    keyB = random.sample(list(self.B_dict_non_sis.keys()), k=1)[0]

                    for param in self.A_dict_non_sis[keyA].keys():
                        minlen = min(len(self.A_dict_non_sis[keyA][param]), len(self.B_dict_non_sis[keyB][param]))
                        plt.figure()
                        plt.scatter(self.A_dict_non_sis[keyA][param][:minlen],
                                    self.B_dict_non_sis[keyB][param][:minlen])
                        plt.xlabel(str(param) + " of the Non-Sister A")
                        plt.ylabel(str(param) + " of the Non-Sister B")
                        plt.title("are they linearly correlated? Pearson Coeff...")
                        plt.show()

            if 'b' in dsets:

                pval_both = dict()
                # Create the dictionaries with the pval stats
                for keyA, keyB in zip(self.A_dict_both.keys(), self.B_dict_both.keys()):
                    p_val_param = dict()
                    for param in self.A_dict_both[keyA].keys():
                        sample1 = self.A_dict_both[keyA][param]
                        sample2 = self.B_dict_both[keyB][param]
                        minlen = min(len(sample1), len(sample2))
                        p_val = stats.pearsonr(x=sample1[:minlen], y=sample2[:minlen])
                        p_val_param.update({param: p_val})
                    pval_both.update({str(keyA) + ' ' + str(keyB): p_val_param})

                if graph == True:

                    # Plot the pearson correlations
                    for param in pval_both[list(pval_both.keys())[0]]:
                        plt.figure()
                        dist_of_pairs = [pval_both[key][param][0] for key in pval_both.keys()]

                        plt.hist(x=dist_of_pairs, range=[-.6, 1],
                                 label=r'$\mu=$' + '{:.2e}'.format(np.mean(dist_of_pairs))
                                       + r', $\sigma=$' + '{:.2e}'.format(
                                     np.std(dist_of_pairs)),
                                 weights=np.ones_like(dist_of_pairs) / float(len(dist_of_pairs)))
                        plt.xlabel(' pearson coeff value')
                        plt.ylabel('PDF (normed hist)')
                        plt.title('Control for ' + param)
                        plt.legend()
                        plt.savefig(
                            '/Users/alestawsky/PycharmProjects/untitled/pearsonoverindividual/Control/' + str(param))
                        plt.close()

                    # Plot the p-values
                    for param in pval_both[list(pval_both.keys())[0]]:
                        plt.figure()
                        dist_of_pairs = [pval_both[key][param][1] for key in pval_both.keys()]

                        plt.hist(x=dist_of_pairs, label=r'$\mu=$' + '{:.2e}'.format(np.mean(dist_of_pairs))
                                                        + r', $\sigma=$' + '{:.2e}'.format(np.std(dist_of_pairs)),
                                 weights=np.ones_like(dist_of_pairs) / float(len(dist_of_pairs)))
                        plt.xlabel(' p-values of pearson coeff')
                        plt.ylabel('PDF (normed hist)')
                        plt.title('Control for ' + param)
                        plt.legend()
                        plt.savefig(
                            '/Users/alestawsky/PycharmProjects/untitled/pearsonoverindividual/Control/' + 'pval, ' + str(
                                param))
                        plt.close()

                if cloud == True:
                    keyA = random.sample(list(self.A_dict_both.keys()), k=1)[0]
                    keyB = random.sample(list(self.B_dict_both.keys()), k=1)[0]

                    for param in self.A_dict_both[keyA].keys():
                        minlen = min(len(self.A_dict_both[keyA][param]), len(self.B_dict_both[keyB][param]))
                        plt.figure()
                        plt.scatter(self.A_dict_both[keyA][param][:minlen], self.B_dict_both[keyB][param][:minlen])
                        plt.xlabel(str(param) + " of the Control A")
                        plt.ylabel(str(param) + " of the Control B")
                        plt.title("are they linearly correlated? Pearson Coeff...")
                        plt.show()

            if comparelineages == True and ['s', 'n', 'b'] == dsets:
                for param in pval_sis[list(pval_sis.keys())[0]].keys():
                    plt.figure()

                    plt.errorbar(x=np.arange(3), y=[np.mean([pval_sis[key][param] for key in pval_sis.keys()]),
                                                    np.mean([pval_non[key][param] for key in pval_non.keys()]),
                                                    np.mean([pval_both[key][param] for key in pval_both.keys()])],
                                 yerr=[np.std([pval_sis[key][param] for key in pval_sis.keys()]),
                                       np.std([pval_non[key][param] for key in pval_non.keys()]),
                                       np.std([pval_both[key][param] for key in pval_both.keys()])])
                    plt.xticks(np.arange(3), ['sis', 'non-sis', 'control'], rotation=20)
                    plt.ylabel(str(param) + ': mean of pearson corr, error of std. div.')
                    plt.show()

        elif seperatebydsets_only == True and separatebysister == False:

            if 's' in dsets:

                pval_sis_param = dict()
                how_many_generations = 30
                # Get a correlation coefficient for each cycle and see how it drops
                for param in self.A_dict_sis[list(self.A_dict_sis.keys())[0]].keys():
                    p_val_cycle = dict()
                    pop_cycle = dict()
                    pop_sis_param = dict()
                    longest_cycle_A = max([len(self.A_dict_sis[keyA][param]) for keyA in self.A_dict_sis.keys()])
                    longest_cycle_B = max([len(self.B_dict_sis[keyB][param]) for keyB in self.B_dict_sis.keys()])
                    longest_cycle = min(longest_cycle_A, longest_cycle_B)
                    for cycle in range(longest_cycle):
                        sample1 = np.array([])
                        sample2 = np.array([])
                        # all the dset's pairs of values at this cycle for this parameter
                        for keyA, keyB in zip(self.A_dict_sis.keys(), self.B_dict_sis.keys()):
                            minlen = min(len(self.A_dict_sis[keyA][param]), len(self.B_dict_sis[keyB][param]))
                            if minlen > cycle:
                                sample1 = np.append(sample1, self.A_dict_sis[keyA][param][cycle])
                                sample2 = np.append(sample2, self.B_dict_sis[keyB][param][cycle])
                        population = len(sample1)
                        p_val = stats.pearsonr(x=sample1, y=sample2)
                        p_val_cycle.update({str(cycle): p_val})
                        pop_cycle.update({str(cycle): population})
                    pval_sis_param.update({str(param): p_val_cycle})
                    pop_sis_param.update({str(param): pop_cycle})

                    for param in pop_sis_param.keys():
                        plt.figure()
                        param_pop = np.array([pop_sis_param[param][cycle] for cycle in pop_sis_param[param].keys()])
                        # arraypval = np.array([pval_sis_param[param][cycles][1] for cycles in list(pval_sis_param[param].keys())[:81]])
                        # print(param_pop[param])
                        plt.plot(param_pop[:81], marker='.', label='number of samples')

                        # plt.plot(arraypval, marker='.', label='pvalue')
                        plt.xlabel('cycles')
                        plt.ylabel('number of samples used')
                        plt.title(param)
                        plt.savefig(
                            '/Users/alestawsky/PycharmProjects/untitled/pearsonoverset_timegraph/Sisters/populations_' + str(
                                param),
                            dpi=300)
                        plt.close()

                if graph == True:

                    # Plot the pearson correlations
                    for param in pval_sis_param.keys():
                        plt.figure()
                        dist_of_pairs = [pval_sis_param[param][cycle][0] for cycle in pval_sis_param[param].keys()]
                        dist_of_pairs = dist_of_pairs[:how_many_generations]
                        print(dist_of_pairs)
                        print('It has nans: ', np.any(np.isnan(dist_of_pairs)))
                        plt.hist(x=dist_of_pairs,
                                 label=r'$\mu=$' + '{:.2e}'.format(np.mean(dist_of_pairs))
                                       + r', $\sigma=$' + '{:.2e}'.format(
                                     np.std(dist_of_pairs)),
                                 weights=np.ones_like(dist_of_pairs) / float(len(dist_of_pairs)))
                        plt.xlabel('Pearson coeff value for all sisters that include that cycle in A and B')
                        plt.ylabel('PDF (normed hist)')
                        plt.title('Sister (up to 20 gens) for ' + param)
                        plt.legend()
                        plt.savefig('/Users/alestawsky/PycharmProjects/untitled/pearsonoverset/Sisters/' + str(param),
                                    dpi=300)
                        plt.close()

                    # Plot the p-values
                    for param in pval_sis_param.keys():
                        plt.figure()
                        dist_of_pairs = [pval_sis_param[param][cycle][1] for cycle in pval_sis_param[param].keys()]
                        dist_of_pairs = dist_of_pairs[:how_many_generations]
                        print('pvalue, ', dist_of_pairs)
                        print('pvalue, It has nans: ', np.any(np.isnan(dist_of_pairs)))
                        plt.hist(x=dist_of_pairs, label=r'$\mu=$' + '{:.2e}'.format(np.mean(dist_of_pairs))
                                                        + r', $\sigma=$' + '{:.2e}'.format(np.std(dist_of_pairs)),
                                 weights=np.ones_like(dist_of_pairs) / float(len(dist_of_pairs)))
                        plt.xlabel('p-values of pearson coeff for all sisters that include that cycle in A and B')
                        plt.ylabel('PDF (normed hist)')
                        plt.title('Sister (up to 20 gens) for ' + param)
                        plt.legend()
                        plt.savefig(
                            '/Users/alestawsky/PycharmProjects/untitled/pearsonoverset/Sisters/' + 'pval, ' + str(
                                param),
                            dpi=300)
                        plt.close()

                if time_graph == True:

                    # Plot the pearson correlations
                    for param in pval_sis_param.keys():
                        dist_of_pairs = [pval_sis_param[param][cycle][0] for cycle in pval_sis_param[param].keys()]
                        dist_of_pairs = np.array(dist_of_pairs[:how_many_generations])
                        print(dist_of_pairs)
                        print('It has nans: ', np.any(np.isnan(dist_of_pairs)))
                        # pvalue_label = r'$\mu=$' + r'{:.2e}'.format(np.mean(dist_of_pairs))+r', $\sigma=$' + r'{:.2e}'.format(np.std(dist_of_pairs))
                        plt.figure()
                        plt.plot(dist_of_pairs, marker='.')
                        plt.xlabel('Cycle/Generation Number')
                        plt.ylabel('Pearson Coefficient')
                        plt.title(r'Sister ' + param + r', $\mu=$' + '{:.2e}'.format(np.mean(dist_of_pairs))
                                  + r', $\sigma=$' + '{:.2e}'.format(np.std(dist_of_pairs)))
                        plt.savefig(
                            '/Users/alestawsky/PycharmProjects/untitled/pearsonoverset_timegraph/Sisters/' + str(param),
                            dpi=300)
                        plt.close()

                    # Plot the p-values
                    for param in pval_sis_param.keys():
                        dist_of_pairs = [pval_sis_param[param][cycle][1] for cycle in pval_sis_param[param].keys()]
                        dist_of_pairs = np.array(dist_of_pairs[:how_many_generations])
                        print('pvalue, ', dist_of_pairs)
                        print('pvalue, It has nans: ', np.any(np.isnan(dist_of_pairs)))
                        plt.figure()
                        plt.plot(np.log10(dist_of_pairs), marker='.')
                        plt.xlabel('Cycle/Generation Number')
                        plt.ylabel('log10 P-Value')
                        plt.title(r'Sister ' + param + r', $\mu=$' + '{:.2e}'.format(np.mean(dist_of_pairs))
                                  + r', $\sigma=$' + '{:.2e}'.format(np.std(dist_of_pairs)))
                        plt.savefig(
                            '/Users/alestawsky/PycharmProjects/untitled/pearsonoverset_timegraph/Sisters/' + 'pval, ' + str(
                                param),
                            dpi=300)
                        plt.close()

                if cloud == True:

                    cycle = random.sample([0, 1, 2, 3, 4, 5], k=1)[0]

                    for param in pval_sis_param.keys():

                        sample1 = np.array([])
                        sample2 = np.array([])
                        # all the dset's pairs of values at this cycle for this parameter
                        for keyA, keyB in zip(self.A_dict_sis.keys(), self.B_dict_sis.keys()):
                            minlen = min(len(self.A_dict_sis[keyA][param]), len(self.B_dict_sis[keyB][param]))
                            if minlen > cycle:
                                sample1 = np.append(sample1, self.A_dict_sis[keyA][param][cycle])
                                sample2 = np.append(sample2, self.B_dict_sis[keyB][param][cycle])

                        plt.figure()
                        plt.scatter(x=sample1, y=sample2)
                        plt.xlabel(str(param) + " of all Sister A in cycle " + str(cycle))
                        plt.ylabel(str(param) + " of all Sister B in cycle " + str(cycle))
                        plt.title("are they linearly correlated? Pearson Coeff (up to 20 gens)...")
                        plt.show()

            if 'n' in dsets:

                pval_non_param = dict()
                # Get a correlation coefficient for each cycle and see how it drops
                for param in self.A_dict_non_sis[list(self.A_dict_non_sis.keys())[0]].keys():
                    p_val_cycle = dict()
                    longest_cycle_A = max(
                        [len(self.A_dict_non_sis[keyA][param]) for keyA in self.A_dict_non_sis.keys()])
                    longest_cycle_B = max(
                        [len(self.B_dict_non_sis[keyB][param]) for keyB in self.B_dict_non_sis.keys()])
                    longest_cycle = min(longest_cycle_A, longest_cycle_B)
                    for cycle in range(longest_cycle):
                        sample1 = np.array([])
                        sample2 = np.array([])
                        # all the dset's pairs of values at this cycle for this parameter
                        for keyA, keyB in zip(self.A_dict_non_sis.keys(), self.B_dict_non_sis.keys()):
                            minlen = min(len(self.A_dict_non_sis[keyA][param]), len(self.B_dict_non_sis[keyB][param]))
                            if minlen > cycle:
                                sample1 = np.append(sample1, self.A_dict_non_sis[keyA][param][cycle])
                                sample2 = np.append(sample2, self.B_dict_non_sis[keyB][param][cycle])
                        p_val = stats.pearsonr(x=sample1, y=sample2)
                        p_val_cycle.update({str(cycle): p_val})
                    pval_non_param.update({str(param): p_val_cycle})

                if graph == True:

                    # Plot the pearson correlations
                    for param in pval_non_param.keys():
                        plt.figure()
                        dist_of_pairs = [pval_non_param[param][cycle][0] for cycle in pval_non_param[param].keys()]
                        how_many_generations = 20
                        dist_of_pairs = dist_of_pairs[:how_many_generations]
                        print(dist_of_pairs)
                        print('It has nans: ', np.any(np.isnan(dist_of_pairs)))
                        plt.hist(x=dist_of_pairs,
                                 label=r'$\mu=$' + '{:.2e}'.format(np.mean(dist_of_pairs))
                                       + r', $\sigma=$' + '{:.2e}'.format(
                                     np.std(dist_of_pairs)),
                                 weights=np.ones_like(dist_of_pairs) / float(len(dist_of_pairs)))
                        plt.xlabel('Pearson coeff value for all Non Sisters that include that cycle in A and B')
                        plt.ylabel('PDF (normed hist)')
                        plt.title('Non Sisters (up to 20 gens) for ' + param)
                        plt.legend()
                        plt.savefig('/Users/alestawsky/PycharmProjects/untitled/pearsonoverset/Non/' + str(param),
                                    dpi=300)
                        plt.close()

                    # Plot the p-values
                    for param in pval_non_param.keys():
                        plt.figure()
                        dist_of_pairs = [pval_non_param[param][cycle][1] for cycle in pval_non_param[param].keys()]
                        how_many_generations = 20
                        dist_of_pairs = dist_of_pairs[:how_many_generations]
                        print('pvalue, ', dist_of_pairs)
                        print('pvalue, It has nans: ', np.any(np.isnan(dist_of_pairs)))
                        plt.hist(x=dist_of_pairs, label=r'$\mu=$' + '{:.2e}'.format(np.mean(dist_of_pairs))
                                                        + r', $\sigma=$' + '{:.2e}'.format(np.std(dist_of_pairs)),
                                 weights=np.ones_like(dist_of_pairs) / float(len(dist_of_pairs)))
                        plt.xlabel('p-values of pearson coeff for all Non Sisters that include that cycle in A and B')
                        plt.ylabel('PDF (normed hist)')
                        plt.title('Non Sisters (up to 20 gens) for ' + param)
                        plt.legend()
                        plt.savefig(
                            '/Users/alestawsky/PycharmProjects/untitled/pearsonoverset/Non/' + 'pval, ' + str(param),
                            dpi=300)
                        plt.close()

                if time_graph == True:

                    # Plot the pearson correlations
                    for param in pval_non_param.keys():
                        dist_of_pairs = [pval_non_param[param][cycle][0] for cycle in pval_non_param[param].keys()]
                        dist_of_pairs = np.array(dist_of_pairs[:how_many_generations])
                        print(dist_of_pairs)
                        print('It has nans: ', np.any(np.isnan(dist_of_pairs)))
                        # pvalue_label = r'$\mu=$' + r'{:.2e}'.format(np.mean(dist_of_pairs))+r', $\sigma=$' + r'{:.2e}'.format(np.std(dist_of_pairs))
                        plt.figure()
                        plt.plot(dist_of_pairs, marker='.')
                        plt.xlabel('Cycle/Generation Number')
                        plt.ylabel('Pearson Coefficient')
                        plt.title(r'Non Sister ' + param + r', $\mu=$' + '{:.2e}'.format(np.mean(dist_of_pairs))
                                  + r', $\sigma=$' + '{:.2e}'.format(np.std(dist_of_pairs)))
                        plt.savefig(
                            '/Users/alestawsky/PycharmProjects/untitled/pearsonoverset_timegraph/Non/' + str(param),
                            dpi=300)
                        plt.close()

                    # Plot the p-values
                    for param in pval_non_param.keys():
                        dist_of_pairs = [pval_non_param[param][cycle][1] for cycle in pval_non_param[param].keys()]
                        dist_of_pairs = np.array(dist_of_pairs[:how_many_generations])
                        print('pvalue, ', dist_of_pairs)
                        print('pvalue, It has nans: ', np.any(np.isnan(dist_of_pairs)))
                        plt.figure()
                        plt.plot(np.log10(dist_of_pairs), marker='.')
                        plt.xlabel('Cycle/Generation Number')
                        plt.ylabel('log10 P-Value')
                        plt.title(r'Non Sister ' + param + r', $\mu=$' + '{:.2e}'.format(np.mean(dist_of_pairs))
                                  + r', $\sigma=$' + '{:.2e}'.format(np.std(dist_of_pairs)))
                        plt.savefig(
                            '/Users/alestawsky/PycharmProjects/untitled/pearsonoverset_timegraph/Non/' + 'pval, ' + str(
                                param),
                            dpi=300)
                        plt.close()

                if cloud == True:

                    cycle = random.sample([0, 1, 2, 3, 4, 5], k=1)[0]

                    for param in pval_sis_param.keys():

                        sample1 = np.array([])
                        sample2 = np.array([])
                        # all the dset's pairs of values at this cycle for this parameter
                        for keyA, keyB in zip(self.A_dict_non_sis.keys(), self.B_dict_non_sis.keys()):
                            minlen = min(len(self.A_dict_non_sis[keyA][param]), len(self.B_dict_non_sis[keyB][param]))
                            if minlen > cycle:
                                sample1 = np.append(sample1, self.A_dict_non_sis[keyA][param][cycle])
                                sample2 = np.append(sample2, self.B_dict_non_sis[keyB][param][cycle])

                        plt.figure()
                        plt.scatter(x=sample1, y=sample2)
                        plt.xlabel(str(param) + " of all Control A in cycle " + str(cycle))
                        plt.ylabel(str(param) + " of all Control B in cycle " + str(cycle))
                        plt.title("are they linearly correlated? Pearson Coeff (up to 20 gens)...")
                        plt.show()

            if 'b' in dsets:

                pval_both_par = dict()
                # Get a correlation coefficient for each cycle and see how it drops
                for param in self.A_dict_both[list(self.A_dict_both.keys())[0]].keys():
                    p_val_cycle = dict()
                    longest_cycle_A = max(
                        [len(self.A_dict_both[keyA][param]) for keyA in self.A_dict_both.keys()])
                    longest_cycle_B = max(
                        [len(self.B_dict_both[keyB][param]) for keyB in self.B_dict_both.keys()])
                    longest_cycle = min(longest_cycle_A, longest_cycle_B)
                    for cycle in range(longest_cycle):
                        sample1 = np.array([])
                        sample2 = np.array([])
                        # all the dset's pairs of values at this cycle for this parameter
                        for keyA, keyB in zip(self.A_dict_both.keys(), self.B_dict_both.keys()):
                            minlen = min(len(self.A_dict_both[keyA][param]), len(self.B_dict_both[keyB][param]))
                            if minlen > cycle:
                                sample1 = np.append(sample1, self.A_dict_both[keyA][param][cycle])
                                sample2 = np.append(sample2, self.B_dict_both[keyB][param][cycle])
                        p_val = stats.pearsonr(x=sample1, y=sample2)
                        p_val_cycle.update({str(cycle): p_val})
                    pval_both_par.update({str(param): p_val_cycle})

                if graph == True:

                    # Plot the pearson correlations
                    for param in pval_both_par.keys():
                        plt.figure()
                        dist_of_pairs = [pval_both_par[param][cycle][0] for cycle in pval_both_par[param].keys()]
                        how_many_generations = 20
                        dist_of_pairs = dist_of_pairs[:how_many_generations]
                        print(dist_of_pairs)
                        print('It has nans: ', np.any(np.isnan(dist_of_pairs)))
                        plt.hist(x=dist_of_pairs,
                                 label=r'$\mu=$' + '{:.2e}'.format(np.mean(dist_of_pairs))
                                       + r', $\sigma=$' + '{:.2e}'.format(
                                     np.std(dist_of_pairs)),
                                 weights=np.ones_like(dist_of_pairs) / float(len(dist_of_pairs)))
                        plt.xlabel('Pearson coeff value for all Control that include that cycle in A and B')
                        plt.ylabel('PDF (normed hist)')
                        plt.title('Control (up to 20 gens) for ' + param)
                        plt.legend()
                        plt.savefig('/Users/alestawsky/PycharmProjects/untitled/pearsonoverset/Control/' + str(param),
                                    dpi=300)
                        plt.close()

                    # Plot the p-values
                    for param in pval_both_par.keys():
                        plt.figure()
                        dist_of_pairs = [pval_both_par[param][cycle][1] for cycle in pval_both_par[param].keys()]
                        how_many_generations = 20
                        dist_of_pairs = dist_of_pairs[:how_many_generations]
                        print('pvalue, ', dist_of_pairs)
                        print('pvalue, It has nans: ', np.any(np.isnan(dist_of_pairs)))
                        plt.hist(x=dist_of_pairs, label=r'$\mu=$' + '{:.2e}'.format(np.mean(dist_of_pairs))
                                                        + r', $\sigma=$' + '{:.2e}'.format(np.std(dist_of_pairs)),
                                 weights=np.ones_like(dist_of_pairs) / float(len(dist_of_pairs)))
                        plt.xlabel('p-values of pearson coeff for all Control that include that cycle in A and B')
                        plt.ylabel('PDF (normed hist)')
                        plt.title('Control (up to 20 gens) for ' + param)
                        plt.legend()
                        plt.savefig(
                            '/Users/alestawsky/PycharmProjects/untitled/pearsonoverset/Control/' + 'pval, ' + str(
                                param),
                            dpi=300)
                        plt.close()

                if time_graph == True:

                    # Plot the pearson correlations
                    for param in pval_both_par.keys():
                        dist_of_pairs = [pval_both_par[param][cycle][0] for cycle in pval_both_par[param].keys()]
                        dist_of_pairs = np.array(dist_of_pairs[:how_many_generations])
                        print(dist_of_pairs)
                        print('It has nans: ', np.any(np.isnan(dist_of_pairs)))
                        # pvalue_label = r'$\mu=$' + r'{:.2e}'.format(np.mean(dist_of_pairs))+r', $\sigma=$' + r'{:.2e}'.format(np.std(dist_of_pairs))
                        plt.figure()
                        plt.plot(dist_of_pairs, marker='.')
                        plt.xlabel('Cycle/Generation Number')
                        plt.ylabel('Pearson Coefficient')
                        plt.title(r'Control Sister ' + param + r', $\mu=$' + '{:.2e}'.format(np.mean(dist_of_pairs))
                                  + r', $\sigma=$' + '{:.2e}'.format(np.std(dist_of_pairs)))
                        plt.savefig(
                            '/Users/alestawsky/PycharmProjects/untitled/pearsonoverset_timegraph/Control/' + str(param),
                            dpi=300)
                        plt.close()

                    # Plot the p-values
                    for param in pval_both_par.keys():
                        dist_of_pairs = [pval_both_par[param][cycle][1] for cycle in pval_both_par[param].keys()]
                        dist_of_pairs = np.array(dist_of_pairs[:how_many_generations])
                        print('pvalue, ', dist_of_pairs)
                        print('pvalue, It has nans: ', np.any(np.isnan(dist_of_pairs)))
                        plt.figure()
                        plt.plot(np.log10(dist_of_pairs), marker='.')
                        plt.xlabel('Cycle/Generation Number')
                        plt.ylabel('log10 P-Value')
                        plt.title(r'Control Sister ' + param + r', $\mu=$' + '{:.2e}'.format(np.mean(dist_of_pairs))
                                  + r', $\sigma=$' + '{:.2e}'.format(np.std(dist_of_pairs)))
                        plt.savefig(
                            '/Users/alestawsky/PycharmProjects/untitled/pearsonoverset_timegraph/Control/' + 'pval, ' + str(
                                param),
                            dpi=300)
                        plt.close()

                if cloud == True:

                    cycle = random.sample([0, 1, 2, 3, 4, 5], k=1)[0]

                    for param in pval_both_par.keys():

                        sample1 = np.array([])
                        sample2 = np.array([])
                        # all the dset's pairs of values at this cycle for this parameter
                        for keyA, keyB in zip(self.A_dict_both.keys(), self.B_dict_both.keys()):
                            minlen = min(len(self.A_dict_both[keyA][param]), len(self.B_dict_both[keyB][param]))
                            if minlen > cycle:
                                sample1 = np.append(sample1, self.A_dict_both[keyA][param][cycle])
                                sample2 = np.append(sample2, self.B_dict_both[keyB][param][cycle])

                        plt.figure()
                        plt.scatter(x=sample1, y=sample2)
                        plt.xlabel(str(param) + " of all Control A in cycle " + str(cycle))
                        plt.ylabel(str(param) + " of all Control B in cycle " + str(cycle))
                        plt.title("are they linearly correlated? Pearson Coeff (up to 20 gens)...")
                        plt.show()

            # if comparelineages == True and ['s', 'n', 'b'] == dsets:
            #     for param in pval_sis_param[list(pval_sis_param.keys())[0]].keys():
            #         plt.figure()
            #         # plt.errorbar(x=, y=, yer=)
            #         plt.errorbar(x=np.arange(3), y=[np.mean([pval_sis_param[key][param] for key in pval_sis_param.keys()]),
            #                                         np.mean([pval_non_param[key][param] for key in pval_non.keys()]),
            #                                         np.mean([pval_both[key][param] for key in pval_both.keys()])],
            #                      yerr=[np.std([pval_sis[key][param] for key in pval_sis.keys()]),
            #                            np.std([pval_non[key][param] for key in pval_non.keys()]),
            #                            np.std([pval_both[key][param] for key in pval_both.keys()])])
            #         plt.xticks(np.arange(3), ['sis', 'non-sis', 'control'], rotation=20)
            #         plt.ylabel(str(param)+': mean of pearson corr, error of std. div.')
            #         plt.show()

        """


        pval_both_pairs = dict()
        for keyA, keyB in zip(self.A_dict_both.keys(), self.B_dict_both.keys()):
            p_val_param = dict()
            for param in self.A_dict_both[keyA].keys():
                sample1 = self.A_dict_both[keyA][param]
                sample2 = self.B_dict_both[keyB][param]
                minlen = min(len(sample1), len(sample2))
                p_val = stats.pearsonr(sample1[:minlen], sample2[:minlen])
                p_val_param.update({param: p_val})
            pval_both_pairs.update({str(keyA) + ' ' + str(keyB): p_val_param})

        # Plot the pearson correlations
        for param in pval_both_pairs[list(pval_both_pairs.keys())[0]]:
            plt.figure()
            dist_of_pairs = [pval_both_pairs[key][param][0] for key in pval_both_pairs.keys()]

            plt.hist(x=dist_of_pairs, range=[-.6, 1], label=r'$\mu=$' + '{:.2e}'.format(np.mean(dist_of_pairs))
                                                            + r', $\sigma=$' + '{:.2e}'.format(np.std(dist_of_pairs)),
                     weights=np.ones_like(dist_of_pairs) / float(len(dist_of_pairs)))
            plt.xlabel(' pearson coeff value')
            plt.ylabel('PDF (normed hist)')
            plt.title('Control Pairs for ' + param)
            plt.legend()
            plt.show()

        # Plot the p-values
        for param in pval_both_pairs[list(pval_both_pairs.keys())[0]]:
            plt.figure()
            dist_of_pairs = [pval_both_pairs[key][param][1] for key in pval_both_pairs.keys()]

            plt.hist(x=dist_of_pairs, label=r'$\mu=$' + '{:.2e}'.format(np.mean(dist_of_pairs))
                                            + r', $\sigma=$' + '{:.2e}'.format(np.std(dist_of_pairs)),
                     weights=np.ones_like(dist_of_pairs) / float(len(dist_of_pairs)))
            plt.xlabel(' p-values of pearson coeff')
            plt.ylabel('PDF (normed hist)')
            plt.title('Control Pairs for ' + param)
            plt.legend()
            plt.show()
        """

        # for param in pval_sis_pairs.keys():
        # print(np.mean([]))
        # for param in pval_sis_pairs[key].keys():
        # fig, ax1 = plt.subplots()
        # ax1.set_xlabel('cycle/generation/division number')
        # ax1.set_ylabel('left-tailed p-value of Null Hyp.', color='tab:red')
        # print(pval_sis_pairs[key][param][0])
        # ax1.plot(pval_sis_pairs[key][param][0], color='red')
        # ax1.set_title(key)
        # ax2 = ax1.twinx()
        # print(pval_sis_pairs[key][param][1])
        # ax2.plot(pval_sis_pairs[key][param][1], color='blue')
        # ax2.set_ylabel('right-tailed p-value of Null Hyp.', color='tab:blue')
        # plt.show()

        # plt.ylabel(str(param) + ' pvalues')
        # plt.savefig('/Users/alestawsky/PycharmProjects/untitled/PearsonIndividual/Sisters/'
        #     + str(param) + '/' + str(key), dpi=300)
        # plt.clear()
        # print('0 ',list(pval_sis_pairs[key][param]))
        # print('1 ',pval_sis_pairs[key][param][1])

        # for keyA, keyB in zip(self.A_dict_sis.keys(), self.B_dict_sis.keys()):
        #     for param in self.A_dict_sis[list(self.A_dict_sis[keyA].keys())[0]].keys():
        #         sample1 = self.A_dict_sis[keyA][param]
        #         sample2 = self.B_dict_sis[keyB][param]
        #         p_val = stats.pearsonr(sample1, sample2)
        #         print('pval0', p_val[0])
        #         print('pval1', p_val[1])
        # plt.title(str(keyA)+ str(keyB) + str())

    # TRANSLATE
    def PearsonAndPopulation(self, dsetA=None, dsetB=None, dtype=None, significance_thresh=None,
                             displacement=None):

        """
        dsetA is the s/n/b sister ID A dictionary, dsetB is the s/n/b sister ID B dictionary,
        dtype is 'Sister','Non-Sister', or 'Control'. significance_thresh is the p-value upperbound that a
        pearson coefficient must have to make it onto the time-plot/histogram.
        """
        if significance_thresh == None:
            significance_thresh = 1
        if displacement == None:
            displacement = 0

        pval_param_A = dict()
        pval_param_B = dict()
        pop_param_A = dict()
        pop_param_B = dict()
        how_many_generations = 30
        # Get a correlation coefficient for each cycle and see how it drops
        for param in dsetA[list(dsetA.keys())[0]].keys():
            p_val_cycle_A_disp = dict()
            p_val_cycle_B_disp = dict()
            pop_cycle_A_disp = dict()
            pop_cycle_B_disp = dict()
            pop_param = dict()
            longest_cycle_A = max([len(dsetA[keyA][param]) for keyA in dsetA.keys()])
            longest_cycle_B = max([len(dsetB[keyB][param]) for keyB in dsetB.keys()])
            longest_cycle = min(longest_cycle_A, longest_cycle_B)
            for cycle in range(longest_cycle - displacement):
                sample1 = np.array([])
                sample2 = np.array([])
                for dispA, dispB in [[0, displacement], [displacement, 0]]:
                    # all the dset's pairs of values at this cycle for this parameter
                    for keyA, keyB in zip(dsetA.keys(), dsetB.keys()):
                        minlen = min(len(dsetA[keyA][param]), len(dsetB[keyB][param]))
                        if minlen > cycle:
                            sample1 = np.append(sample1, dsetA[keyA][param][cycle + dispA])
                            sample2 = np.append(sample2, dsetB[keyB][param][cycle + dispB])
                    population = len(sample1)
                    p_val = stats.pearsonr(x=sample1, y=sample2)
                    if dispA == 0:
                        p_val_cycle_B_disp.update({str(cycle): p_val})
                        pop_cycle_B_disp.update({str(cycle): population})
                    elif dispB == 0:
                        p_val_cycle_A_disp.update({str(cycle): p_val})
                        pop_cycle_A_disp.update({str(cycle): population})
            pval_param_A.update({str(param): p_val_cycle_A_disp})
            pval_param_B.update({str(param): p_val_cycle_B_disp})
            pop_param_A.update({str(param): pop_cycle_A_disp})
            pop_param_B.update({str(param): pop_cycle_B_disp})

        return pop_param_A, pop_param_B, pval_param_A, pval_param_B

    # TRANSLATE
    def GraphsOfCycleChars(self):
        os.mkdir('/Users/alestawsky/PycharmProjects/untitled/Sis_AB_comparisons')
        for key in self.dict_with_all_sister_traces['Sister_Trace_A_0'].keys():
            os.mkdir('/Users/alestawsky/PycharmProjects/untitled/Sis_AB_comparisons/' + key)

        for dataID in range(len(self.dict_with_all_sister_traces.values()) // 2):
            data_A = self.dict_with_all_sister_traces['Sister_Trace_A_' + str(dataID)]
            data_B = self.dict_with_all_sister_traces['Sister_Trace_B_' + str(dataID)]
            for key in data_A.keys():
                my_dir = '/Users/alestawsky/PycharmProjects/untitled/Sis_AB_comparisons/' + key + '/'
                plt.figure()
                plt.plot(np.arange(len(data_A[key])), data_A[key], 'ro', label='Trajectory A')  # red circles
                plt.plot(np.arange(len(data_B[key])), data_B[key], 'bo', label='Trajectory B')  # blue circles
                plt.xlabel('Division Generations')
                plt.ylabel('value of {}'.format(key))
                plt.title('Sister A/B, id: {}, {} comparison'.format(dataID, key))
                plt.legend()
                plt.savefig(my_dir + str(dataID) + '.png', bbox_inches='tight')

    # KEEP FOR REFERENCE BUT DELETE FOR NEXT EDITION
    def GraphTrajCorrelations(self):
        # #  MAKE THE AUTOCORRELATION CYCLE SIS/NONSIS DIRECTORIES
        # os.mkdir('/Users/alestawsky/PycharmProjects/untitled/Sis_Traj_data_ACF_graphs')
        # os.mkdir('/Users/alestawsky/PycharmProjects/untitled/NonSis_Traj_data_ACF_graphs')
        # os.mkdir('/Users/alestawsky/PycharmProjects/untitled/Both_Traj_data_ACF_graphs')
        #
        # # LOOP THROUGH ALL PAIRS OF SISTER CELLS A,B
        # sis_mean_corr_per_cycle = dict()
        # acfa_array = dict()
        # acfb_array = dict()
        #
        # for dataID in range(len(self.Sisters)):
        #     # Calculate their ACF
        #     acfA, acfB = self.Sisters.AutocorrelationFFT(dataID, normalize=True, maxlen=len(self.Sisters), enforcelen=False)
        #     # separate by parameter
        #     acfa_array.update({dataID:acfA})
        #     acfb_array.update({dataID:acfB})
        #     if dataID == 0:
        #         # MAKE THE CYCLE PARAMETERS LENGTH, AREA, FLUORESCENCE, MEANFLUORESCHENCE DIRECTORIES INSIDE SUPER DIRECTORY ABOVE
        #         for param in acfA.keys():
        #             # SHOULD BE 4 EACH
        #             os.mkdir('/Users/alestawsky/PycharmProjects/untitled/Sis_Traj_data_ACF_graphs/' + param)
        #             os.mkdir('/Users/alestawsky/PycharmProjects/untitled/NonSis_Traj_data_ACF_graphs/' + param)
        #             os.mkdir('/Users/alestawsky/PycharmProjects/untitled/Both_Traj_data_ACF_graphs/' + param)
        #
        #     for keyA, keyB in zip(acfA.keys(), acfB.keys()):
        #         fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, dpi = 300)
        #         ax1.axhline(y=0, color='k')
        #         ax1.plot(acfA[keyA], label='Trajectory A', marker='.')
        #         ax1.legend()
        #         ax2.axhline(y=0, color='k')
        #         ax2.plot(acfB[keyB], label='Trajectory B', marker='.')
        #         plt.xlabel('value of K (displacement)')
        #         yyl = plt.ylabel('correlation of '+str(keyB))
        #         yyl.set_position((yyl.get_position()[0], 1))  # This says use the top of the bottom axis as the reference point.
        #         yyl.set_verticalalignment('center')
        #         plt.suptitle('Sisters, id: {}'.format(dataID))
        #         ax2.legend()
        #         plt.savefig('/Users/alestawsky/PycharmProjects/untitled/Sis_Traj_data_ACF_graphs/'+str(keyA)+'/'
        #                     + str(dataID) + '.png', bbox_inches='tight')
        #         plt.close(fig)
        # # Get the mean of all correlations in a certain cycle
        # mean_of_correlations = dict()
        # std_of_correlations = dict()
        # cn_lives = range(max([len(self.sis_acfA_array[id][key]) for id in range(len(self.sis_acfA_array))]))
        # cn1_lives = range(max([len(self.sis_acfB_array[id][key]) for id in range(len(self.sis_acfB_array))]))
        # for key in self.sis_acfA_array[0].keys():
        #     for cn, cn1 in zip(cn_lives, cn1_lives):
        #         bla = np.array([di for di in range(len(self.sis_acfA_array)) if len(self.sis_acfA_array[di][key]) > cn])
        #         bla1 = np.array(
        #             [di for di in range(len(self.sis_acfB_array)) if len(self.sis_acfB_array[di][key]) > cn1])
        #         if cn == 0:
        #             mean_of_correlations.update({key: np.mean([self.sis_acfA_array[blaA][key][cn] - self.sis_acfB_array
        #             [blaB][key][cn] for blaA, blaB in zip(bla, bla1)])})
        #         if cn1 == 0:
        #             std_of_correlations.update({key: np.std([self.sis_acfA_array[blaA][key][cn] - self.sis_acfB_array
        #             [blaB][key][cn] for blaA, blaB in zip(bla, bla1)])})
        #         else:
        #             mean_of_correlations[key] = np.append(mean_of_correlations[key], np.mean([self.sis_acfA_array[blaA]
        #                                                                                       [key][cn] -
        #                                                                                       self.sis_acfB_array[blaB][
        #                                                                                           key][cn] for
        #                                                                                       blaA, blaB in
        #                                                                                       zip(bla, bla1)]))
        #             std_of_correlations[key] = np.append(std_of_correlations[key], np.std([self.sis_acfA_array[blaA]
        #                                                                                    [key][cn] -
        #                                                                                    self.sis_acfB_array[blaB][
        #                                                                                        key][cn] for blaA, blaB
        #                                                                                    in zip(bla, bla1)]))
        #
        # # Print the standard deviation of the correlations
        # fig = plt.figure()
        # for key in std_of_correlations.keys():
        #     plt.plot(std_of_correlations[key], label=key)
        # plt.legend()
        # plt.ylabel('std of corrs')
        # plt.xlabel('cycle number')
        # plt.savefig('/Users/alestawsky/PycharmProjects/untitled/Sis_Traj_data_ACF_graphs/std_dev_over_cycles.png', bbox_inches='tight')
        # plt.close(fig)
        # # Print the mean of the correlations
        # fig = plt.figure()
        # for key in mean_of_correlations.keys():
        #     plt.plot(mean_of_correlations[key], label=key)
        # plt.legend()
        # plt.ylabel('mean of corrs')
        # plt.xlabel('cycle number')
        # plt.savefig('/Users/alestawsky/PycharmProjects/untitled/Sis_Traj_data_ACF_graphs/mean_over_cycles.png',
        #             bbox_inches='tight')
        # plt.close(fig)

        # LOOP THROUGH ALL PAIRS OF NONSISTER CELLS A,B
        sis_mean_corr_per_cycle = dict()
        acfa_array = dict()
        acfb_array = dict()

        for dataID in range(len(self.Nonsisters)):
            # Calculate their ACF
            acfA, acfB = self.Nonsisters.AutocorrelationFFT(dataID, normalize=True, maxlen=len(self.Sisters),
                                                            enforcelen=False)
            # separate by parameter
            acfa_array.update({dataID: acfA})
            acfb_array.update({dataID: acfB})

            for keyA, keyB in zip(acfA.keys(), acfB.keys()):
                fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, dpi=300)
                ax1.axhline(y=0, color='k')
                ax1.plot(acfA[keyA], label='Trajectory A', marker='.')
                ax1.legend()
                ax2.axhline(y=0, color='k')
                ax2.plot(acfB[keyB], label='Trajectory B', marker='.')
                plt.xlabel('value of K (displacement)')
                yyl = plt.ylabel('correlation of ' + str(keyB))
                yyl.set_position(
                    (yyl.get_position()[0], 1))  # This says use the top of the bottom axis as the reference point.
                yyl.set_verticalalignment('center')
                plt.suptitle('Non-Sisters, id: {}'.format(dataID))
                ax2.legend()
                plt.savefig('/Users/alestawsky/PycharmProjects/untitled/NonSis_Traj_data_ACF_graphs/' + str(keyA) + '/'
                            + str(dataID) + '.png', bbox_inches='tight')
                plt.close(fig)

        # LOOP THROUGH ALL PAIRS OF BOTH CELLS A,B
        sis_mean_corr_per_cycle = dict()
        acfa_array = dict()
        acfb_array = dict()

        for dataID in range(len(self.Both)):
            # Calculate their ACF
            acfA, acfB = self.Both.AutocorrelationFFT(dataID, normalize=True, maxlen=len(self.Sisters),
                                                      enforcelen=False)
            # separate by parameter
            acfa_array.update({dataID: acfA})
            acfb_array.update({dataID: acfB})

            for keyA, keyB in zip(acfA.keys(), acfB.keys()):
                fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, dpi=300)
                ax1.axhline(y=0, color='k')
                ax1.plot(acfA[keyA], label='Trajectory A', marker='.')
                ax1.legend()
                ax2.axhline(y=0, color='k')
                ax2.plot(acfB[keyB], label='Trajectory B', marker='.')
                plt.xlabel('value of K (displacement)')
                yyl = plt.ylabel('correlation of ' + str(keyB))
                yyl.set_position(
                    (yyl.get_position()[0],
                     1))  # This says use the top of the bottom axis as the reference point.
                yyl.set_verticalalignment('center')
                plt.suptitle('Both, id: {}'.format(dataID))
                ax2.legend()
                plt.savefig(
                    '/Users/alestawsky/PycharmProjects/untitled/Both_Traj_data_ACF_graphs/' + str(keyA) + '/'
                    + str(dataID) + '.png', bbox_inches='tight')
                plt.close(fig)

    # ALMOST USELESS...
    def CrossCorrelation(self):
        # longest non_sis to use to as control
        non_sis_data_id = [dataid for dataid in range(len(self.Nonsisters)) if len(self.Nonsisters[dataid]) ==
                           max([len(self.Nonsisters[di]) for di in range(len(self.Nonsisters))])]
        print(non_sis_data_id[0])
        data_C = self.Nonsisters[non_sis_data_id[0]]
        os.mkdir('/Users/alestawsky/PycharmProjects/untitled/Sis_AB_correlations')
        for key in [t for t in self.Sisters[0].columns.values if t != 'timeA' and t != 'timeB']:
            os.mkdir('/Users/alestawsky/PycharmProjects/untitled/Sis_AB_correlations/' + key)

        for dataID in range(len(self.Sisters)):
            data_A = self.Sisters[dataID][[key for key in self.Sisters[dataID].columns.values if key.endswith('A')]]
            data_B = self.Sisters[dataID][[key for key in self.Sisters[dataID].columns.values if key.endswith('B')]]

            for keyA, keyB in zip(data_A.keys(), data_B.keys()):
                if keyA != 'timeA':
                    keyA = ''.join(keyA)
                    keyB = ''.join(keyB)
                    my_dir = '/Users/alestawsky/PycharmProjects/untitled/Sis_AB_correlations/' + key + '/'
                    plt.figure()
                    ax1 = plt.subplot(211)
                    sister_corr = np.correlate(data_A[keyA], data_B[keyB], mode='full')
                    nonsis_corr = np.correlate(data_A[keyA], data_C[keyA], mode='full')
                    ax1.plot(sister_corr, 'r', label='Sister Cells')  # red circles
                    ax2 = plt.subplot(212)
                    ax2.plot(nonsis_corr, 'b', label='Non-Sister Cells')  # blue circles
                    plt.xlabel('Division Generations')
                    yyl = plt.ylabel('value of {} correlation'.format(key))
                    yyl.set_position(
                        (yyl.get_position()[0], 1))  # This says use the top of the bottom axis as the reference point.
                    yyl.set_verticalalignment('center')
                    plt.suptitle('Sister A/B, id: {}, {} correlation'.format(dataID, keyA[:-1]))
                    plt.legend()
                    plt.savefig(my_dir + str(dataID) + '.png', bbox_inches='tight')

    def CompMeans(self):

        # NOTE THAT WE ARE CHOOSING THE FIRST 87 SISTER TRACES INSTEAD OF 132 WE HAVE
        print('lengths of sis A and B', len(list(self.A_dict_sis.keys())[:87]), len(list(self.B_dict_sis.keys())[:87]))
        print('lengths of non sis A and B', len(self.A_dict_non_sis.keys()), len(self.B_dict_non_sis.keys()))
        print('lengths of both A and B', len(self.A_dict_both.keys()), len(self.B_dict_both.keys()))

        some = dict()
        for cycle_param in self.A_dict_sis['Sister_Trace_A_0'].keys():
            some.update({cycle_param: np.mean(np.abs([np.mean(self.A_dict_sis[TrajA][cycle_param])
                                                      - np.mean(self.B_dict_sis[TrajB][cycle_param]) for TrajA, TrajB in
                                                      zip(list(self.A_dict_sis.keys())[:87],
                                                          self.B_dict_sis.keys())]))})
        other = dict()
        for cycle_param in self.A_dict_non_sis['Non-sister_Trace_A_0'].keys():
            other.update({cycle_param: np.mean(np.abs([np.mean(self.A_dict_non_sis[TrajA][cycle_param])
                                                       - np.mean(self.B_dict_non_sis[TrajB][cycle_param]) for
                                                       TrajA, TrajB in
                                                       zip(self.A_dict_non_sis.keys(), self.B_dict_non_sis.keys())]))})
        more = dict()
        for cycle_param in self.A_dict_both[list(self.A_dict_both.keys())[0]].keys():
            more.update({cycle_param: np.mean(np.abs([np.mean(self.A_dict_both[TrajA][cycle_param])
                                                      - np.mean(self.B_dict_both[TrajB][cycle_param]) for
                                                      TrajA, TrajB in
                                                      zip(self.A_dict_both.keys(), self.B_dict_both.keys())]))})
        some1 = dict()
        for cycle_param in self.A_dict_sis['Sister_Trace_A_0'].keys():
            some1.update({cycle_param: np.std(np.abs([np.mean(self.A_dict_sis[TrajA][cycle_param])
                                                      - np.mean(self.B_dict_sis[TrajB][cycle_param]) for TrajA, TrajB in
                                                      zip(list(self.A_dict_sis.keys())[:87],
                                                          list(self.B_dict_sis.keys())[:87])]))})
        other1 = dict()
        for cycle_param in self.A_dict_non_sis['Non-sister_Trace_A_0'].keys():
            other1.update({cycle_param: np.std(np.abs([np.mean(self.A_dict_non_sis[TrajA][cycle_param])
                                                       - np.mean(self.B_dict_non_sis[TrajB][cycle_param]) for
                                                       TrajA, TrajB in
                                                       zip(self.A_dict_non_sis.keys(), self.B_dict_non_sis.keys())]))})
        more1 = dict()
        for cycle_param in self.A_dict_both[list(self.A_dict_both.keys())[0]].keys():
            more1.update({cycle_param: np.std(np.abs([np.mean(self.A_dict_both[TrajA][cycle_param])
                                                      - np.mean(self.B_dict_both[TrajB][cycle_param]) for
                                                      TrajA, TrajB in
                                                      zip(self.A_dict_both.keys(), self.B_dict_both.keys())]))})

        dist = dict()
        for cycle_param in self.A_dict_sis['Sister_Trace_A_0'].keys():
            dist.update({cycle_param: np.abs([np.mean(self.A_dict_sis[TrajA][cycle_param])
                                              - np.mean(self.B_dict_sis[TrajB][cycle_param]) for TrajA, TrajB in
                                              zip(list(self.A_dict_sis.keys())[:87],
                                                  list(self.B_dict_sis.keys())[:87])])})
        dist1 = dict()
        for cycle_param in self.A_dict_non_sis['Non-sister_Trace_A_0'].keys():
            dist1.update({cycle_param: np.abs([np.mean(self.A_dict_non_sis[TrajA][cycle_param])
                                               - np.mean(self.B_dict_non_sis[TrajB][cycle_param]) for
                                               TrajA, TrajB in
                                               zip(self.A_dict_non_sis.keys(), self.B_dict_non_sis.keys())])})
        dist2 = dict()
        for cycle_param in self.A_dict_both[list(self.A_dict_both.keys())[0]].keys():
            dist2.update({cycle_param: np.abs([np.mean(self.A_dict_both[TrajA][cycle_param])
                                               - np.mean(self.B_dict_both[TrajB][cycle_param]) for
                                               TrajA, TrajB in
                                               zip(self.A_dict_both.keys(), self.B_dict_both.keys())])})

        return some, some1, dist, other, other1, dist1, more, more1, dist2

