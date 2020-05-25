# MUST IMPORT DEFNS, DATA (DICT) SCRIPT, and

from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


def lsqm(x, y, cov):
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


def main(ensemble, graph, metadatastruct, pooled):

    phi_star_A_array = []
    phi_star_B_array = []
    beta_A_array = []
    beta_B_array = []
    x_valuesA=np.array([])
    x_valuesB=[]
    y_valuesA=[]
    y_valuesB=[]
    if ensemble == 'Sisters only': # A VS B in Sister Cells (Not Pooled)
        for dfA, dfB in zip(metadatastruct.A_dict_sis.values(), metadatastruct.B_dict_sis.values()):
            phi_star_A, beta_A = lsqm(x=np.log(np.divide(dfA['length_birth'], metadatastruct.x_avg)),
                                      y=np.multiply(dfA['generationtime'], dfA['growth_length']),
                                      cov=False)
            phi_star_B, beta_B = lsqm(x=np.log(np.divide(dfB['length_birth'], metadatastruct.x_avg)),
                                      y=np.multiply(dfB['generationtime'], dfB['growth_length']),
                                      cov=False)
            if pooled:
                x_valuesA = np.append(x_valuesA, np.log(np.divide(dfA['length_birth'], metadatastruct.x_avg)))
                x_valuesB = np.append(x_valuesB, np.log(np.divide(dfB['length_birth'], metadatastruct.x_avg)))
                y_valuesA = np.append(y_valuesA, np.multiply(dfA['generationtime'], dfA['growth_length']))
                y_valuesB = np.append(y_valuesB, np.multiply(dfB['generationtime'], dfB['growth_length']))
            phi_star_A_array.append(phi_star_A)
            phi_star_B_array.append(phi_star_B)
            beta_A_array.append(beta_A)
            beta_B_array.append(beta_B)
        # print('phi_star_A_array:',len(phi_star_A_array))
        # print('phi_star_B_array:',len(phi_star_B_array))
        # print('beta_A_array:',len(beta_A_array))
        # print('beta_B_array:',len(beta_B_array))

        if graph:
            # PLOTS BETAS

            # ONLY IN WINDOWS WITHOUT SEABORN
            # plt.figure(figsize=(20, 20))

            # plt.scatter(dfA['length_birth'])
            t = np.arange(-2.5, 2.5)
            s = [[phi + beta * ti for phi, beta in zip(phi_star_A_array, beta_A_array)] for ti in t]
            s1 = [[phi + beta * ti for phi, beta in zip(phi_star_B_array, beta_B_array)] for ti in t]
            ff = plt.plot(s, color='r', label='Sister Cells')
            gg = plt.plot(s1, color='b', label='Non-Sister Cells')
            plt.title(r'Individual $\beta$ for Sister/Non-Sister Cells Trajectories')
            plt.xlabel(r'$ln(\frac{x_n^{(k)}}{x^*})$', fontsize=25)
            plt.ylabel(r'$\phi^{(k)}_n$', fontsize=25)
            plt.legend(handles=[ff[0], gg[0]])
            plt.show()
        if pooled:
            # DICTIONARY OF LENGTH BIRTH FINAL FN GENTIME GROWTHLENGTH (KEYS) THEN THE CYCLE NUMBER THEN
            columns_pooled_over_cycles_A_sis = {key : [[df[key].iloc[num] for df in metadatastruct.A_dict_sis.values() if len(df) - 1
                            >= num] for num in range(max([len(DF) for DF in metadatastruct.A_dict_sis.values()]))] for
                                  key in metadatastruct.A_dict_sis['Sister_Trace_A_0'].columns}
            columns_pooled_over_cycles_B_sis = {
                key: [[df[key].iloc[num] for df in metadatastruct.B_dict_sis.values() if len(df) - 1
                       >= num] for num in range(max([len(DF) for DF in metadatastruct.B_dict_sis.values()]))] for
                key in metadatastruct.B_dict_sis['Sister_Trace_B_0'].columns}

            x_values = np.append(x_valuesA, x_valuesB)
            y_values = np.append(y_valuesA, y_valuesB)
            pooled_phi_star_A, pooled_beta_A = lsqm(x=x_valuesA,
                                      y=y_valuesA,
                                      cov=False)
            pooled_phi_star_B, pooled_beta_B = lsqm(x=x_valuesB,
                                      y=y_valuesB,
                                      cov=False)
            pooled_phi_star, pooled_beta = lsqm(x=x_values,
                                                y=y_values,
                                                cov=False)

            # DICTIONARY TO ACCESS THE POOLED PHI, BETA FOR A, B AND ALSO COLUMNS POOLED OVER CYCLES
            # LAST ONE IS: pooled_data_dict['columns_pooled_over_cycles_A_sis'] = list of instances at that cycle
            # and spans all cycles.
            pooled_data_dict = {
                                'pooled_phi_star_A' : pooled_phi_star_A,
                                'pooled_beta_A' : pooled_beta_A,
                                'pooled_phi_star_B': pooled_phi_star_B,
                                'pooled_beta_B': pooled_beta_B,
                                'pooled_phi_star': pooled_phi_star,
                                'pooled_beta': pooled_beta,
                                'columns_pooled_over_cycles_A_sis': columns_pooled_over_cycles_A_sis,
                                'columns_pooled_over_cycles_B_sis': columns_pooled_over_cycles_B_sis

            }
        return phi_star_A_array, phi_star_B_array, beta_A_array, beta_B_array, pooled_data_dict

    elif ensemble == 'NonSisters only': # A VS B in NonSister Cells (Not Pooled)
        for dfA, dfB in zip(metadatastruct.A_dict_non_sis.values(), metadatastruct.B_dict_non_sis.values()):
            phi_star_A, beta_A = lsqm(x=np.log(np.divide(dfA['length_birth'], metadatastruct.x_avg)),
                                      y=np.multiply(dfA['generationtime'], dfA['growth_length']),
                                      cov=False)
            phi_star_B, beta_B = lsqm(x=np.log(np.divide(dfB['length_birth'], metadatastruct.x_avg)),
                                      y=np.multiply(dfB['generationtime'], dfB['growth_length']),
                                      cov=False)
            if pooled:
                x_valuesA = np.append(x_valuesA, np.log(np.divide(dfA['length_birth'], metadatastruct.x_avg)))
                x_valuesB = np.append(x_valuesB, np.log(np.divide(dfB['length_birth'], metadatastruct.x_avg)))
                y_valuesA = np.append(y_valuesA, np.multiply(dfA['generationtime'], dfA['growth_length']))
                y_valuesB = np.append(y_valuesB, np.multiply(dfB['generationtime'], dfB['growth_length']))
            phi_star_A_array.append(phi_star_A)
            phi_star_B_array.append(phi_star_B)
            beta_A_array.append(beta_A)
            beta_B_array.append(beta_B)

        if graph:
            # PLOTS BETAS

            # ONLY IN WINDOWS WITHOUT SEABORN
            # plt.figure(figsize=(20, 20))

            # plt.scatter(dfA['length_birth'])
            t = np.arange(-2.5, 2.5)
            s = [[phi + beta * ti for phi, beta in zip(phi_star_A_array, beta_A_array)] for ti in t]
            s1 = [[phi + beta * ti for phi, beta in zip(phi_star_B_array, beta_B_array)] for ti in t]
            ff = plt.plot(s, color='r', label='Sister Cells')
            gg = plt.plot(s1, color='b', label='Non-Sister Cells')
            plt.title(r'Individual $\beta$ for Sister/Non-Sister Cells Trajectories')
            plt.xlabel(r'$ln(\frac{x_n^{(k)}}{x^*})$', fontsize=25)
            plt.ylabel(r'$\phi^{(k)}_n$', fontsize=25)
            plt.legend(handles=[ff[0], gg[0]])
            plt.show()
        if pooled:
            columns_pooled_over_cycles_A_non_sis = {
                key: [[df[key].iloc[num] for df in metadatastruct.A_dict_non_sis.values() if len(df) - 1
                       >= num] for num in range(max([len(DF) for DF in metadatastruct.A_dict_non_sis.values()]))] for
                key in metadatastruct.A_dict_non_sis['Non-sister_Trace_A_0'].columns}
            columns_pooled_over_cycles_B_non_sis = {
                key: [[df[key].iloc[num] for df in metadatastruct.B_dict_non_sis.values() if len(df) - 1
                       >= num] for num in range(max([len(DF) for DF in metadatastruct.B_dict_non_sis.values()]))] for
                key in metadatastruct.B_dict_non_sis['Non-sister_Trace_B_0'].columns}
            x_values = np.append(x_valuesA, x_valuesB)
            y_values = np.append(y_valuesA, y_valuesB)
            pooled_phi_star_A, pooled_beta_A = lsqm(x=x_valuesA,
                                                    y=y_valuesA,
                                                    cov=False)
            pooled_phi_star_B, pooled_beta_B = lsqm(x=x_valuesB,
                                                    y=y_valuesB,
                                                    cov=False)
            pooled_phi_star, pooled_beta = lsqm(x=x_values,
                                                y=y_values,
                                                cov=False)
            pooled_data_dict = {
                                'pooled_phi_star_A': pooled_phi_star_A,
                                'pooled_beta_A': pooled_beta_A,
                                'pooled_phi_star_B': pooled_phi_star_B,
                                'pooled_beta_B': pooled_beta_B,
                                'pooled_phi_star': pooled_phi_star,
                                'pooled_beta': pooled_beta,
                                'columns_pooled_over_cycles_A_non_sis': columns_pooled_over_cycles_A_non_sis,
                                'columns_pooled_over_cycles_B_non_sis': columns_pooled_over_cycles_B_non_sis

            }

        return phi_star_A_array, phi_star_B_array, beta_A_array, beta_B_array, pooled_data_dict

    elif ensemble == 'Both': # A VS B in Sister/NonSister Cells (POOLED -- Equal Amount from Both)
        # A_dict_non_sis_sampled = random.sample(list(metadatastruct.A_dict_non_sis.values()), min(len(metadatastruct.A_dict_sis.values()), len(metadatastruct.A_dict_non_sis.values())))
        # A_dict_sis_sampled = random.sample(list(metadatastruct.A_dict_sis.values()), min(len(metadatastruct.A_dict_sis.values()), len(metadatastruct.A_dict_non_sis.values())))
        # B_dict_non_sis_sampled = random.sample(list(metadatastruct.B_dict_non_sis.values()), min(len(metadatastruct.B_dict_sis.values()),len(metadatastruct.B_dict_non_sis.values())))
        # B_dict_sis_sampled = random.sample(list(metadatastruct.B_dict_sis.values()), min(len(metadatastruct.B_dict_sis.values()),len(metadatastruct.B_dict_non_sis.values())))

        # A_dict_non_sis_sampled = dict(random.sample(list(metadatastruct.A_dict_non_sis.items()),
        #                                        min(len(metadatastruct.A_dict_sis.values()),
        #                                            len(metadatastruct.A_dict_non_sis.values()))))
        # A_dict_sis_sampled = dict(random.sample(list(metadatastruct.A_dict_sis.items()),
        #                                    min(len(metadatastruct.A_dict_sis.values()),
        #                                        len(metadatastruct.A_dict_non_sis.values()))))
        # B_dict_non_sis_sampled = dict(random.sample(list(metadatastruct.B_dict_non_sis.items()),
        #                                        min(len(metadatastruct.B_dict_sis.values()),
        #                                            len(metadatastruct.B_dict_non_sis.values()))))
        # B_dict_sis_sampled = dict(random.sample(list(metadatastruct.B_dict_sis.items()),
        #                                    min(len(metadatastruct.B_dict_sis.values()),
        #                                        len(metadatastruct.B_dict_non_sis.values()))))
        #
        # # SEPARATING A VS B NOT CARING FOR SIS OR NON SIS
        # A_dict_both = A_dict_non_sis_sampled
        # A_dict_both.update(A_dict_sis_sampled)
        # B_dict_both = B_dict_non_sis_sampled
        # B_dict_both.update(B_dict_sis_sampled)

        # SEPARATING A VS B NOT CARING FOR SIS OR NON SIS
        A_dict_both = metadatastruct.A_dict_both
        B_dict_both = metadatastruct.B_dict_both

        # if not len(A_dict_non_sis_sampled) == len(B_dict_non_sis_sampled) == len(A_dict_sis_sampled) == len(B_dict_sis_sampled):
        #     IOError('something went wrong, not all samples have the same amount')
        for dfA, dfB in zip(A_dict_both.values(), B_dict_both.values()):
            phi_star_A, beta_A = lsqm(x=np.log(np.divide(dfA['length_birth'], metadatastruct.x_avg)),
                                      y=np.multiply(dfA['generationtime'], dfA['growth_length']),
                                      cov=False)
            phi_star_B, beta_B = lsqm(x=np.log(np.divide(dfB['length_birth'], metadatastruct.x_avg)),
                                      y=np.multiply(dfB['generationtime'], dfB['growth_length']),
                                      cov=False)
            if pooled:
                x_valuesA = np.append(x_valuesA, np.log(np.divide(dfA['length_birth'], metadatastruct.x_avg)))
                x_valuesB = np.append(x_valuesB, np.log(np.divide(dfB['length_birth'], metadatastruct.x_avg)))
                y_valuesA = np.append(y_valuesA, np.multiply(dfA['generationtime'], dfA['growth_length']))
                y_valuesB = np.append(y_valuesB, np.multiply(dfB['generationtime'], dfB['growth_length']))
            phi_star_A_array.append(phi_star_A)
            phi_star_B_array.append(phi_star_B)
            beta_A_array.append(beta_A)
            beta_B_array.append(beta_B)

        if graph:
            # PLOTS BETAS

            # ONLY IN WINDOWS WITHOUT SEABORN
            # plt.figure(figsize=(20, 20))

            # plt.scatter(dfA['length_birth'])
            t = np.arange(-2.5, 2.5)
            s = [[phi + beta * ti for phi, beta in zip(phi_star_A_array, beta_A_array)] for ti in t]
            s1 = [[phi + beta * ti for phi, beta in zip(phi_star_B_array, beta_B_array)] for ti in t]
            ff = plt.plot(s, color='r', label='Sister Cells')
            gg = plt.plot(s1, color='b', label='Non-Sister Cells')
            plt.title(r'Individual $\beta$ for Sister/Non-Sister Cells Trajectories')
            plt.xlabel(r'$ln(\frac{x_n^{(k)}}{x^*})$', fontsize=25)
            plt.ylabel(r'$\phi^{(k)}_n$', fontsize=25)
            plt.legend(handles=[ff[0], gg[0]])
            plt.show()
        if pooled:
            columns_pooled_over_cycles_A_both = {
                key: [[df[key].iloc[num] for df in metadatastruct.A_dict_both.values() if len(df) - 1
                       >= num] for num in range(max([len(DF) for DF in metadatastruct.A_dict_both.values()]))] for
                key in metadatastruct.A_dict_both[list(metadatastruct.A_dict_both.keys())[0]].columns}
            columns_pooled_over_cycles_B_both = {
                key: [[df[key].iloc[num] for df in metadatastruct.B_dict_both.values() if len(df) - 1
                       >= num] for num in range(max([len(DF) for DF in metadatastruct.B_dict_both.values()]))] for
                key in metadatastruct.B_dict_both[list(metadatastruct.B_dict_both.keys())[0]].columns}
            x_values = np.append(x_valuesA, x_valuesB)
            y_values = np.append(y_valuesA, y_valuesB)
            pooled_phi_star_A, pooled_beta_A = lsqm(x=x_valuesA,
                                                    y=y_valuesA,
                                                    cov=False)
            pooled_phi_star_B, pooled_beta_B = lsqm(x=x_valuesB,
                                                    y=y_valuesB,
                                                    cov=False)
            pooled_phi_star, pooled_beta = lsqm(x=x_values,
                                                y=y_values,
                                                cov=False)
            pooled_data_dict = {
                                'pooled_phi_star_A': pooled_phi_star_A,
                                'pooled_beta_A': pooled_beta_A,
                                'pooled_phi_star_B': pooled_phi_star_B,
                                'pooled_beta_B': pooled_beta_B,
                                'pooled_phi_star': pooled_phi_star,
                                'pooled_beta': pooled_beta,
                                'columns_pooled_over_cycles_A_both': columns_pooled_over_cycles_A_both,
                                'columns_pooled_over_cycles_B_both': columns_pooled_over_cycles_B_both
                                }

        return phi_star_A_array, phi_star_B_array, beta_A_array, beta_B_array, pooled_data_dict





