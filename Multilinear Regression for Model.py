import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression as LR
import matplotlib.pyplot as plt
from correlation_dataframe_and_heatmap import output_heatmap
from correlation_dataframe_and_heatmap import correlation_dataframe
import pickle
import scipy.stats as stats
import scipy.linalg as linalg
from sklearn.linear_model import LinearRegression
import pingouin as pg
import itertools

"""
Partial Correlation in Python (clone of Matlab's partialcorr)
This uses the linear regression approach to compute the partial 
correlation (might be slow for a huge number of variables). The 
algorithm is detailed here:
    http://en.wikipedia.org/wiki/Partial_correlation#Using_linear_regression
Taking X and Y two variables of interest and Z the matrix with all the variable minus {X, Y},
the algorithm can be summarized as
    1) perform a normal linear least-squares regression with X as the target and Z as the predictor
    2) calculate the residuals in Step #1
    3) perform a normal linear least-squares regression with Y as the target and Z as the predictor
    4) calculate the residuals in Step #3
    5) calculate the correlation coefficient between the residuals from Steps #2 and #4; 
    The result is the partial correlation between X and Y while controlling for the effect of Z
Date: Nov 2014
Author: Fabian Pedregosa-Izquierdo, f@bianp.net
Testing: Valentina Borghesani, valentinaborghesani@gmail.com
"""


def partial_corr(C):
    """
    Returns the sample linear partial correlation coefficients between pairs of variables in C, controlling
    for the remaining variables in C.
    Parameters
    ----------
    C : array-like, shape (n, p)
        Array with the different variables. Each column of C is taken as a variable
    Returns
    -------
    P : array-like, shape (p, p)
        P[i, j] contains the partial correlation of C[:, i] and C[:, j] controlling
        for the remaining variables in C.
    """

    C = np.asarray(C)
    p = C.shape[1]
    P_corr = np.zeros((p, p), dtype=np.float)
    for i in range(p):
        P_corr[i, i] = 1
        for j in range(i + 1, p):
            idx = np.ones(p, dtype=np.bool)
            idx[i] = False
            idx[j] = False
            beta_i = linalg.lstsq(C[:, idx], C[:, j])[0]
            beta_j = linalg.lstsq(C[:, idx], C[:, i])[0]

            res_j = C[:, j] - C[:, idx].dot(beta_i)
            res_i = C[:, i] - C[:, idx].dot(beta_j)

            corr = stats.pearsonr(res_i, res_j)[0]
            P_corr[i, j] = corr
            P_corr[j, i] = corr

    return P_corr


def LinearRegressionMatrices():
    pickle_name = 'metastructdata'
    # Import the Refined Data
    pickle_in = open(pickle_name + ".pickle", "rb")
    struct = pickle.load(pickle_in)
    pickle_in.close()

    m_d_dependance_units = struct.m_d_dependance_units

    # columns_m = ['length_birth_m', 'phi_m', 'growth_length_m']
    #
    # columns_d = ['length_birth_d', 'phi_d', 'growth_length_d']
    #
    # corr_mat = partial_corr(m_d_dependance_units[columns_m])
    #
    # output_heatmap(corr_mat, 'partial correlations', x_labels=columns_m, y_labels=columns_m)
    #
    # corr_mat = partial_corr(m_d_dependance_units[columns_d])
    #
    # output_heatmap(corr_mat, 'partial correlations', x_labels=columns_d, y_labels=columns_d)

    print('Using Global Averages:')

    # # log normalize the sizes, y
    # m_d_dependance_units['length_birth_m'] = np.log(m_d_dependance_units['length_birth_m']) - np.log(np.mean(m_d_dependance_units['length_birth_m']))
    # m_d_dependance_units['length_birth_d'] = np.log(m_d_dependance_units['length_birth_d']) - np.log(np.mean(m_d_dependance_units['length_birth_d']))
    #
    # # phi to delta phi
    # m_d_dependance_units['phi_m'] = m_d_dependance_units['phi_m'] - np.mean(m_d_dependance_units['phi_m'])
    # m_d_dependance_units['phi_d'] = m_d_dependance_units['phi_d'] - np.mean(m_d_dependance_units['phi_d'])
    #
    # # alpha to delta alpha
    # m_d_dependance_units['growth_length_m'] = m_d_dependance_units['growth_length_m'] - np.mean(m_d_dependance_units['growth_length_m'])
    # m_d_dependance_units['growth_length_d'] = m_d_dependance_units['growth_length_d'] - np.mean(m_d_dependance_units['growth_length_d'])

    # corr_mat = partial_corr(m_d_dependance_units[columns_m])
    #
    # output_heatmap(corr_mat, 'partial correlations', x_labels=columns_m, y_labels=columns_m)
    #
    # corr_mat = partial_corr(m_d_dependance_units[columns_d])
    #
    # output_heatmap(corr_mat, 'partial correlations', x_labels=columns_d, y_labels=columns_d)
    #
    # exit()

    # print('Using Trap Averages:')
    #
    # # log normalize the sizes, y
    # m_d_dependance_units['length_birth_m'] = np.log(m_d_dependance_units['length_birth_m']) - np.log(m_d_dependance_units['trap_average_length_birth'])
    # m_d_dependance_units['length_birth_d'] = np.log(m_d_dependance_units['length_birth_d']) - np.log(m_d_dependance_units['trap_average_length_birth'])
    #
    # # phi to delta phi
    # m_d_dependance_units['phi_m'] = m_d_dependance_units['phi_m'] - m_d_dependance_units['trap_average_phi']
    # m_d_dependance_units['phi_d'] = m_d_dependance_units['phi_d'] - m_d_dependance_units['trap_average_phi']
    #
    # # alpha to delta alpha
    # m_d_dependance_units['growth_length_m'] = m_d_dependance_units['growth_length_m'] - m_d_dependance_units['trap_average_growth_length']
    # m_d_dependance_units['growth_length_d'] = m_d_dependance_units['growth_length_d'] - m_d_dependance_units['trap_average_growth_length']

    # group them in pairs
    mother_pairs = m_d_dependance_units[['length_birth_m', 'phi_m']] #,'growth_length_m'

    # multilinear regression
    # # for phi first
    # reg = LinearRegression().fit(mother_pairs, m_d_dependance_units['phi_d'])
    #
    # print('score for predicting phi_d \n', reg.score(mother_pairs, m_d_dependance_units['phi_d']))
    #
    # phi_coeffs = reg.coef_
    #
    # print('coefficients for predicting phi_d \n', phi_coeffs)
    #
    # print('intercepts for predicting phi_d \n', reg.intercept_)

    # # for y second
    reg = LinearRegression().fit(mother_pairs, m_d_dependance_units['length_birth_d'])

    print('score for predicting length_birth_d \n', reg.score(mother_pairs, m_d_dependance_units['length_birth_d']))

    length_birth_coeffs = reg.coef_

    print('coefficients for predicting length_birth_d \n', length_birth_coeffs)

    print('intercepts for predicting length_birth_d \n', reg.intercept_)

    # for alpha third
    reg = LinearRegression().fit(mother_pairs, m_d_dependance_units['phi_d'])

    print('score for predicting growth_length_d \n', reg.score(mother_pairs, m_d_dependance_units['phi_d']))

    growth_length_coeffs = reg.coef_

    print('coefficients for predicting growth_length_d \n', growth_length_coeffs)

    print('intercepts for predicting growth_length_d \n', reg.intercept_)

    matrix = np.vstack([length_birth_coeffs, growth_length_coeffs]) # phi_coeffs,

    print('matrix \n', matrix)


def semi_partial_correlations_same_cell(what_average='Global'):
    pickle_name = 'metastructdata'
    # Import the Refined Data
    pickle_in = open(pickle_name + ".pickle", "rb")
    struct = pickle.load(pickle_in)
    pickle_in.close()

    m_d_dependance_units = struct.m_d_dependance_units

    """ From here we will take our X, Y, X_covars and Y_covars """
    columns_m = pd.Series(data=['generationtime_m', 'length_birth_m', 'growth_length_m', 'division_ratios__f_n_m', 'phi_m'],
                          index=['generationtime_m', 'length_birth_m', 'growth_length_m', 'division_ratios__f_n_m', 'phi_m']) # no final length
    columns_d = pd.Series(data=['generationtime_d', 'length_birth_d', 'growth_length_d', 'division_ratios__f_n_d', 'phi_d'],
                          index=['generationtime_d', 'length_birth_d', 'growth_length_d', 'division_ratios__f_n_d', 'phi_d']) # no final length

    """ First we choose either the Global Average or the Trap Average to subtract """
    if what_average == 'Global':
        print('Using Global Averages!')

        # log normalize the sizes, y
        m_d_dependance_units['length_birth_m'] = np.log(m_d_dependance_units['length_birth_m']) - np.log(np.mean(m_d_dependance_units['length_birth_m']))
        m_d_dependance_units['length_birth_d'] = np.log(m_d_dependance_units['length_birth_d']) - np.log(np.mean(m_d_dependance_units['length_birth_d']))

        # phi to delta phi
        m_d_dependance_units['phi_m'] = m_d_dependance_units['phi_m'] - np.mean(m_d_dependance_units['phi_m'])
        m_d_dependance_units['phi_d'] = m_d_dependance_units['phi_d'] - np.mean(m_d_dependance_units['phi_d'])

        # alpha to delta alpha
        m_d_dependance_units['growth_length_m'] = m_d_dependance_units['growth_length_m'] - np.mean(m_d_dependance_units['growth_length_m'])
        m_d_dependance_units['growth_length_d'] = m_d_dependance_units['growth_length_d'] - np.mean(m_d_dependance_units['growth_length_d'])

        # generationtime to delta generationtime
        m_d_dependance_units['generationtime_m'] = m_d_dependance_units['generationtime_m'] - np.mean(m_d_dependance_units['generationtime_m'])
        m_d_dependance_units['generationtime_d'] = m_d_dependance_units['generationtime_d'] - np.mean(m_d_dependance_units['generationtime_d'])

        # division ratio to delta division ratio
        m_d_dependance_units['division_ratios__f_n_m'] = m_d_dependance_units['division_ratios__f_n_m'] - np.mean(m_d_dependance_units['division_ratios__f_n_m'])
        m_d_dependance_units['division_ratios__f_n_d'] = m_d_dependance_units['division_ratios__f_n_d'] - np.mean(m_d_dependance_units['division_ratios__f_n_d'])
    elif what_average == 'Trap':
        print('Using Trap Averages:')

        # log normalize the sizes, y
        m_d_dependance_units['length_birth_m'] = np.log(m_d_dependance_units['length_birth_m']) - np.log(m_d_dependance_units['trap_average_length_birth'])
        m_d_dependance_units['length_birth_d'] = np.log(m_d_dependance_units['length_birth_d']) - np.log(m_d_dependance_units['trap_average_length_birth'])

        # phi to delta phi
        m_d_dependance_units['phi_m'] = m_d_dependance_units['phi_m'] - m_d_dependance_units['trap_average_phi']
        m_d_dependance_units['phi_d'] = m_d_dependance_units['phi_d'] - m_d_dependance_units['trap_average_phi']

        # alpha to delta alpha
        m_d_dependance_units['growth_length_m'] = m_d_dependance_units['growth_length_m'] - m_d_dependance_units['trap_average_growth_length']
        m_d_dependance_units['growth_length_d'] = m_d_dependance_units['growth_length_d'] - m_d_dependance_units['trap_average_growth_length']

        # generationtime to delta generationtime
        m_d_dependance_units['generationtime_m'] = m_d_dependance_units['generationtime_m'] - m_d_dependance_units['trap_average_generationtime']
        m_d_dependance_units['generationtime_d'] = m_d_dependance_units['generationtime_d'] - m_d_dependance_units['trap_average_generationtime']

        # division ratio to delta division ratio
        m_d_dependance_units['division_ratios__f_n_m'] = m_d_dependance_units['division_ratios__f_n_m'] - m_d_dependance_units['trap_average_division_ratios__f_n']
        m_d_dependance_units['division_ratios__f_n_d'] = m_d_dependance_units['division_ratios__f_n_d'] - m_d_dependance_units['trap_average_division_ratios__f_n']
    else:
        IOError('what_average has to be either Global or Trap!')

    """ Then we choose our X and Y, ie. the two variables we want to ultimately compare and get the pearson correlation coefficient of """
    # The matrix we will save at the end
    combos_df = pd.DataFrame(columns=['index', 'n', 'r', 'CI95%', 'r2', 'adj_r2', 'p-val', 'BF10', 'power', 'X', 'Y', 'X_covars', 'Y_covars'])
    # All possible X, Y combinations for the Same-Cell matrix
    same_X_Y_combos = np.array(list(itertools.combinations(columns_m, 2)))
    print('there will be ', len(same_X_Y_combos), ' combinations of X and Y')
    for num_of_combos in range(len(same_X_Y_combos)):

        # define our X and Y
        X = same_X_Y_combos[num_of_combos][0]
        Y = same_X_Y_combos[num_of_combos][1]
        print('X is ', X, ' and Y is ', Y)

        psble_cntrl_combos = []
        # loop through how many number of variables we can control for (for both X and Y b/c this is Same-Cell)
        for num_of_vars_cntrld in range(len(columns_m)-2+1):
            # the possible combinations of variables we can control for if we keep the number of variables we use constant
            psble_cntrl_combos = psble_cntrl_combos + list(itertools.combinations(columns_m.drop([X, Y]), num_of_vars_cntrld))
        # all possible combinations of variables we can control for (for both X and Y b/c this is Same-Cell)
        combined_X_Y_cntrl = list(itertools.combinations(psble_cntrl_combos + psble_cntrl_combos, 2))

        # loop over all possible X_covar and Y_covar combinations for a given X and Y
        for num_of_combos in range(len(combined_X_Y_cntrl)):
            # names of the control RVs for X and Y
            X_covar = combined_X_Y_cntrl[num_of_combos][0]
            Y_covar = combined_X_Y_cntrl[num_of_combos][1]
            # print('X_covar', len(X_covar), X_covar, type(X_covar))
            # print('Y_covar', len(Y_covar), Y_covar, type(Y_covar))

            # Deciding the X control RVs
            if len(X_covar) != 0:
                covar_X_dict = dict()
                for cntrl in range(len(X_covar)):
                    covar_X_dict.update({X_covar[cntrl] : m_d_dependance_units[X_covar[cntrl]]})
            else:
                covar_X_dict = None

            # Deciding the Y control RVs
            if len(Y_covar) != 0:
                covar_Y_dict = dict()
                for cntrl in range(len(Y_covar)):
                    covar_Y_dict.update({Y_covar[cntrl] : m_d_dependance_units[Y_covar[cntrl]]})
            else:
                covar_Y_dict = None

            # combine the covar_X_dict and the covar_Y_dict
            if covar_X_dict is not None and covar_Y_dict is not None:
                covar_dicts = covar_Y_dict
                covar_dicts.update(covar_X_dict)
            if covar_X_dict is None and covar_Y_dict is not None:
                covar_dicts = covar_Y_dict
            if covar_Y_dict is None and covar_X_dict is not None:
                covar_dicts = covar_X_dict
            if covar_X_dict is None and covar_Y_dict is None:
                super_dict = {X: m_d_dependance_units[X], Y: m_d_dependance_units[Y]}
            else:
                super_dict = dict({X: m_d_dependance_units[X], Y: m_d_dependance_units[Y]})
                super_dict.update(covar_dicts)

            # print('super dict', super_dict.keys())
            # print('covar_X_dict', covar_X_dict)
            # print('covar_Y_dict', covar_Y_dict.keys())
            # print('covar_dicts', covar_dicts.keys())

            # the data frame we are going to feed to the partial_corr function
            df = pd.DataFrame(data=super_dict)

            # correctly formatting the data to put it into the partial_corr function
            if len(X_covar) != 0 and len(Y_covar) != 0:
                X_covar = list(X_covar)
                Y_covar = list(Y_covar)
                result = df.partial_corr(x=X, y=Y, x_covar=X_covar, y_covar=Y_covar, method='pearson')
            elif len(X_covar) != 0 and len(Y_covar) == 0:
                X_covar = list(X_covar)
                Y_covar = 'None'
                result = df.partial_corr(x=X, y=Y, x_covar=X_covar, method='pearson')
            elif len(X_covar) == 0 and len(Y_covar) != 0:
                X_covar = 'None'
                Y_covar = list(Y_covar)
                result = df.partial_corr(x=X, y=Y, y_covar=Y_covar, method='pearson')
            elif len(X_covar) == 0 and len(Y_covar) == 0:
                X_covar = 'None'
                Y_covar = 'None'
                result = df.partial_corr(x=X, y=Y, method='pearson')

            # so that 'pearson' is not the index for each semi-partial correlation, may lead to problems
            result = result.reset_index()

            # Add the needed reference columns and values
            result['X'] = X
            result['Y'] = Y
            result['X_covars'] = str(X_covar) # note we will have to parse to look it up later!
            result['Y_covars'] = str(Y_covar) # note we will have to parse to look it up later!
            combos_df = combos_df.append(result, ignore_index=True)

    # save it to the struct
    if what_average == 'Global':
        struct.semi_partial_corrs_same_cell_Global = combos_df
    else:
        struct.semi_partial_corrs_same_cell_Trap = combos_df
    pickle_out = open(pickle_name + ".pickle", "wb")
    pickle.dump(struct, pickle_out)
    pickle_out.close()

    print(combos_df)


def main():
    LinearRegressionMatrices()
    # semi_partial_correlations_same_cell(what_average='Global')
    # semi_partial_correlations_same_cell(what_average='Trap')

    # pickle_name = 'metastructdata'
    # # Import the Refined Data
    # pickle_in = open(pickle_name + ".pickle", "rb")
    # struct = pickle.load(pickle_in)
    # pickle_in.close()
    #
    # semi_partial_corrs_same_cell_Global = struct.semi_partial_corrs_same_cell_Global
    #
    # print(semi_partial_corrs_same_cell_Global.columns)
    # rv = 'phi_m'
    # for rv in ['generationtime_m', 'length_birth_m', 'growth_length_m', 'division_ratios__f_n_m', 'phi_m']:
    #     mask = (semi_partial_corrs_same_cell_Global['X'] == rv) | (semi_partial_corrs_same_cell_Global['Y'] == rv)
    #     # with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    #     #     print(semi_partial_corrs_same_cell_Global[mask].sort_values('r')[['r', 'X', 'Y', 'X_covars', 'Y_covars']])
    #     plt.hist(semi_partial_corrs_same_cell_Global[mask]['r'], label=str(min(semi_partial_corrs_same_cell_Global[mask]['r']))+', '+
    #                                                                    str(max(semi_partial_corrs_same_cell_Global[mask]['r'])))
    #     plt.title(rv)
    #     plt.legend()
    #     plt.show()

    
if __name__ == '__main__':
    main()
