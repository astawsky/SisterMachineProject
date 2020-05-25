
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import scipy.stats as stats


def output_heatmap(df, title, x_labels, y_labels):
    sns.heatmap(df, annot=True, xticklabels=x_labels, yticklabels=y_labels)
    plt.title(title)
    plt.show()


'''
df_columns is the collection of the vectors that will appear as the column y-tick marks
df_rows is the collection of the vectors that will appear as the row x-tick marks
'''
def correlation_dataframe(df_columns, df_rows):

    new_df = pd.DataFrame(data=np.ones([len(df_rows.columns), len(df_columns.columns)]), columns=df_columns.columns)
    new_df.rename(index=dict(zip(np.arange(len(df_rows.columns)), df_rows.columns)), inplace=True)
    for row in df_rows.columns:
        for column in df_columns.columns:
            new_df[column].loc[row] = stats.pearsonr(df_rows[row], df_columns[column])[0]

    return new_df


if __name__ == '__main__':
    # Import the Refined Data
    pickle_in = open("metastructdata_old.pickle", "rb")
    old_struct = pickle.load(pickle_in)
    pickle_in.close()

    # Import the Refined Data
    pickle_in = open("metastructdata.pickle", "rb")
    struct = pickle.load(pickle_in)
    pickle_in.close()

    columns_m = ['generationtime_m', 'length_birth_m', 'length_final_m', 'growth_length_m', 'division_ratios__f_n_m', 'phi_m']
    columns_d = ['generationtime_d', 'length_birth_d', 'length_final_d', 'growth_length_d', 'division_ratios__f_n_d', 'phi_d']

    # transfer to the units and then log normalize the lengths
    m_d_dependance = struct.m_d_dependance_units
    m_d_dependance['length_birth_m'] = np.log(m_d_dependance['length_birth_m'] / struct.x_avg)
    m_d_dependance['length_final_m'] = np.log(m_d_dependance['length_final_m'] / struct.x_avg)
    m_d_dependance['length_birth_d'] = np.log(m_d_dependance['length_birth_d'] / struct.x_avg)
    m_d_dependance['length_final_d'] = np.log(m_d_dependance['length_final_d'] / struct.x_avg)
    labels_same = [col[:-2] for col in columns_m] # can be columns_d
    labels_same[1], labels_same[2] = 'log norm of LB', 'log norm of LF'  # change the names of the labels_same
    columns_m = ['generationtime_m', 'length_birth_m', 'length_final_m', 'growth_length_m', 'division_ratios__f_n_m', 'phi_m']
    columns_d = ['generationtime_d', 'length_birth_d', 'length_final_d', 'growth_length_d', 'division_ratios__f_n_d', 'phi_d']
    labels_m = columns_m
    labels_d = columns_d
    labels_m[1], labels_m[2] = 'log norm of LB_m', 'log norm of LF_m'  # change the names of the labels_m
    labels_d[1], labels_d[2] = 'log norm of LB_d', 'log norm of LF_d'  # change the names of the labels_d
    columns_m = ['generationtime_m', 'length_birth_m', 'length_final_m', 'growth_length_m', 'division_ratios__f_n_m', 'phi_m']
    columns_d = ['generationtime_d', 'length_birth_d', 'length_final_d', 'growth_length_d', 'division_ratios__f_n_d', 'phi_d']
    trap_avg_columns = ['trap_average_generationtime', 'trap_average_length_birth', 'trap_average_length_final',
                                     'trap_average_growth_length', 'trap_average_division_ratios__f_n', 'trap_average_phi']
    labels_trap_avg = trap_avg_columns
    labels_trap_avg[1], labels_trap_avg[2] = 'trap_avg log norm of LB', 'trap_avg log norm of LF' # change the names of the labels_trap_avg
    columns_m = ['generationtime_m', 'length_birth_m', 'length_final_m', 'growth_length_m', 'division_ratios__f_n_m', 'phi_m']
    columns_d = ['generationtime_d', 'length_birth_d', 'length_final_d', 'growth_length_d', 'division_ratios__f_n_d', 'phi_d']
    trap_avg_columns = ['trap_average_generationtime', 'trap_average_length_birth', 'trap_average_length_final',
                        'trap_average_growth_length', 'trap_average_division_ratios__f_n', 'trap_average_phi']

    # same-cell correlations

    df = correlation_dataframe(struct.m_d_dependance[columns_m], struct.m_d_dependance[columns_m]) # can be columns_d

    output_heatmap(df, title='Same-Cell Correlations (Symmetric)', x_labels=labels_same, y_labels=labels_same)

    # mother and daughter
    symbol_labels_m = [r'$\tau_{mom}$', r'$ln(\frac{x(0)}{x^*})_{mom}$', r'$ln(\frac{x(\tau)}{x^*})_{mom}$']
    symbol_labels_d = [r'$\tau_{daughter}$', r'$ln(\frac{x(0)}{x^*})_{daughter}$', r'$ln(\frac{x(\tau)}{x^*})_{daughter}$']

    df = correlation_dataframe(struct.m_d_dependance[columns_m[:3]], struct.m_d_dependance[columns_d[:3]])

    output_heatmap(df, title='mother daughter correlations', x_labels=symbol_labels_m[:3], y_labels=symbol_labels_d[:3])

    # trap_average and bacteria inside that trap
    df = correlation_dataframe(struct.m_d_dependance[columns_m], struct.m_d_dependance[trap_avg_columns])

    output_heatmap(df, title='Bacteria and trace-specific-average Correlations', x_labels=labels_same,
                   y_labels=labels_trap_avg)
