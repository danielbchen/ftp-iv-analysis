from linearmodels import IV2SLS
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import statsmodels.formula.api as smf


def main():
    """
    """

    # Question 1:
    admin = admin_data_loader()
    survey = survey_data_loader()
    df = ftp_merger(admin, survey)
    df = drop_y_columns(df)
    df = colunm_renamer(df)
    print('QUESTION 1: \n', 
          len(df), 
          'of the original sample members from the admin records remain in the survey data.')

    print('\n\n')

    # Question 2: 
    sum_stats = summary_stats(df)
    print('QUESTION 2: \n',
          'Please find a table below summarizing the counts: \n\n',
          sum_stats)

    print('\n\n')

    # Question 3:
    df = new_dummy_creator(df) 
    xtabs = xtab_generator(df)
    print('QUESTION 3: \n',
          'Cross-tabluation of original assignment vs. belief in time limit: \n\n',
          xtabs)

    print('\n\n')

    # Question 4:
    df = mean_imputer(df)
    time_limit_results = time_limit_ols(df)
    print('QUESTION 4: \n',
          'Effect of believing in the time limit on welfare receipt: \n\n',
          time_limit_results)

    print('\n\n')

    # Question 5:
    corr = correlation_test(df, 'e', 'TLyes')
    print('QUESTION 5: \n',
          'Correlation between e and TLyes:',
          corr)

    print('\n\n')

    # Question 6:
    iv_results = iv_estimation(df)
    print('QUESTION 6: \n',
          'Instrumental Variable Estimates: \n\n',
          iv_results)

def admin_data_loader():
    """Loads ftp administrative dataset."""

    path = '/Users/danielchen/Desktop/UChicago/Year Two/Autumn 2020/Program Evaluation/Problem Sets/Problem Set 2/ftp_ar.dta'
    df = pd.read_stata(path)

    return df


def survey_data_loader():
    """Loads ftp survey dataset."""

    path = '/Users/danielchen/Desktop/UChicago/Year Two/Autumn 2020/Program Evaluation/Problem Sets/Problem Set 1/ftp_srv.dta'
    df = pd.read_stata(path)

    return df


def ftp_merger(dataframe1, dataframe2):
    """Merges the two ftp datasets."""

    df = pd.merge(dataframe1, dataframe2, on='sampleid')

    return df


def drop_y_columns(dataframe):
    """Drops duplicate columns."""

    df = dataframe.copy()

    cols_to_drop = [col for col in df if col.endswith('_y')]
    df = df.drop(cols_to_drop, 1)

    return df


def colunm_renamer(dataframe):
    """Removes '_x' from column names after merging."""

    df = dataframe.copy()

    col_names = [col for col in df.columns.values]
    new_names = [col_name[:-2] if col_name.endswith('_x') else col_name for col_name in col_names]
    df.columns = new_names

    return df


def summary_stats(dataframe):
    """Returns a dataframe showing how many people believed in the time limit 
    vs. how many people did not. 
    """

    df = dataframe.copy()

    categories = [
        'Believed Subject to Time Limit',
        "Didn't Believe Subject to Time Limit",
        "Don't Know"
    ]

    sum_table = pd.DataFrame({'CATEGORY': categories, 
                              'COUNTS': df['fmi2'].value_counts()})

    sum_table = sum_table.append({'CATEGORY': 'Valid Responses', 
                                  'COUNTS': len(df['fmi2'])}, 
                                  ignore_index=True)

    return sum_table


def new_dummy_creator(dataframe):
    """Creates a new treatment variable. 1 for those who believed in time limit. 
    0 for those who did not. Everyone else is dropped. 
    """

    df = dataframe.copy()

    df = df[(df['fmi2'] == 1) | (df['fmi2'] == 2)]

    df['TLyes'] = [1 if val == 1 else 0 for val in df['fmi2']]

    return df


def xtab_generator(dataframe):
    """Generates crosstabs of original dummy variable vs. new dummy variable."""

    df = dataframe.copy()

    tabs = pd.crosstab(index=df['e'], columns=df['TLyes'], 
                       margins=True, margins_name='Total',
                       rownames=['Original Treatment'],
                       colnames=['Time Limit Belief'])

    return tabs


def mean_imputer(dataframe):
    """Takes in the admin and survey merged data and returns a dataframe with
    covariate columns containing NA values filled with the mean of that column. 
    """

    df = dataframe.copy()

    columns = [
        'male',
        'agelt20',
        'age2534',
        'age3544',
        'agege45',
        'black',
        'hisp',
        'otheth',
        'martog',
        'marapt',
        'nohsged',
        'applcant'
    ]
    df[columns] = df[columns].fillna(df[columns].mean())

    return df


def paramater_retriever(list_object, parameter):
    """Retrieves a specified parameter from ols regression output."""

    if parameter == 'coefficients':
        values = [item.params for item in list_object]
    elif parameter == 'standard error':
        values = [item.bse for item in list_object]
    elif parameter == 'pvalues':
        values = [item.pvalues for item in list_object]
    elif parameter == 'conf_int_low':
        values = [item.conf_int()[0] for item in list_object]
    elif parameter == 'conf_int_high':
        values = [item.conf_int()[1] for item in list_object]
    
    values = [value[1] for value in values]

    return values


def time_limit_ols(dataframe):
    """Runs OLS to estimate effect of believing in the time limit on welfare
    receipt during years 1-4 of the sample period.
    """

    df = dataframe.copy()

    welfare_vars = [
        'vrecc217',
        'vrecc2t5',
        'vrecc6t9',
        'vrec1013',
        'vrec1417',
    ]
    ind_vars = [
        'TLyes',
        'male',
        'agelt20',
        'age2534',
        'age3544',
        'agege45',
        'black',
        'hisp',
        'otheth',
        'martog',
        'marapt',
        'nohsged',
        'applcant',
        'yremp',
        'emppq1',
        'yrearn',
        'yrearnsq',
        'pearn1',
        'recpc1',
        'yrrec',
        'yrkrec',
        'rfspc1',
        'yrrfs',
        'yrkrfs',
    ]
    right_hand_side = ' + '.join([var for var in ind_vars])

    formulas = [var + ' ~ ' + right_hand_side for var in welfare_vars]
    regressions = [smf.ols(formula=formula, data=df).fit()
                   for formula in formulas]

    ols_results = pd.DataFrame({
        'Welfare_Variable': welfare_vars,
        'Coefficient': paramater_retriever(regressions, 'coefficients'),
        'Std_Error': paramater_retriever(regressions, 'standard error'),
        'p_value': paramater_retriever(regressions, 'pvalues'),
        'Conf_Low': paramater_retriever(regressions, 'conf_int_low'),
        'Conf_High': paramater_retriever(regressions, 'conf_int_high')})

    ols_results['t_stat'] = ols_results['Coefficient'] / \
        ols_results['Std_Error']

    ols_results = ols_results[[
        'Welfare_Variable',
        'Coefficient',
        'Std_Error',
        't_stat',
        'p_value',
        'Conf_Low',
        'Conf_High'
    ]]

    return ols_results


def correlation_test(dataframe, var1, var2):
    """Returns Pearson's R coefficient."""

    df = dataframe.copy()

    correlation, _ = pearsonr(df[var1], df[var2])

    return correlation


def iv_estimation(dataframe):
    """Takes in a dataframe and returns estimates where random assignment 
    serves as an instrument for believing in the time limit. 
    """

    df = dataframe.copy()

    dependent_variables = [
        # Employment variables
        'vempq217',
        'vempq2t5',
        'vempq6t9',
        'vemp1013',
        'vemp1417',
        # Welfare variables
        'vrecc217',
        'vrecc2t5',
        'vrecc6t9',
        'vrec1013',
        'vrec1417',
        # Income variables
        'tinc217',
        'tinc2t5',
        'tinc6t9',
        'tinc1013',
        'tinc1417'
    ]

    df['CONSTANT'] = 1
    controls = [
        'male',
        'agelt20',
        'age2534',
        'age3544',
        'agege45',
        'black',
        'hisp',
        'otheth',
        'martog',
        'marapt',
        'nohsged',
        'applcant',
        'yremp',
        'emppq1',
        'yrearn',
        'yrearnsq',
        'pearn1',
        'recpc1',
        'yrrec',
        'yrkrec',
        'rfspc1',
        'yrrfs',
        'yrkrfs',
        'CONSTANT'
    ]

    iv_models = [IV2SLS(df[dep_var], df[controls], df['TLyes'], df['e']).fit() for dep_var in dependent_variables]

    iv_results = pd.DataFrame({'Variable': dependent_variables,
                               'Coefficient': [model.params[-1] for model in iv_models],
                               'Std_Error': [model.std_errors[-1] for model in iv_models],
                               't_stat': [model.tstats[-1] for model in iv_models],
                               'p_value': [model.pvalues[-1] for model in iv_models]})

    return iv_results
