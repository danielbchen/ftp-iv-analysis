import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main():
    """
    """

    # Question 1:
    admin = admin_data_loader()
    survey = survey_data_loader()
    df = ftp_merger(admin, survey)
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
                              'COUNTS': df['fmi2_x'].value_counts()})

    sum_table = sum_table.append({'CATEGORY': 'Valid Responses', 
                                  'COUNTS': len(df['fmi2_x'])}, 
                                  ignore_index=True)

    return sum_table


def new_dummy_creator(dataframe):
    """Creates a new treatment variable. 1 for those who believed in time limit. 
    0 for those who did not. Everyone else is dropped. 
    """

    df = dataframe.copy()

    df = df[(df['fmi2_x'] == 1) | (df['fmi2_x'] == 2)]

    df['NEW_TREAT'] = [1 if val == 1 else 0 for val in df['fmi2_x']]

    return df


def xtab_generator(dataframe):
    """Generates crosstabs of original dummy variable vs. new dummy variable."""

    df = dataframe.copy()

    tabs = pd.crosstab(index=df['e_x'], columns=df['NEW_TREAT'], 
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
        'male_x',
        'agelt20_x',
        'age2534_x',
        'age3544_x',
        'agege45_x',
        'black_x',
        'hisp_x',
        'otheth_x',
        'martog_x',
        'marapt_x',
        'nohsged_x',
        'applcant_x'
    ]
    df[columns] = df[columns].fillna(df[columns].mean())

    return df


def time_limit_ols(dataframe):
    """Estimates effect of believing in the time limit on welfare receipt."""

    df = dataframe.copy()

    dependent_variables = [
        'vrecc217_x',
        'vrecc2t5_x',
        'vrecc6t9_x',
        'vrec1013_x',
        'vrec1417_x',
    ]

    covariates = [
        'NEW_TREAT',
        'male_x',
        'agelt20_x',
        'age2534_x',
        'age3544_x',
        'agege45_x',
        'black_x',
        'hisp_x',
        'otheth_x',
        'martog_x',
        'marapt_x',
        'nohsged_x',
        'applcant_x',
        'yremp_x',
        'emppq1_x',
        'yrearn_x',
        'yrearnsq_x',
        'pearn1_x',
        'recpc1_x',
        'yrrec_x',
        'yrkrec_x',
        'rfspc1_x',
        'yrrfs_x',
        'yrkrfs_x',
    ]

    right_hand_side =  ' + '.join([variable for variable in covariates])
    formulas = [dep_var + ' ~ ' + right_hand_side for dep_var in dependent_variables]

    regressions = [smf.ols(formula=formula, data=df).fit() for formula in formulas]

    df['vrecc217']
