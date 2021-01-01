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
    xtabs = treat_dummy_xtab(df)
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


def treat_dummy_xtab(dataframe):
    """Creates a new treatment variable. 1 for those who believed in time limit. 
    0 for those who did not. Everyone else is dropped. 
    Tabulates across original treatment variable. 
    """

    df = dataframe.copy()

    df = df[(df['fmi2_x'] == 1) | (df['fmi2_x'] == 2)]

    df['NEW_TREAT'] = [1 if val == 1 else 0 for val in df['fmi2_x']]

    tabs = pd.crosstab(index=df['e_x'], columns=df['NEW_TREAT'], 
                       margins=True, margins_name='Total',
                       rownames=['Original Treatment'],
                       colnames=['Time Limit Belief'])

    return tabs


def mean_imputer(dataframe):
    """Fills missing values in covariate columns with column mean."""

    df = dataframe.copy()