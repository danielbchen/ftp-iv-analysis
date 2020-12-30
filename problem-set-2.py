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
    print('Question 1:\n', 
          len(df), 
          'of the original sample members from the admin records remain in the survey data.')


def admin_data_loader():
    """
    Loads ftp administrative dataset. 
    """

    path = '/Users/danielchen/Desktop/UChicago/Year Two/Autumn 2020/Program Evaluation/Problem Sets/Problem Set 2/ftp_ar.dta'
    df = pd.read_stata(path)

    return df


def survey_data_loader():
    """
    Loads ftp survey dataset.
    """

    path = '/Users/danielchen/Desktop/UChicago/Year Two/Autumn 2020/Program Evaluation/Problem Sets/Problem Set 1/ftp_srv.dta'
    df = pd.read_stata(path)

    return df


def ftp_merger(dataframe1, dataframe2):
    """
    Merges the two ftp datasets. 
    """

    df = pd.merge(dataframe1, dataframe2, on='sampleid')

    return df