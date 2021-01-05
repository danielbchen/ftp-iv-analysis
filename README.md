# Project Overview

In this project, I use an instrumental variable to estimate the effect of a time limited welfare eligibility on family well-being and contrast findings to estimation via ordinary least squares.

Data are from the Family Transition Program (FTP). FTP was the first welfare reform initiative in which some families reached a time limit on their welfare eligibility and had their benefits canceled. The program took place in  Escambia County, Florida from 1994 to 1999. Key findings from the study, as well as additional background information can be found [here](https://www.mdrc.org/publication/family-transition-program).

This README contains the full analysis. A [.pdf version](https://github.com/danielbchen/problem-set-2/blob/main/(PDF%20VERSION)%20Instrumental%20Variables%20Using%20Data%20from%20Florida's%20Family%20Transition%20Program%20(FTP)%20in%201994.pdf) is also available along with the [original Jupyter Notebook](https://github.com/danielbchen/problem-set-2/blob/main/Instrumental%20Variables%20Using%20Data%20from%20Florida's%20Family%20Transition%20Program%20(FTP)%20in%201994.ipynb).

# Preparing the Data


```python
from linearmodels import IV2SLS # pip install linearmodels
import numpy as np
import os
import pandas as pd
from scipy.stats import pearsonr
import statsmodels.formula.api as smf
import warnings

warnings.filterwarnings('ignore')
path = os.path.dirname(os.path.abspath("__file__"))
```

I leverage both the administrative data (which contains official employment and income records) and survey data (which contains participant self-reported data on well-being) in the analysis. We'll need to load, merge, and clean the data before estimating treatment effects. 


```python
def admin_data_loader():
    """Loads ftp administrative dataset."""

    admin_path = os.path.join(path, 'ftp_ar.dta')
    df = pd.read_stata(admin_path)

    return df


def survey_data_loader():
    """Loads ftp survey dataset."""

    survey_path = os.path.join(path, 'ftp_srv.dta')
    df = pd.read_stata(survey_path)

    return df


def ftp_merger(dataframe1, dataframe2):
    """Merges the two ftp datasets."""

    df = pd.merge(dataframe1, dataframe2, on='sampleid')

    return df
```


```python
admin = admin_data_loader()
survey = survey_data_loader()
df = ftp_merger(admin, survey)
df
```






<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sampleid</th>
      <th>e_x</th>
      <th>cflag</th>
      <th>longtdec</th>
      <th>b_aidst</th>
      <th>gender</th>
      <th>ethnic</th>
      <th>marital</th>
      <th>afdctime</th>
      <th>afdchild</th>
      <th>...</th>
      <th>emppq1_y</th>
      <th>yrearn_y</th>
      <th>yrearnsq_y</th>
      <th>pearn1_y</th>
      <th>recpc1_y</th>
      <th>yrrec_y</th>
      <th>yrkrec_y</th>
      <th>rfspc1_y</th>
      <th>yrrfs_y</th>
      <th>yrkrfs_y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>1.0</td>
      <td>1</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>1</td>
      <td>5700</td>
      <td>32490000</td>
      <td>600</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>12</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100</td>
      <td>0</td>
      <td>NaN</td>
      <td>2</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>...</td>
      <td>1</td>
      <td>2350</td>
      <td>5522500</td>
      <td>1100</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>11</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1000</td>
      <td>0</td>
      <td>NaN</td>
      <td>1</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>1</td>
      <td>7500</td>
      <td>56250000</td>
      <td>1600</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1004</td>
      <td>0</td>
      <td>1.0</td>
      <td>1</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1</td>
      <td>9600</td>
      <td>92160000</td>
      <td>1700</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>11</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1007</td>
      <td>0</td>
      <td>1.0</td>
      <td>5</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>0</td>
      <td>400</td>
      <td>160000</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>11</td>
      <td>1</td>
      <td>1</td>
      <td>12</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1724</th>
      <td>994</td>
      <td>1</td>
      <td>1.0</td>
      <td>1</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>1</td>
      <td>35100</td>
      <td>1232010000</td>
      <td>8500</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1725</th>
      <td>995</td>
      <td>1</td>
      <td>1.0</td>
      <td>5</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>6.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>12</td>
    </tr>
    <tr>
      <th>1726</th>
      <td>996</td>
      <td>0</td>
      <td>1.0</td>
      <td>7</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1</td>
      <td>300</td>
      <td>90000</td>
      <td>100</td>
      <td>1</td>
      <td>1</td>
      <td>12</td>
      <td>1</td>
      <td>1</td>
      <td>12</td>
    </tr>
    <tr>
      <th>1727</th>
      <td>997</td>
      <td>0</td>
      <td>1.0</td>
      <td>6</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1</td>
      <td>3300</td>
      <td>10890000</td>
      <td>1800</td>
      <td>1</td>
      <td>1</td>
      <td>10</td>
      <td>1</td>
      <td>1</td>
      <td>12</td>
    </tr>
    <tr>
      <th>1728</th>
      <td>999</td>
      <td>0</td>
      <td>NaN</td>
      <td>5</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>7.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>1</td>
      <td>1100</td>
      <td>1210000</td>
      <td>900</td>
      <td>1</td>
      <td>1</td>
      <td>11</td>
      <td>1</td>
      <td>1</td>
      <td>12</td>
    </tr>
  </tbody>
</table>
<p>1729 rows × 2830 columns</p>
</div>



The merged dataframe contains duplicate columns indicated by the "_x" or "_y" suffixes attached to certain column names. The next two functions remove duplicate columns and cleans up the remaining names.


```python
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
```


```python
df = drop_y_columns(df)
df = colunm_renamer(df)
df
```





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sampleid</th>
      <th>e</th>
      <th>cflag</th>
      <th>longtdec</th>
      <th>b_aidst</th>
      <th>gender</th>
      <th>ethnic</th>
      <th>marital</th>
      <th>afdctime</th>
      <th>afdchild</th>
      <th>...</th>
      <th>nkids0</th>
      <th>nkids1</th>
      <th>nkids2</th>
      <th>nkidsge3</th>
      <th>ageykid</th>
      <th>himed</th>
      <th>hioth</th>
      <th>khimed</th>
      <th>khioth</th>
      <th>married</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>1.0</td>
      <td>1</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100</td>
      <td>0</td>
      <td>NaN</td>
      <td>2</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>11.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1000</td>
      <td>0</td>
      <td>NaN</td>
      <td>1</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1004</td>
      <td>0</td>
      <td>1.0</td>
      <td>1</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1007</td>
      <td>0</td>
      <td>1.0</td>
      <td>5</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1724</th>
      <td>994</td>
      <td>1</td>
      <td>1.0</td>
      <td>1</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1725</th>
      <td>995</td>
      <td>1</td>
      <td>1.0</td>
      <td>5</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>6.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>6.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1726</th>
      <td>996</td>
      <td>0</td>
      <td>1.0</td>
      <td>7</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1727</th>
      <td>997</td>
      <td>0</td>
      <td>1.0</td>
      <td>6</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1728</th>
      <td>999</td>
      <td>0</td>
      <td>NaN</td>
      <td>5</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>7.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>1729 rows × 1982 columns</p>
</div>



The cleaned dataframe contains 1729 rows or observations for 1729 unique families.

# Understanding Key Variables + Summary Statistics

- `e` is the treatment dummy where 0 means that a family was randomly assigned to the control group and 1 means that a family was randomly assigned to the treatment group. The treatment group had their benefits time limited - in addition to receiving a variety benefits that is beyond the analysis scope of this notebook - of while the control group did not. 
- `fmi2` is from the survey data. Families were asked if they were believed to have been subject to the time limit or not. Possible responses also include "don't know" or "no response". 

First, let's get a broad overview of how many people believed that they were subject to the time limit versus those who did not believe that they were subject to the time limit. 


```python
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
```


```python
summary_stats(df)
```





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CATEGORY</th>
      <th>COUNTS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Believed Subject to Time Limit</td>
      <td>666</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Didn't Believe Subject to Time Limit</td>
      <td>365</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Don't Know</td>
      <td>118</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Valid Responses</td>
      <td>1729</td>
    </tr>
  </tbody>
</table>
</div>



For this analysis, I will only be working with observations where the family believed that they were subject to the time limit or the opposite. I will not be working with those who don't know. I'll subset the data and create a new dummy variable for these people. The new dummy variable is referred to as `TLyes`.


```python
def new_dummy_creator(dataframe):
    """Creates a new treatment variable. 1 for those who believed in time limit. 
    0 for those who did not. Everyone else is dropped. 
    """

    df = dataframe.copy()

    df = df[(df['fmi2'] == 1) | (df['fmi2'] == 2)]

    df['TLyes'] = [1 if val == 1 else 0 for val in df['fmi2']]

    return df
```


```python
df = new_dummy_creator(df) 
```

Next, I'd like to get a sense of whether or not there was confusion around the time limit. While a family may have been randomly assigned to the treatment group, and therefore had their benefits time limited, it's possible that the family may have not believed that they were subject to the time limit and vice versa. Below, I cross-tabulate the assignment variable `e` against the self-reported belief variable `TLyes`.


```python
def xtab_generator(dataframe):
    """Generates crosstabs of original dummy variable vs. new dummy variable."""

    df = dataframe.copy()

    tabs = pd.crosstab(index=df['e'], columns=df['TLyes'], 
                       margins=True, margins_name='Total',
                       rownames=['Original Treatment'],
                       colnames=['Time Limit Belief'])

    return tabs
```


```python
xtab_generator(df)
```





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Time Limit Belief</th>
      <th>0</th>
      <th>1</th>
      <th>Total</th>
    </tr>
    <tr>
      <th>Original Treatment</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>300</td>
      <td>205</td>
      <td>505</td>
    </tr>
    <tr>
      <th>1</th>
      <td>65</td>
      <td>461</td>
      <td>526</td>
    </tr>
    <tr>
      <th>Total</th>
      <td>365</td>
      <td>666</td>
      <td>1031</td>
    </tr>
  </tbody>
</table>
</div>



From the table above, it's clear that participants were confused as to whether or not the time limit applied to their families. Of the 505 families originally assigned to control, roughly 60% were correct in identifying that the time limit did not apply to them. However, a plurality (40.6%) incorrectly thought that their benefits were time limited when then they in reality were not. Participants assigned to the treatment group better understood the guidelines dictating their benefits as 87.6% of these 562 families correctly identified that their benefits were time limited. Conversely, the remaining 12.4% thought that they had no limits when, in reality, they did. 

# Estimating Effects via Ordinary Least Squares (OLS)

It's possible to determine the effect of the time limit on well-being with a simple OLS regression. There are a number of covariates which contain missing data. I'll impute the means for each one of these controls where there is a `NaN`. 


```python
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
        'applcant',
    ]
    df[columns] = df[columns].fillna(df[columns].mean())

    return df
```


```python
df = mean_imputer(df)
```

With no more missing data, I'm going to create a helper function that retrieves paramaters such as the estimated Betas and standard errors from statsmodels's regression output. As I'm running multiple regressions, the output will be more clearly summarized in a new dataframe. 


```python
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
```


```python
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
    regressions = [smf.ols(formula=formula, data=df).fit() for formula in formulas]

    ols_results = pd.DataFrame({
        'Welfare_Variable': welfare_vars,
        'Coefficient': paramater_retriever(regressions, 'coefficients'),
        'Std_Error': paramater_retriever(regressions, 'standard error'),
        'p_value': paramater_retriever(regressions, 'pvalues'),
        'Conf_Low': paramater_retriever(regressions, 'conf_int_low'),
        'Conf_High': paramater_retriever(regressions, 'conf_int_high')})

    ols_results['t_stat'] = ols_results['Coefficient'] / ols_results['Std_Error']

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
```


```python
time_limit_ols(df)
```





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Welfare_Variable</th>
      <th>Coefficient</th>
      <th>Std_Error</th>
      <th>t_stat</th>
      <th>p_value</th>
      <th>Conf_Low</th>
      <th>Conf_High</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>vrecc217</td>
      <td>0.027352</td>
      <td>0.017174</td>
      <td>1.592680</td>
      <td>0.111546</td>
      <td>-0.006348</td>
      <td>0.061053</td>
    </tr>
    <tr>
      <th>1</th>
      <td>vrecc2t5</td>
      <td>0.034013</td>
      <td>0.019017</td>
      <td>1.788532</td>
      <td>0.073991</td>
      <td>-0.003305</td>
      <td>0.071330</td>
    </tr>
    <tr>
      <th>2</th>
      <td>vrecc6t9</td>
      <td>0.007862</td>
      <td>0.027806</td>
      <td>0.282743</td>
      <td>0.777432</td>
      <td>-0.046702</td>
      <td>0.062426</td>
    </tr>
    <tr>
      <th>3</th>
      <td>vrec1013</td>
      <td>-0.016129</td>
      <td>0.031071</td>
      <td>-0.519116</td>
      <td>0.603794</td>
      <td>-0.077101</td>
      <td>0.044842</td>
    </tr>
    <tr>
      <th>4</th>
      <td>vrec1417</td>
      <td>-0.072005</td>
      <td>0.031382</td>
      <td>-2.294493</td>
      <td>0.021967</td>
      <td>-0.133587</td>
      <td>-0.010424</td>
    </tr>
  </tbody>
</table>
</div>



The table above returns OLS results where the `Welfare_Variable` column refers to binary variables that indicate whether a family ever received benefits throughout different points over the four year study period. The `Coefficient` column contains the estimated Betas of believing in the time limit. The data suggest that in year three (`vrec1013`) and year four (`vrec1417`), families who believed in the time limit were less roughly 1% and 7% less likely to ever receive benefits relative to families who did not believe in the time limit. 

The interpretation is plausible. If I believed that my benefits were restricted after a certain period of time, then I would be less likely to rely on these benefits as time passes. Though the interpretation is logical, this doesn't guarantee that our estimates are *correct*. In the next section, I explain problems with using ordinary least squares. 

## Limitations of Ordinary Least Squares 

Ordinary Least Squares relies on the assumption that the independent variable is *exogenous* for valid estimates. However, the treatment variable is `TLyes`, and this variable is *endogenous*. As seen in the Understanding Key Variables + Summary Statistics section, there was clearly some confusion about who the time limits actually applied to. There may have been genuine confusion as to the treatment received by the treated, but families may also be *self-selecting* into treatment and control groups. 

In other words, because individuals seek to maximize their outcomes, individuals and families may self-select into the group that does not believe in time limited benefits. Arguably, it's better to receive unlimited benefits as opposed to limited benefits, so individuals may not believe in the time limit even if they were randomly assigned to the treatment group which imposed a time limit. 

In summary, OLS will not return consistent estimates of the effect of believing in the time limit. In the next section, I explore the possibility of using an instrumental variable and elaborate on the required assumptions.

# Instrumental Variable Approach

When the explanatory variable (in this case `TLyes`) is correlated with the error term, OLS will return biased results. The instrumental variable introduces a third variable that changes the explanatory variable but has no direct effect on the dependent variable. It is only through the explanatory variable that the instrument has an effect on the dependent variable. 

In this scenario, we need a third variable, Z, that influences the exogenous part of treatment `TLyes` (or the part that is not correlated with the endogenous part of the error term). By this logic, Z must be uncorrelated with the error term. 

In the following subsections, I will review the assumptions necessary to use an instrumental variable and evaluate whether or not they hold up in this context. I propose using `e` or the original experimental treatment indicator as an instrument for `TLyes`.

## 1. Exclusion Restriction 

The instrument must not be correlated with the error term and must be exogenous. Mathematically speaking, we can say that cor(z, u) = 0 and E(Y<sub>0i</sub>| Z = 1, D = 0) = E(Y<sub>0i</sub> | Z = 0, D = 0). In other words, the untreated potential outcome for an individual when the instrument is turned on is equal to the untreated potential outcome for an individual when the instrument is turned off. The instrument itself has no effect on the potential outcome conditional on the same unit being assigned to control or treatment. 

This condition is fundamentally untestable because we never observe the error term, however, it is *likely met*. Because the original experimental dummy (`e`) is exogenous through random assignment, it's plausible that the assignment itself has no direct effect on any of the dependent variables. It is only through the belief (or disbelief) in the time limit (`TLyes`) that there is an effect on the variables of interest.


## 2. Relevance


The instrument must be correlated with the treatment dummy.

In this scenario, the assignment to treatment or control must be correlated with the belief in the time limit. If the instrument changes, then we also expect the belief in the time limit to also change.

This condition is *satisfied*. Below, I test the correlation between the two variables and test against the decision rule. When using an instrumental variable, the decision rule states that our F-stat must be greater than 10 when running a first-stage regression where the endogenous variable is explained by the instrument (`TLyes` ~ `e`).


```python
def correlation_test(dataframe, var1, var2):
    """Returns Pearson's R coefficient."""

    df = dataframe.copy()

    correlation, _ = pearsonr(df[var1], df[var2])

    return correlation


def first_stage_test(dataframe):
    """Prints regression output where TLyes is explained by e."""

    df = dataframe.copy()

    output = smf.ols(formula='TLyes ~ e', data=df).fit().summary()

    print(output)
```


```python
correlation_test(df, 'e', 'TLyes')
```




    0.49181417014926404



Pearson's R is about .5, suggesting that when `e` moves, so does `TLyes`.


```python
first_stage_test(df)
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                  TLyes   R-squared:                       0.242
    Model:                            OLS   Adj. R-squared:                  0.241
    Method:                 Least Squares   F-statistic:                     328.3
    Date:                Mon, 04 Jan 2021   Prob (F-statistic):           6.72e-64
    Time:                        14:23:10   Log-Likelihood:                -559.62
    No. Observations:                1031   AIC:                             1123.
    Df Residuals:                    1029   BIC:                             1133.
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept      0.4059      0.019     21.887      0.000       0.370       0.442
    e              0.4705      0.026     18.119      0.000       0.420       0.521
    ==============================================================================
    Omnibus:                       57.896   Durbin-Watson:                   2.090
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):               31.800
    Skew:                          -0.268   Prob(JB):                     1.24e-07
    Kurtosis:                       2.328   Cond. No.                         2.64
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


From the output above, the t-stat is roughly 18. The F-stat is simply the t-stat squared, and 18<sup>2</sup> is clearly greater than 10.

# Instrumental Variable Estimation

In this section, I estimate the effect of believing in the time limit on employment, welfare receipt, and income outcomes where `e` is the instrument for `TLyes`. The dataframe below displays the results for each regression.


```python
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
```


```python
iv_estimation(df)
```





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Variable</th>
      <th>Coefficient</th>
      <th>Std_Error</th>
      <th>t_stat</th>
      <th>p_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>vempq217</td>
      <td>0.054892</td>
      <td>0.042294</td>
      <td>1.297855</td>
      <td>1.943373e-01</td>
    </tr>
    <tr>
      <th>1</th>
      <td>vempq2t5</td>
      <td>0.069899</td>
      <td>0.059466</td>
      <td>1.175448</td>
      <td>2.398153e-01</td>
    </tr>
    <tr>
      <th>2</th>
      <td>vempq6t9</td>
      <td>0.236567</td>
      <td>0.060066</td>
      <td>3.938420</td>
      <td>8.201985e-05</td>
    </tr>
    <tr>
      <th>3</th>
      <td>vemp1013</td>
      <td>0.267877</td>
      <td>0.059908</td>
      <td>4.471472</td>
      <td>7.768306e-06</td>
    </tr>
    <tr>
      <th>4</th>
      <td>vemp1417</td>
      <td>0.079870</td>
      <td>0.059562</td>
      <td>1.340957</td>
      <td>1.799343e-01</td>
    </tr>
    <tr>
      <th>5</th>
      <td>vrecc217</td>
      <td>0.002752</td>
      <td>0.035092</td>
      <td>0.078411</td>
      <td>9.375009e-01</td>
    </tr>
    <tr>
      <th>6</th>
      <td>vrecc2t5</td>
      <td>0.008770</td>
      <td>0.038303</td>
      <td>0.228966</td>
      <td>8.188954e-01</td>
    </tr>
    <tr>
      <th>7</th>
      <td>vrecc6t9</td>
      <td>-0.014991</td>
      <td>0.055916</td>
      <td>-0.268096</td>
      <td>7.886254e-01</td>
    </tr>
    <tr>
      <th>8</th>
      <td>vrec1013</td>
      <td>-0.131234</td>
      <td>0.063073</td>
      <td>-2.080680</td>
      <td>3.746320e-02</td>
    </tr>
    <tr>
      <th>9</th>
      <td>vrec1417</td>
      <td>-0.398856</td>
      <td>0.066351</td>
      <td>-6.011292</td>
      <td>1.840502e-09</td>
    </tr>
    <tr>
      <th>10</th>
      <td>tinc217</td>
      <td>4699.456443</td>
      <td>2063.357267</td>
      <td>2.277578</td>
      <td>2.275175e-02</td>
    </tr>
    <tr>
      <th>11</th>
      <td>tinc2t5</td>
      <td>292.806157</td>
      <td>480.596033</td>
      <td>0.609256</td>
      <td>5.423546e-01</td>
    </tr>
    <tr>
      <th>12</th>
      <td>tinc6t9</td>
      <td>1154.322187</td>
      <td>595.716788</td>
      <td>1.937703</td>
      <td>5.265947e-02</td>
    </tr>
    <tr>
      <th>13</th>
      <td>tinc1013</td>
      <td>2002.333511</td>
      <td>693.309335</td>
      <td>2.888081</td>
      <td>3.876001e-03</td>
    </tr>
    <tr>
      <th>14</th>
      <td>tinc1417</td>
      <td>1249.994589</td>
      <td>785.365854</td>
      <td>1.591608</td>
      <td>1.114728e-01</td>
    </tr>
  </tbody>
</table>
</div>



## Effect on Employment

In analyzing the employment variables (those that start with "vemp"), the IV estimates reveal that those who believed in the time limit were more likely to be employed throughout the study. However, these differences are only statistically significant throughout the middle of the study. By the end, employment levels were no different - statistically speaking - between those who believed in the time limit versus those who did not. 

## Effect on Welfare Receipt

Over the course of the entire study, when looking at welfare receipt variables (those that start with "vrec"), there was no statistical difference in the amount of welfare received between those who believed in the time limit and those who did not (variable `vrecc217`). However, when looking at the period specific effects, by year three (`vrec1013`) and year four (variable `vrec1417`), those who believed in the time limit were between 13% and 40% more likely to be employed than those who did not believe in the limit. These differences are statistically significant at the 95% level of confidence.

When comparing these results to the OLS results above, it's clear that OLS has not only underestimated the effect of believing in the time limit in year three and year four, but OLS has also misidentified statistical significance. 

###### A note on standard errors:
Finally, when contrasting the IV estimates to OLS, it's clear that IV estimates will always return *larger* standard errors compared to OLS. In this scenario, the standard errors are larger by about two-fold. Intuitively, this makes sense because we're using less data to explain outcomes. We're using the variation in the perceived time limit explained by the assignment to treatment or control that is uncorrelated with the disturbance term. Since we're working with less data, we are less certain of our results and, consequently, the standard errors increase. 

## Effect on Income
Income is the last outcome of interest (variables starting with "tinc"). The data show that those who believed in the time limit had greater levels of income than those who did not. However, the increase is only statistically different from zero in year three of the study (`tinc1013`). 

# Conclusion

There are two key takeaways from this project:

1. Ordinary Least Squares returns biased results when our explanatory variable is endogenous. In this scenario, participant self-selection invalidates our estimates when using OLS. We need an exogenous, instrumental variable, in order to make causal claims. 

2. From a causal inference perspective, at the end of the day, it appears that imposing a time limit on welfare benefits had no effect on employment, welfare, or income outcomes. After four years, participants who believed in the time limit fared no different from participants who did not believe in the time limit. 

    However, statistical magnitudes can be different from real world magnitudes. For example, in year four (`tinc1417`), individuals who believed in the time limit had a higher income, on average, by about 1250 dollars. Since 0 is included in the 95 percent confidence interval, statistical analyses would tell us that the increase is not meaningful. Even so, it's difficult to make the case in a real world setting that an added 1250 dollars is nothing - especially for individuals and families who lie on the lower end of the income distribution.
