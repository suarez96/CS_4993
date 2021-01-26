import pandas as pd
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import os
from TextPreprocessor import TextPreprocessor
from tqdm import tqdm
tqdm.pandas()


class OccupationPreprocessor:

    all_descriptions = {}
    all_job_samples = {}

    def __init__(self):
        return

    @staticmethod
    def first_n_digits(string, n=4):

        # if default number of digits desired, don't do anything
        if n == 4:
            return string

        # else pad left with zeros until 4 digits reached
        padded_str = '{0:0>4}'.format(string)
        return padded_str[:n]

    @staticmethod
    def unpack_descriptions(row):
        # unpack all descriptions from a row and
        duty = row['main_duties']

        # split duty field into separate duties and remove initial generic blurb
        for description in duty.strip('-').split(';'):
            if 'duties' not in description:
                OccupationPreprocessor.all_descriptions[description] = row['Noc_code']

        return row

    @staticmethod
    def unpack_gendered_entries(job):

        # change gendered entries such as 'chairman/woman' into separate samples, 'chairman', 'chairwoman'
        gendered_constituents = []
        if 'man/woman' in job:
            # change original entry to 'job(man)', then append job(woman) to end of list
            gendered_constituents.append(job.replace('man/woman', 'man'))
            gendered_constituents.append(job.replace('man/woman', 'woman'))
        elif 'men/women' in job:
            gendered_constituents.append(job.replace('men/women', 'men'))
            gendered_constituents.append(job.replace('men/women', 'women'))
        elif 'boy/girl' in job:
            gendered_constituents.append(job.replace('boy/girl', 'boy'))
            gendered_constituents.append(job.replace('boy/girl', 'girl'))
        elif 'waiter/waitress' in job:
            gendered_constituents.append(job.replace('waiter/waitress', 'waiter'))
            gendered_constituents.append(job.replace('waiter/waitress', 'waitress'))
        elif 'host/hostess' in job:
            gendered_constituents.append(job.replace('host/hostess', 'host'))
            gendered_constituents.append(job.replace('host/hostess', 'hostess'))
        elif 'master/mistress' in job:
            gendered_constituents.append(job.replace('master/mistress', 'master'))
            gendered_constituents.append(job.replace('master/mistress', 'mistress'))
        else:
            gendered_constituents = [job]

        return gendered_constituents

    @staticmethod
    def prepare_df(file_or_df, input_column, code_column, preprocess_text=False, n_digits=4):

        read_functions = {
            'csv': pd.read_csv,
            'xlsx': pd.read_excel
        }

        # check if dataframe was passed or a file name
        if isinstance(file_or_df, pd.DataFrame):
            ext = None
        else:
            ext = file_or_df.split('.')[-1]

        if ext in read_functions.keys():
            df = read_functions[ext](file_or_df)
        else:
            df = file_or_df

        # drop null or missing codes so program doesn't crash
        df[code_column].replace('', np.nan, inplace=True)
        df.dropna(subset=[code_column], inplace=True)

        # strip single quotes
        df['code'] = df[code_column].apply(
            lambda x: str(x).strip('\'')
        )
        # take double coded inputs and take the first one
        df['code'] = df['code'].apply(
            lambda x: int(x) if (',' not in x and '.' not in x)
            else int(x.split(',')[0].split('.')[0])
        )
        # transform the target variable into the desired length
        df['code'] = df['code'].apply(
            OccupationPreprocessor.first_n_digits, args=(n_digits,)
        )

        if preprocess_text:
            print("Input preprocessed")
            df['input'] = df[input_column].progress_apply(TextPreprocessor.preprocess_text)
        else:
            print("Input unprocessed by default")
            df['input'] = df[input_column]

        return df[['input', 'code']]

    @staticmethod
    def extract_job_samples(row):
        NOC_code = int(row['Noc_code'])

        # split jobs contained in row by ';' and .replace('-', '; ') is for '-', .replace('-', '; ')
        # REVISE WHETHER TO KEEP - separation. logic is that lieutenant-governor can be described as lieutenant governer, no hyphen
        # make unique set
        # strip extra characters
        # and take nonempty elements
        uncleaned_jobs = [
            j for j in row['job_title'].split(';')
            if (j != '' and j != ' ')
        ]

        jobs = []

        for job in uncleaned_jobs:
            jobs += OccupationPreprocessor.unpack_gendered_entries(job)

        # remove duplicate entries
        jobs = set(jobs)

        # parse counts of each job
        row['n_sample_jobs'] = len(jobs)

        # iterate through job and add to dictionary
        for j in jobs:

            if j not in OccupationPreprocessor.all_job_samples:
                OccupationPreprocessor.all_job_samples[j] = NOC_code

            # safe check, if job appears more than once, clause will print the both NOC Codes
            else:
                if OccupationPreprocessor.all_job_samples[j] != NOC_code:
                    print(j, 'repeated', OccupationPreprocessor.all_job_samples[j], NOC_code)

        return row

    @staticmethod
    def parse_1(row):
        # get info from first digit of 4 digit code
        row['1_digit_target'] = OccupationPreprocessor.first_n_digits(row['Noc_code'], 1)
        row['1_digit_group'] = df_skill_type[df_skill_type['skilltype_code'] == row['1_digit_target']][
            'skilltype_title']

        return row

    @staticmethod
    def parse_2(row):
        # get info from first 2 digits of 4 digit code

        # check if NOC code is long enough for parsing
        row['2_digit_target'] = OccupationPreprocessor.first_n_digits(row['Noc_code'], 2)
        row['2_digit_group'] = df_major_group[df_major_group['majorgroup_code'] == '\'' + str(row['2_digit_target'])][
            'majorgroup_title']

        return row

    @staticmethod
    def parse_3(row):
        # get info from first 3 digits of 4 digit code

        # check if NOC code is long enough for parsing
        row['3_digit_target'] = OccupationPreprocessor.first_n_digits(row['Noc_code'], 3)
        row['3_digit_group'] = df_minor_group[df_minor_group['minorgroup_code'] == '\'' + str(row['3_digit_target'])][
            'minorgroup_title']

        return row