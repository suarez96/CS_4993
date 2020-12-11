import pandas as pd
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import os

class TextPreprocessor:

    char_occurences = {}

    abbreviations_map = {}

    STRIP_ABBREVIATIONS = True

    def __init__(self, strip_abbrev=True):
        TextPreprocessor.STRIP_ABBREVIATIONS = strip_abbrev
        TextPreprocessor.abbreviations_map = {}
        TextPreprocessor.char_occurences = {}
        return

    @staticmethod
    def handle_single_quotes(text):
        """
        handle plurals, which are the main use of the single quote. Afterwards, drop all other single quotes
        """
        text = text.replace("s'", '').replace("'s", '')
        return text.replace("'", '')

    @staticmethod
    def handle_parentheses(text):
        """
        Parentheses seem to fall into two general cases in the VAST majority of instances:
        1. Indicates an abbreviation
        2. Indicates an exception, by using keywords such as "except" or "non"
        """
        parentheses_idx = 0
        split = text.split("(")
        for i, substr in enumerate(split):
            if ')' in substr:
                parentheses_idx = i
                break

        # fragment before the fragment with the paren.
        str1 = split[parentheses_idx - 1].strip()
        assert not ')' in str1

        # fragment w parenthesis
        str2 = split[parentheses_idx].split(")")[0].strip()

        if 'except' in str2 or 'non' in str2:
            text = text.replace(str2, '')
            # TODO, do something with exceptions

        else:
            # take the shorter string as the abbreviation
            ab, ex = (str1, str2) if len(str1) < len(str2) else (str2, str1)

            # save abbreviation
            TextPreprocessor.abbreviations_map[ab] = ex

            # remove the found abbreviation from job title
            if TextPreprocessor.STRIP_ABBREVIATIONS:
                text = text.replace(ab, '')

        # remove parentheses, leading and trailing whitespace
        text = text.replace('(', '').replace(')', '').strip()

        return text

    @staticmethod
    def preprocess_text(text):

        # handle slashes
        text = text.replace("/", ' ')

        # remove redundant semi-colons
        text = text.strip(';')

        # hyphens are semantic noise, remove
        text = text.replace('-', ' ')

        # handle '
        if "'" in text:
            text = TextPreprocessor.handle_single_quotes(text)

        # handle ,
        text = text.replace(",", '')

        # handle .
        text = text.replace(".", '')

        # handle parentheses, only one check necessary since we already
        # verified they are all paired with corresponding ')'
        if "(" in text:
            text = TextPreprocessor.handle_parentheses(text)

        # remove leading and trailing whitespace
        text = text.strip()

        # normalize case
        return text.lower()

    @staticmethod
    def find_character(string, char):

        occurrences = 0
        for occupation in string.split(';'):
            if char in occupation:
                print(occupation)
                occurrences += 1

        if char in TextPreprocessor.char_occurences:
            TextPreprocessor.char_occurences[char] += occurrences
        else:
            TextPreprocessor.char_occurences[char] = occurrences
