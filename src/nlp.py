import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer


def clean_corpus(names_list):
    """remove special chars name in names_list

    :param names_list: list of shop names
    :type names_list: list
    :return: original list with special chars removed.
    :rtype: list
    """
    cleaned = []
    for n in names_list:
        n_cleaned = re.sub("[.,\")(!]", "", n)
        cleaned.append(n_cleaned.upper())
    return cleaned


def tokenize_it(n_list):
    """tokenize each item in n_list

    :param n_list: list of words (i.e. shop names)
    :type n_list: list
    :return: list of lists of tokenized words
    :rtype: list
    """
    return [t.split() for t in n_list]


def clean_names(names_list):
    """Create list of cleaned up names from names_list

    :param names_list: list of shop names
    :type names_list: list
    :return: list of cleaned up names
    :rtype: list
    """
    corpus = clean_corpus(names_list)
    corpus = tokenize_it(corpus)
    return [' '.join(item) for item in corpus]


def get_top_words(df, corpus, top_n):
    """Return count vectorized dataframe of top_n most frequent
    words in corpus (i.e. shop names)

    :param df: dataframe to append corpus info to
    :type df: pd.DataFrame
    :param corpus: list of words (i.e. shop_names)
    :type corpus: list
    :param top_n: number of top ranked items by frequency
    :type top_n: int
    :return: original dataframe with vectorized words
    :rtype: pd.DataFrame
    """
    cvect = CountVectorizer()
    count_matrix = cvect.fit_transform(corpus)
    word_counts = np.sum(count_matrix.toarray(), axis=0)
    vocab = cvect.get_feature_names()
    count_rank = np.argsort(word_counts)[::-1]
    word_rank = np.array(vocab)[count_rank]

    count_df = pd.DataFrame(data=count_matrix.toarray(),
                            columns=vocab)

    df = pd.concat([df, count_df.loc[:, word_rank[:top_n]]], axis=1)
    return df