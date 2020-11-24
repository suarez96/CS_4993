from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec
from nltk.tokenize.treebank import TreebankWordDetokenizer
from tqdm import tqdm
from collections import Counter
import pandas as pd
import os
import numpy as np

import nltk
nltk.download('punkt')


class Embedder:

    default_doc2vec_params = {
        'vec_size': 32,
        'alpha': 0.001,
        'window': 3,
        'min_count': 2,
        'min_alpha': 0.00025,
        'dm': 1
    }

    detokenizer = TreebankWordDetokenizer()

    def __init__(self, d2v_trial_name, d2v_params, train_data, corpus_column, infer_params={
        'alpha': 0.03,
        'steps': 128
    }):
        self.train_df = train_data
        self.corpus = list(train_data[corpus_column])

        self.tagged_data = [TaggedDocument(words=word_tokenize(item.lower()),
                                           tags=[str(i)]) for i, item in enumerate(self.corpus)]

        self.curr_model_name = "{}.model".format(d2v_trial_name)

        self.doc2vec_params = d2v_params
        self.autofill_params()

        # parameters for inference
        self.infer_params = infer_params

        self.doc2vec_model = None
        self.tfidf_model = None

    def autofill_params(self):
        for k, v in Embedder.default_doc2vec_params.items():
            if k not in self.doc2vec_params:
                self.doc2vec_params[k] = v
                print("Defaulted doc2vec param: {}={}".format(k, v))

        assert 'epochs' in self.doc2vec_params, "Must specify epochs hyperparameter!"

    def train_doc2vec(self):
        try:
            assert not os.path.exists(self.curr_model_name), "Model {} already exists! Update model output name".format(
                self.curr_model_name)

            self.doc2vec_model = Doc2Vec(vector_size=self.doc2vec_params['vec_size'],
                            alpha=self.doc2vec_params['alpha'],
                            window=self.doc2vec_params['window'],
                            min_alpha=self.doc2vec_params['min_alpha'],
                            min_count=self.doc2vec_params['min_count'],
                            dm=self.doc2vec_params['dm'])

            self.doc2vec_model.build_vocab(self.tagged_data)

            for epoch in tqdm(range(self.doc2vec_params['epochs'])):
                self.doc2vec_model.train(self.tagged_data,
                            total_examples=self.doc2vec_model.corpus_count,
                            epochs=self.doc2vec_model.epochs)
                # LR scheduling
                self.doc2vec_model.alpha -= 0.00002

            self.doc2vec_model.save(self.curr_model_name)
            print("Model {} Saved".format(self.curr_model_name))

        except AssertionError:
            print("Existing Model {} Found".format(self.curr_model_name))

    def load_doc2vec_model(self):
        self.doc2vec_model = Doc2Vec.load(self.curr_model_name)
        print("Model {} Loaded".format(self.curr_model_name))

    def get_doc2vec_embeddings(self, occ):
        test_data = word_tokenize(occ)
        test_vector = self.doc2vec_model.infer_vector(
            test_data,
            steps=self.infer_params['steps'],
            alpha=self.infer_params['alpha']
        )
        return test_vector

    def get_occ_and_code_from_tokens(self, training_doc):
        """
        Return the train input in readable form as well as its corresponding NOC code
        """
        tokens = self.tagged_data[int(training_doc[0])][0]

        detokenized_job = Embedder.detokenizer.detokenize(tokens).replace(" )", ")")

        code = int(self.train_df.iloc[int(training_doc[0])]['code'])

        return detokenized_job, code

    def infer_doc2vec(self, str_input, verbose=False):
        """
        :param str_input:
        :param verbose:
        :return:
        """

        job_vector = self.get_doc2vec_embeddings(str_input)

        # to find most similar doc using tags
        similar_doc = self.doc2vec_model.docvecs.most_similar([job_vector])

        codes = []

        if verbose:
            print('---------Test on {}---------'.format(str_input))

        for doc in similar_doc:

            job, code = self.get_occ_and_code_from_tokens(doc)

            codes.append(code)

            if verbose:
                print('{} - {}'.format(job, code))

        return Counter(codes)

    def process_votes(self, counter):

        if len(counter) >= 3:
            v1, v2, v3 = (int(w) for w, c in counter.most_common(3))

        elif len(counter) == 2:
            v1, v2 = (int(w) for w, c in counter.most_common(2))
            v3 = 0

        elif len(counter) == 1:
            v1 = counter.most_common(1)[0][0]
            v2, v3 = 0, 0

        return pd.Series([v1, v2, v3])

    def infer_and_vote(self, occ, verbose=False):
        """
        :param embedder: An INSTANCE of embedder class
        :param occ:
        :param verbose:
        :return:
        """
        counter = self.infer_doc2vec(occ, verbose=verbose)
        return self.process_votes(counter)

    def load_tfidf_model(self):
        if self.tfidf_model is None:
            self.tfidf_model = load('vectorizer.joblib')

    def train_tfidf(self):
        # for effient load an dstore of objects w/ large numpy arrays internally
        from joblib import dump, load
        # Remove highly uncommon word (freq < 5) from corpus to reduce dimensionality
        self.tfidf_model = TfidfVectorizer(min_df=5,
                                          stop_words="english",
                                          lowercase=True)

        dump(self.tfidf_model, 'vectorizer.joblib')
        vectorized_X_train = self.tfidf_model.fit_transform(self.corpus)
        print("TF-IDF training vector shape", vectorized_X_train.shape)
        return vectorized_X_train

    def get_tfidf_embeddings(self, data):
        # Transform new data using existing TFIDF model
        test_vector = self.tfidf_model.transform(data)
        return test_vector

    @staticmethod
    def vectorize_embeddings(data):
        return np.array([list(x) for x in np.array(data)])

    def apply_fx(self, data, fx, args):
        # TODO test, incomplete

        ret = []
        for i, r in data.itertuples():
            ret.append(self.fx())
        return ret
