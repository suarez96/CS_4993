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
from joblib import dump, load
from collections import Counter
from OccupationPreprocessor import OccupationPreprocessor

import nltk
nltk.download('punkt')


class Embedder:

    def __init__(self, database_file='../Data/doc2vec_train_set.csv'):
        self.train_database = pd.DataFrame(pd.read_csv(database_file))

    def embed(self, data):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    @staticmethod
    def check_exact_match(row, reference_df):
        exact_matches = reference_df.loc[reference_df['input'] == str(row)]
        code = exact_matches['code'].values[0] if len(exact_matches) == 1 else -1
        return code

    @staticmethod
    def vectorize_embeddings(data):
        return np.array([list(x) for x in np.array(data)])

    def apply_fx(self, data, fx, args):
        # TODO test, incomplete

        ret = []
        for i, r in data.itertuples():
            ret.append(self.fx())
        return ret


class tfidfEmbedder(Embedder):

    def __init__(self, model_path='../vectorizer.joblib', training=False, corpus=None):
        super().__init__()
        self.tfidf_model = None
        if training:
            assert corpus is not None, "Null Training Corpus. Define before continuing."
            self.train()
        else:
            if os.path.exists(model_path):
                self.load(model_path)
            else:
                print("Path to model not found. Load model manually with load(<path>) before performing inference")

    def load(self, path):
        if self.tfidf_model is None:
            self.tfidf_model = load(path)
            print("Model loaded from {}".format(path))
        else:
            print("Model already loaded")

    def train(self):
        # for effient load an dstore of objects w/ large numpy arrays internally

        # Remove highly uncommon word (freq < 5) from corpus to reduce dimensionality
        self.tfidf_model = TfidfVectorizer(min_df=5,
                                           stop_words="english",
                                           lowercase=True)

        vectorized_X_train = self.tfidf_model.fit_transform(self.corpus)
        dump(self.tfidf_model, 'vectorizer.joblib')
        print("TF-IDF training vector shape", vectorized_X_train.shape)
        return vectorized_X_train

    @staticmethod
    def ensemble_predict(row, predictor_cols, default_predictor):

        # find majority vote for all methods
        counter = Counter(row[predictor_cols])
        if -1 in counter:
            counter[-1] = 0
        votes = counter.most_common(1)

        # take svm as tie-breaker because CURRENTLY most accurate
        winning_class, highest_num_votes = votes[0]
        return row[default_predictor] if highest_num_votes < 2 else winning_class

    def embed(self, data):
        # Transform new data using existing TFIDF model
        test_vector = self.tfidf_model.transform(data)
        return test_vector


class Doc2VecEmbedder(Embedder):

    default_doc2vec_params = {
        'vec_size': 32,
        'alpha': 0.001,
        'window': 3,
        'min_count': 2,
        'min_alpha': 0.00025,
        'dm': 1
    }

    def __init__(self, model_path='../trial_11.model', d2v_params={}, train_data=None, corpus_column="input", training=False,
                 infer_params={'alpha': 0.03, 'steps': 128}, scoring="hyper"):
        super().__init__()

        self.model_path = model_path

        # parameters for inference
        self.infer_params = infer_params

        self.doc2vec_model = None

        # method of scoring
        self.scoring = scoring.lower()
        assert self.scoring in ['hyper', 'count'], "Scoring Mode must be one of \'Hyper\' or\'Count\'."

        self.detokenizer = TreebankWordDetokenizer()

        assert train_data is not None, print(
            "For both inference and training, you must specify the training dataframe" +
            "and the column in that dataframe used to build the corpus (the default column is \'input\')."
        )

        self.train_df = train_data
        self.corpus = list(self.train_df[corpus_column])
        self.tagged_data = [TaggedDocument(words=word_tokenize(item.lower()),
                                           tags=[str(i)]) for i, item in enumerate(self.corpus)]

        if training:
            self.doc2vec_params = d2v_params
            self.autofill_params()
        else:
            self.load()

    def autofill_params(self):
        for k, v in Doc2VecEmbedder.default_doc2vec_params.items():
            if k not in self.doc2vec_params:
                self.doc2vec_params[k] = v
                print("Defaulted doc2vec param: {}={}".format(k, v))

        assert 'epochs' in self.doc2vec_params, "Must specify epochs hyperparameter!"

    def train(self):
        try:
            assert not os.path.exists(self.model_path), "Model {} already exists! Update model output name".format(
                self.model_path)

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

            self.doc2vec_model.save(self.model_path)
            print("Model {} Saved".format(self.model_path))

        except AssertionError as e:
            print(e)

    def load(self):
        self.doc2vec_model = Doc2Vec.load(self.model_path)
        print("Doc2vec model succesfully loaded from {}".format(self.model_path))

    def embed(self, occ):
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

        detokenized_job = self.detokenizer.detokenize(tokens).replace(" )", ")")

        code = int(self.train_df.iloc[int(training_doc[0])]['code'])

        return detokenized_job, code

    def single_inference(self, model_input, verbose):
        """
        :param str_input:
        :param verbose:
        :return:
        """

        job_vector = self.embed(model_input)

        # to find most similar doc using tags
        similar_doc = self.doc2vec_model.docvecs.most_similar([job_vector], topn=50)

        codes = []

        if verbose:
            print('---------Test on {}---------'.format(model_input))

        for doc in similar_doc:

            job, code = self.get_occ_and_code_from_tokens(doc)

            codes.append(code)

            if verbose:
                print('{} - {}'.format(job, code))

        return codes

    def infer(self, model_input, verbose=False):

        if type(model_input) == str:
            model_input = pd.DataFrame(pd.Series(model_input), columns=['input'])

        elif type(model_input) != pd.DataFrame:
            model_input = pd.DataFrame(list(model_input), columns=['input'])

        pred = []

        for row in model_input.itertuples():
            pred.append(
                self.single_inference(row.input, verbose)
            )

        return pred

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

    @staticmethod
    def hyperbolic_scoring(row, level, topn, scoring, return_size=5):
        """
        :param pred: array of predictions produced by the 'infer' method
        :param level: the level of the hierarchy at which we want to predict
        :param topn: how many of the top 100 (the number of predictions always returned by 'infer') predictions
        we want to use to calculate the scoring
        :param level_constraint: the prediction of a higher level, used to filter out extraneous predictions at lower
        levels of the hierarchy
        :return: list of tuples of type (predicted class, score), sorted from highest to lowest score
        """
        scores = {}
        pred = row['pred']
        constraint = row['level_constraint']
        # denominator is 1 if scoring mode is count, otherwise it is the index + 2
        scaling_factor = (lambda x: x**0) if scoring == "count" else (lambda x: x+2)
        for idx, pred in enumerate(pred[:topn]):

            # turn prediction into string
            pred_first_n_digits = OccupationPreprocessor.first_n_digits(pred, level)

            # if the level constraint doesn't match the first n digits of the lower-level prediction, we skip
            if constraint != "" and str(pred_first_n_digits)[:len(constraint)] != str(constraint):
                continue

            if pred_first_n_digits in scores:
                scores[pred_first_n_digits] += 1 / scaling_factor(idx)
            else:
                scores[pred_first_n_digits] = 1 / scaling_factor(idx)

        # make a template of null codes, in case of fewer codes then requested we still have constant
        # size of returned array
        template = np.ones(return_size, dtype=np.int) * -1

        if scores:
            # sort jobs by highest score, then trim excess jobs
            sorted_codes = np.array(
                [k for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True)]
            )[:return_size]
            # pad to the right of the array in case there are fewer than 'return_size' top codes
            template[:sorted_codes.shape[0]] = sorted_codes

        return template

    def score_predictions(self, preds, level, topn, level_constraints=[], return_size=5):
        """
        :param preds: list of top 50 doc2vec predictions
        :param level:
        :param topn:
        :param level_constraints: list of constraints for each sample
        :return:
        """

        # if single prediction
        if not any(isinstance(i, list) for i in preds):
            preds = [preds]

        if level_constraints == []:
            level_constraints = ["" for pred in preds]

        # multi prediction
        if type(preds) != pd.DataFrame:
            preds_df = []
            for pred, constraint in zip(preds, level_constraints):
                preds_df.append({'pred': pred, 'level_constraint':constraint})
            preds = pd.DataFrame(preds_df)

        scores = preds.apply(Doc2VecEmbedder.hyperbolic_scoring, axis=1, args=(level, topn, self.scoring, return_size))

        return scores

    def infer_and_vote(self, occ, verbose=False):
        """
        :param embedder: An INSTANCE of embedder class
        :param occ:
        :param verbose:
        :return:
        """
        counter = Counter(self.infer(occ, verbose=verbose))
        return self.process_votes(counter)
