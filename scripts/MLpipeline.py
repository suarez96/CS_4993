import pickle
import pandas as pd
from Embedder import Embedder, Doc2VecEmbedder, tfidfEmbedder
from OccupationPreprocessor import OccupationPreprocessor
from TextPreprocessor import TextPreprocessor
import numpy as np
import argparse
import os
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

tqdm.pandas()

parser = argparse.ArgumentParser()

parser.add_argument('-t', '--test_set', type=str, help="Path to test set csv file. Necessary when one_inference flag not set.")
parser.add_argument('-j', '--job', type=str, help="Job title for single inference. Necessary when one_inference flag set.")
parser.add_argument('-i', '--input_column', type=str, help="Column specifying job titles", default="input")
parser.add_argument('-c', '--code_column', type=str, help="Column specifying NOC codes", default="code")
parser.add_argument('-s', '--sample_size', type=int, help="Sample size taken randomly from test set", default=-1)
parser.add_argument('-o', '--one_inference', action='store_true')
parser.add_argument('-d', '--doc2vec_model', type=str, help="Path to doc2vec .model file", default="../trial_11.model")

args = parser.parse_args()

if __name__ == '__main__':

    assert os.path.split(os.getcwd())[-1] == 'scripts', "Please run again from the 'scripts' directory"

    d2v_train_df = OccupationPreprocessor.prepare_df(
        '../Data/doc2vec_train_set.csv',
        input_column='input',
        code_column='code',
        preprocess_text=True
    )

    # TODO, print the title alongside the NOC code to see if makes sense
    # TODO, make a top 5

    with open('../first_dig_tfidf_clfs.pkl', 'rb') as f:
        clf1=pickle.load(f)

    with open('../second_third_fourth_dig_tfidf_clfs.pkl', 'rb') as f2:
        clf2=pickle.load(f2)

    if args.one_inference:
        tfidf_input = pd.DataFrame({
            'input': [args.job],
            'code': [-1]
        })

        d2v_input = pd.DataFrame({
            'input': [TextPreprocessor.preprocess_text(args.job)],
            'code': [-1]
        })

    else:
        tfidf_input = OccupationPreprocessor.prepare_df(
            args.test_set,  # './Data/overlap_test_set_v4_acanoc_no_train_data.csv',
            input_column=args.input_column,
            code_column=args.code_column,
            preprocess_text=False
        )

        d2v_input = OccupationPreprocessor.prepare_df(
            args.test_set,  # './Data/overlap_test_set_v4_acanoc_no_train_data.csv',
            input_column=args.input_column,
            code_column=args.code_column,
            preprocess_text=True
        )

    assert len(d2v_input) == len(tfidf_input), "Test set lengths do not match"

    d2vembedder = Doc2VecEmbedder(model_path=args.doc2vec_model, train_data=d2v_train_df, infer_params={
        'steps': 2048,
        'alpha': 0.03
    })

    tfidfembedder = tfidfEmbedder()

    if args.sample_size != -1 and not args.one_inference:
        tfidf_input = tfidf_input.sample(args.sample_size, random_state=123)
        d2v_input = d2v_input.sample(args.sample_size, random_state=123)

    tfidf_test_vectors = tfidfembedder.embed(tfidf_input['input'])

    assert len(tfidf_input) == len(d2v_input)

    d2v_predictions = d2vembedder.infer(d2v_input['input'])

    prediction_df = pd.DataFrame({
        'input':d2v_input['input'].astype(str),
        'svm_pred': clf1['SVM'].predict(tfidf_test_vectors),
        'rf_pred': clf1['RF'].predict(tfidf_test_vectors),
        'knn_pred': clf1['KNN'].predict(tfidf_test_vectors),
        'code': d2v_input['code'].astype(str)
    })

    if 'v4_pred' in d2v_input.columns:
        prediction_df['acanoc_pred'] = d2v_input['v4_pred']

    prediction_df['d2v_pred'] = d2vembedder.score_predictions(d2v_predictions, level=1, topn=10)

    print("Searching for exact matches in database...")
    prediction_df['exact_match'] = d2v_input['input'].progress_apply(d2vembedder.check_exact_match,
                                                                     args=(d2vembedder.train_database,))

    print("Predicting on first level...")
    prediction_df['p_all_1'] = prediction_df.progress_apply(tfidfEmbedder.ensemble_predict, axis=1, args=(
        ['rf_pred', 'svm_pred', 'knn_pred', 'd2v_pred'], 'svm_pred',
    ))

    prediction_df['vectors'] = tfidf_test_vectors.toarray().tolist()

    def pipeline(row, prev_level_col, svm_out_col, rf_out_col, knn_out_col):
        np_array = np.array(row['vectors']).reshape(1, -1)
        p_1 = row[prev_level_col]
        row[svm_out_col] = clf2[p_1]['SVM'].predict(np_array)[0]
        row[rf_out_col] = clf2[p_1]['RF'].predict(np_array)[0]
        row[knn_out_col] = clf2[p_1]['KNN'].predict(np_array)[0]
        return row

    prediction_df['d2v_pred_234'] = d2vembedder.score_predictions(
        preds=d2v_predictions, level_constraints=prediction_df['p_all_1'].tolist(),
        level=4,
        topn=30
    )

    print("Predicting on second level...")
    prediction_df = prediction_df.progress_apply(
        pipeline, axis=1, args=('p_all_1', 'svm_pred_234', 'rf_pred_234', 'knn_pred_234',)
    )

    print("Combining Doc2Vec and TFIDF predictions")
    prediction_df['p_all_234'] = prediction_df.progress_apply(tfidfEmbedder.ensemble_predict, axis=1, args=(
        ['svm_pred_234', 'rf_pred_234', 'knn_pred_234', 'd2v_pred_234'], 'knn_pred_234',
    ))

    print('Predictions:\n', prediction_df[['input', 'p_all_234', 'code']].head(5))

    if not args.one_inference:
        if not os.path.exists('../output/'):
            os.mkdir('../output/')
            print('output directory created')
        test_file_name_stripped = args.test_set.split('/')[-1]
        # write out file
        prediction_df.drop(columns=['vectors']).to_csv('../output/ML_pipeline_out_sample_{}_{}'.format(
            args.sample_size,
            test_file_name_stripped
        ), index=False)
        print(
            'Accuracy (Micro-F1):\n', accuracy_score(
                prediction_df['p_all_234'].astype(int),
                prediction_df['code'].astype(int)
            )
        )
        print(
            'Macro-F1:\n', f1_score(
                prediction_df['p_all_234'].astype(int),
                prediction_df['code'].astype(int),
                average='macro'
            )
        )

