import pickle
import pandas as pd
from scripts.Embedder import Embedder, Doc2VecEmbedder, tfidfEmbedder
from scripts.OccupationPreprocessor import OccupationPreprocessor
import numpy as np

d2v_train_df=OccupationPreprocessor.prepare_df(
            './Data/doc2vec_train_set.csv',
            input_column='input',
            code_column='code',
            preprocess_text=True
        )

if __name__ == '__main__':

    # TODO, work for single sample
    # TODO, make a top 5

    with open('first_dig_tfidf_clfs.pkl', 'rb') as f:
        clf1=pickle.load(f)

    with open('second_third_fourth_dig_tfidf_clfs.pkl', 'rb') as f2:
        clf2=pickle.load(f2)

    tfidf_test_df = OccupationPreprocessor.prepare_df(
        args.test_set,  # './Data/overlap_test_set_v4_acanoc_no_train_data.csv',
        input_column='input',
        code_column='code',
        preprocess_text=False
    )

    d2v_test_df = OccupationPreprocessor.prepare_df(
        args.test_set,  # './Data/overlap_test_set_v4_acanoc_no_train_data.csv',
        input_column='input',
        code_column='code',
        preprocess_text=True
    )

    assert len(d2v_test_df) == len(tfidf_test_df), "Test set lengths do not match"

    d2vembedder = Doc2VecEmbedder(model_name='trial_11.model', train_data=d2v_train_df, infer_params={
        'steps': 2048,
        'alpha': 0.03
    })

    tfidfembedder = tfidfEmbedder()

    sample_pipeline_df = tfidf_test_set.sample(5, random_state=123)

    d2v_predictions = sample_pipeline_df['input'].apply(d2vembedder.infer, axis=1)

    prediction_df = pd.DataFrame({
        'svm_pred': clf1['SVM'].predict(tfidf_test_vectors),
        'rf_pred': clf1['RF'].predict(tfidf_test_vectors),
        'knn_pred': clf1['KNN'].predict(tfidf_test_vectors),
        'd2v_pred': d2vembedder.score_predictions(d2v_predictions, level=1, topn=10),
        'code': sample_pipeline_df['code'].astype(str)
    })

    prediction_df['exact_match'] = sample_pipeline_df['input'].apply(embedder.check_exact_match,
                                                                     args=(embedder.train_database,))

    prediction_df['p_all_1'] = tfidf_test_df.apply(tfidfEmbedder.ensemble_predict, axis=1, args=(
        ['rf_pred', 'svm_pred', 'knn_pred', 'd2v_pred'], 'svm_pred',
    ))

    prediction_df['vectors'] = tfidf_test_vectors.toarray().tolist()

    def pipeline(row):
        np_array = np.array(row['vectors']).reshape(1, -1)
        p_1 = row['p_all_1']
        row['svm_pred_234'] = clf2[p_1]['SVM'].predict(np_array)[0]
        row['rf_pred_234'] = clf2[p_1]['RF'].predict(np_array)[0]
        row['knn_pred_234'] = clf2[p_1]['KNN'].predict(np_array)[0]
        row['d2v_pred_234'] = embedder.hyperbolic_scoring(
            d2v_predictions,
            level=4,
            topn=30,
            level_constraint=row['p_all_1']
        )
        return row

    prediction_df['p_all_234'] = tfidf_test_df.apply(tfidfEmbedder.ensemble_predict, axis=1, args=(
        ['svm_pred_234', 'rf_pred_234', 'knn_pred_234', 'd2v_pred_234'], 'knn_pred_234',
    ))

    print('Predictions:\n', prediction_df[['p_all_234', 'code']])