import sys

import gensim
import numpy as np
import xgboost as xgb

import file_util.file_processor as file_processor
import logistic_regression.word2vec_builder as word2vec


def get_queries(lines):
    qid_query_pair = {}
    for line in lines:
        elements = line.split("\t")
        qid_query_pair[elements[0]] = elements[2]
    return qid_query_pair


def set_rank_by_score(scored_passage):
    ranked_passage_list = sorted(scored_passage, key=lambda i: i['score'], reverse=True)
    rank = 1
    for top_passage in ranked_passage_list:
        top_passage["rank"] = rank
        rank += 1
    return ranked_passage_list


def get_coefficients(path):
    file_reader = open(path, 'r')
    line = file_reader.readline()
    elements = line.split(",")
    coefficients = np.asarray(elements)
    coefficients = coefficients.astype(float)
    return np.array(coefficients)


def normalize_features(features):
    min = np.min(features)
    max = np.max(features)
    range = max - min
    normalized_features = 1 - ((max - features) / range)
    return normalized_features


def get_probability_score(trained_xgb_model, feature_vector):
    feature_vector = np.float32(feature_vector)
    feature_vector = normalize_features(feature_vector)
    feature_vector = np.asmatrix(feature_vector)
    dtest = xgb.DMatrix(feature_vector)
    score = trained_xgb_model.predict(dtest)
    return score


def run_validation(validation_file_path, w2vec_model_file_path, trained_xgb_model, result_path, threshold,
                   vector_dimension):
    w2vec_model = gensim.models.KeyedVectors.load_word2vec_format(w2vec_model_file_path)
    lines = file_processor.read_lines(validation_file_path)
    lines.pop(0)  # removing column header
    qid_queries = get_queries(lines)
    counter = 0;
    total_queries = len(qid_queries)
    for qid in qid_queries:
        counter += 1
        print("Started->", counter)
        query = qid_queries[qid]
        pid_passage_relevance_tuple = file_processor.get_candidate_passages_relevance_by_qid(lines, qid)
        scored_passage = []
        for ppr in pid_passage_relevance_tuple:
            passage = ppr["passage"]
            relevancy = float(ppr["relevancy"])
            feature_vector = word2vec.build_feature_vector(query, passage, w2vec_model, vector_dimension)
            score = get_probability_score(trained_xgb_model, feature_vector)
            pre_relevancy = 1.0 if score >= threshold else 0.0
            scored_passage.append(
                {"qid": qid, "pid": ppr["pid"], "rank": 0, "score": score[0], "relevancy": relevancy,
                 "pre_relevancy": pre_relevancy, "assigment_name": "A1", "algorithm_name": "LM"})
        sorted_passage = set_rank_by_score(scored_passage)
        file_processor.write_scored_passage(sorted_passage, result_path)
        # select top 250  queries
        if counter > 250:
            break


if __name__ == "__main__":
    validation_file_path = "../dataset/validation_data.tsv"
    w2vec_pre_trained_model = '../dataset/glove-twitter-50.gz'
    trained_model_path = "../result/lamda_mart.model"
    result_output = "../result/LM.txt"

    vector_dimension = 50
    trained_xgb_model = xgb.Booster({'nthread': 4})  # init model
    trained_xgb_model.load_model(trained_model_path)  # load data

    run_validation(validation_file_path, w2vec_pre_trained_model, trained_xgb_model, result_output, .50,
                   vector_dimension)
    sys.exit()
