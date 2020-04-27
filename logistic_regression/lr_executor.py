import sys

import gensim
import numpy as np

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


def sigmoid(coefficients, feature_vectors):
    return 1.0 / (1 + np.exp(-np.dot(feature_vectors, coefficients.T)))


def get_probability_score(coefficients, feature_vector):
    feature_vector = normalize_features(feature_vector)
    array = np.array([1.0])
    fv = np.insert(array, array.size, feature_vector)
    probability = sigmoid(coefficients, fv)
    return np.float32(probability)


def run_validation(validation_file_path, w2vec_model_file_path, coefficient_path, result_path, threshold,
                   vector_dimension):
    coefficients = get_coefficients(coefficient_path)
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
            score = get_probability_score(coefficients, feature_vector)
            # print(score)
            scored_passage.append(
                {"qid": qid, "pid": ppr["pid"], "rank": 0, "score": score, "relevancy": relevancy,
                 "assigment_name": "A1", "algorithm_name": "LR"})
        sorted_passage = set_rank_by_score(scored_passage)
        file_processor.write_scored_passage(sorted_passage, result_path)
        # select top 250 queries
        if counter > 250:
            break


if __name__ == "__main__":
    validation_file_path = "../dataset/validation_data.tsv"
    w2vec_pre_trained_model = '../dataset/glove-twitter-50.gz'

    coefficients_input = "../result/logistic-coefficients.txt"
    result_output = "../result/LR.txt"

    vector_dimension = 50

    run_validation(validation_file_path, w2vec_pre_trained_model, coefficients_input, result_output, .5,
                   vector_dimension)
    sys.exit()
