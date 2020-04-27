import sys
import warnings
from pathlib import Path

import gensim
import numpy as np

import index.text_pre_processor as text_processor

warnings.filterwarnings(action='ignore')


def build_feature_vector(query, passage, w2vec_model, vector_size):
    average_query_vector = calculate_average_word_embedding(query, w2vec_model, vector_size)
    average_passage_vector = calculate_average_word_embedding(passage, w2vec_model, vector_size)
    feature_vector = np.concatenate([average_query_vector, average_passage_vector])
    return feature_vector


def calculate_average_word_embedding(query_or_passage, w2vec_model, vector_size):
    query_or_passage_words = text_processor.preprocess_text(query_or_passage)
    vector = np.zeros(vector_size)
    average_embeddings_vector = np.zeros(vector_size)
    for word in query_or_passage_words:
        if word in w2vec_model:
            vector += np.array(w2vec_model[word])
        average_embeddings_vector = np.divide(vector, len(query_or_passage_words))
    return average_embeddings_vector


def get_word_embedded_feature_vector(line, w2vec_model, vec_dimension):
    elements = line.split("\t")
    query = elements[2]
    passage = elements[3]
    relevancy = float(elements[4])
    feature_vector = build_feature_vector(query, passage, w2vec_model, vec_dimension)
    dimension = feature_vector.size
    relevance_vector = np.insert(feature_vector, dimension, relevancy)
    return relevance_vector


def build_feature_vector_training_input(data_path, w2vec_trained_model_path, vec_dimension, out_feature_vector):
    w2vec_model = gensim.models.KeyedVectors.load_word2vec_format(w2vec_trained_model_path)
    w2vec_model.init_sims(replace=True)
    counter = 0;
    dataset = []
    with open(data_path) as infile:
        for line in infile:
            if counter != 0:  # skipping header
                line_size = len(line.split("\t"))
                if line_size == 5:
                    print("Counter->", counter)
                    relevance_vector = get_word_embedded_feature_vector(line, w2vec_model, vec_dimension)
                    relevance_vector = np.float32(relevance_vector)
                    dataset.append(relevance_vector)
            counter += 1
    p = Path(out_feature_vector)
    with p.open('wb') as f:
        np.save(f, np.array(dataset))


if __name__ == "__main__":
    validation_file_path = "../dataset/validation_data.tsv"
    training_file_path = "../dataset/train_data.tsv"

    w2vec_pre_trained_model = '../dataset/glove-twitter-50.gz'

    out_train__fv = "../dataset/fv_train.npy"
    out_validate_fv = "../dataset/fv_validation.npy"

    vec_dimension = 50

    build_feature_vector_training_input(validation_file_path, w2vec_pre_trained_model, vec_dimension, out_validate_fv)
    print("Validation Feature Done")
    build_feature_vector_training_input(training_file_path, w2vec_pre_trained_model, vec_dimension, out_train__fv)
    print("Training Feature Done")
    sys.exit()
