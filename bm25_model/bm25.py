import math

import index.index_builder as index_builder
import index.text_pre_processor as text_processor


def get_bm25_parameters():
    bm25_parameters = {
        "k1": 1.2,  # 𝑘1 determines how the term frequency weight changes as fi increases
        "k2": 100,  # 𝑘2 determines how the query term weight fluctuates when qfi increases.
        "b": 0.75  # b regulates the impact of the length normalization
    }
    return bm25_parameters


def get_candidate_passages_from_index(query_term, index):
    relevant_candidate_passages = []
    for term in query_term:
        if term in index:
            candidate_passage_list = index[term]
            relevant_candidate_passages.extend(candidate_passage_list)
    unique_candidate_passage = list({v['pid']: v for v in relevant_candidate_passages}.values())
    return unique_candidate_passage


def calculate_adl(pid_passage):
    sum_of_all_passage_length = 0;
    for pid in pid_passage:
        tokens = text_processor.preprocess_text(pid_passage[pid])
        sum_of_all_passage_length += len(tokens)
    adl = (sum_of_all_passage_length / len(pid_passage))
    return adl


def calculate_complex_parameter_k(dl, adl):
    parameters = get_bm25_parameters()
    k1 = parameters["k1"]
    b = parameters["b"]
    dl_adl = (dl / adl)
    k = k1 * ((1 - b) + b * dl_adl);
    return k;


def calculate_score_for_each_query_term(N, ni, fi, qfi, k, R, ri):
    parameters = get_bm25_parameters()
    k1 = parameters["k1"]
    k2 = parameters["k2"]
    idf_portion = ((ri + 0.5) / (R - ri + 0.5)) / ((ni - ri + 0.5) / (N - ni - R + ri + 0.5));
    tf_portion = (((k1 + 1) * fi) / (k + fi)) * (((k2 + 1) * qfi) / (k2 + qfi));
    score = ((math.log(idf_portion)) * tf_portion);
    return score


def calculate_score_bm25(candidate_passage, query_terms, N, R, adl, index, relevant_passage_index):
    overall_score = 0;
    dl = candidate_passage["length"]  # dl is the length of the document
    fi = candidate_passage["frequency"]  # fi= is the frequency of term i in the document;
    # k =complex parameter that normalizes the tf component by document length
    k = calculate_complex_parameter_k(dl, adl)
    for term in query_terms:
        if term in index:
            ni = len(index[term])  # ni= total number of documents where the term appear;
        else:
            ni = 0;
        qfi = query_terms[term]  # qfi= is the frequency of term i in the query
        if term in relevant_passage_index:
            ri = len(relevant_passage_index[term])  # ri= The number of relevant documents containing term i
        else:
            ri = 0
        overall_score += calculate_score_for_each_query_term(N, ni, fi, qfi, k, R, ri);
    return overall_score


def is_relevant_passage(relevant_pid_passage, pid):
    for r_pid in relevant_pid_passage:
        if r_pid == pid:
            return 1.0
    return 0.0;


def set_rank_by_score(scored_passage):
    ranked_passage_list = sorted(scored_passage, key=lambda i: i['score'], reverse=True)
    rank = 1
    for top_passage in ranked_passage_list:
        top_passage["rank"] = rank
        rank += 1
    return ranked_passage_list


def get_bm25_scored_passage_by_qid(qid, query, all_pid_passage, relevant_pid_passage):
    scored_passage = []
    query_terms = text_processor.count_term_frequency(query)
    index = index_builder.build_inverted_index(all_pid_passage)
    relevant_passage_index = index_builder.build_inverted_index(relevant_pid_passage)

    adl = calculate_adl(all_pid_passage);  # average document length
    N = len(all_pid_passage)  # N=total number of documents in the collection;
    R = len(relevant_pid_passage)  # R= The number of relevant documents for the query.

    candidate_passages = get_candidate_passages_from_index(query_terms, index)

    for cp in candidate_passages:
        score = calculate_score_bm25(cp, query_terms, N, R, adl, index, relevant_passage_index)
        relevancy = is_relevant_passage(relevant_pid_passage, cp["pid"]);
        scored_passage.append(
            {"qid": qid, "pid": cp["pid"], "rank": 0, "score": score, "relevancy": relevancy, "assigment_name": "A1",
             "algorithm_name": "BM25"})
        sorted_passage = set_rank_by_score(scored_passage)
    return sorted_passage
