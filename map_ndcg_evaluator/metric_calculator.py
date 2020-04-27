import copy
import math

import file_util.file_processor as file_processor


def get_queries_id(scored_passage):
    qid_list = []
    for sp in scored_passage:
        qid_list.append(sp["qid"])
    unique_qids = list(set(qid_list))
    return unique_qids


def calculate_precision_at_rank(scored_passage):
    number_of_relevant_passages_at_rank = 0
    for sp in scored_passage:
        if sp["relevancy"] == 1.0:
            number_of_relevant_passages_at_rank += 1
        number_of_passages_at_rank = sp["rank"]
        # This effectiveness measure is known as precision at rank p.
        precision = float(number_of_relevant_passages_at_rank / number_of_passages_at_rank)
        sp["precision"] = precision


def calculate_average_precision_for_each_query(scored_passage, number_of_relevant_passage):
    total_precision_score = 0;
    for sp in scored_passage:
        if sp["relevancy"] == 1.0:
            total_precision_score += sp["precision"]
    average_precision = 0;
    if number_of_relevant_passage != 0:
        average_precision = float(total_precision_score / number_of_relevant_passage)
    return average_precision


def get_scored_passage_from_result(result_file_path):
    result_lines = file_processor.read_lines(result_file_path)
    scored_passage = []
    for line in result_lines:
        elements = line.split("\t")
        qid = elements[0]
        pid = elements[2]
        rank = int(elements[3])
        score = float(elements[4])
        relevancy = float(elements[6])
        scored_passage.append({"qid": qid, "pid": pid, "rank": rank, "score": score, "relevancy": relevancy})
    return scored_passage


def get_score_passages_by_qid(qid, scored_passage):
    ranked_passage = []
    for sp in scored_passage:
        if sp["qid"] == qid:
            ranked_passage.append(sp)
    return ranked_passage


def get_number_of_relevant_passage(ranked_passage):
    number_of_relevant_passage = 0
    for rp in ranked_passage:
        if rp["relevancy"] == 1.0:
            number_of_relevant_passage += 1
    return number_of_relevant_passage


def get_average_precision_of_queries(result_file_path):
    'Calculate Average Precession (AP) for each query'
    all_scored_passage = get_scored_passage_from_result(result_file_path)
    queries = get_queries_id(all_scored_passage)
    average_precision_of_queries = {}
    for qid in queries:
        scored_passage = get_score_passages_by_qid(qid, all_scored_passage)
        number_of_relevant_passage = get_number_of_relevant_passage(scored_passage)
        calculate_precision_at_rank(scored_passage)
        average_precision = calculate_average_precision_for_each_query(scored_passage, number_of_relevant_passage)
        average_precision_of_queries[qid] = average_precision
        # print(qid + "->" + str(average_precision))
    return average_precision_of_queries


def calculate_mean_average_precision(file_path):
    'measuring the Mean Average Precision (MAP)'

    average_precision_of_queries = get_average_precision_of_queries(file_path)
    number_of_queries = len(average_precision_of_queries)
    overall_score = 0
    for av in average_precision_of_queries:
        overall_score += average_precision_of_queries[av]
    mean_average_precision = float(overall_score / number_of_queries)
    return mean_average_precision


def calculate_dcg_at_rank(scored_passage):
    # DCG_at_rank= rel_1 + i=2,∑rel_i/log2(i)
    # rel_i=relevancy score at rank i
    for i in range(len(scored_passage)):
        sp = scored_passage[i]
        if i != 0:
            sp["dcg_gain"] = (sp["relevancy"] / math.log2(sp["rank"]))
            sp["dcg_value"] = scored_passage[i - 1]["dcg_value"] + sp["dcg_gain"]
        else:
            sp["dcg_gain"] = sp["relevancy"]
            sp["dcg_value"] = sp["relevancy"]


def calculate_dcg_at_p(scored_passage):
    # DCG_at_rank= rel_1 + i=2,∑rel_i/log2(i)
    # rel_i=relevancy score at rank i
    for i in range(len(scored_passage)):
        sp = scored_passage[i]
        if i != 0:
            sp["dcg_gain"] = (math.pow(2, sp["relevancy"]) - 1) / math.log(1 + i)
            sp["dcg_value"] = scored_passage[i - 1]["dcg_value"] + sp["dcg_gain"]
        else:
            sp["dcg_gain"] = sp["relevancy"]
            sp["dcg_value"] = sp["relevancy"]


def calculate_ndcg_at_rank(scored_passage):
    calculate_dcg_at_p(scored_passage)
    # sort the passage based on relevancy
    copy_scored_passage = copy.deepcopy(scored_passage)
    dcg_sort_by_relevancy = sorted(scored_passage, key=lambda i: i['relevancy'], reverse=True)
    # NDCG=normalized discounted cumulative gain (NDCG) values
    # IDCG=the ideal DCG value
    # NDCG=(DCG/IDCG)
    for i in range(len(dcg_sort_by_relevancy)):
        sp = dcg_sort_by_relevancy[i]
        if i != 0:
            # i+1= is the rank based on the relevancy
            sp["idcg_gain"] = (sp["relevancy"] / math.log2(i + 1))
            sp["idcg_value"] = dcg_sort_by_relevancy[i - 1]["idcg_value"] + sp["idcg_gain"]
        else:
            sp["idcg_gain"] = sp["relevancy"]
            sp["idcg_value"] = sp["relevancy"]
    for i in range(len(copy_scored_passage)):
        copy_scored_passage[i]["ndcg_value"] = copy_scored_passage[i]["dcg_value"] / dcg_sort_by_relevancy[i][
            "idcg_value"]
    return copy_scored_passage


def calculate_ndcg_whole_query_set(result_file_path):
    'measuring the NDCH@100 for the full query set'

    all_scored_passage = get_scored_passage_from_result(result_file_path)
    queries = get_queries_id(all_scored_passage)
    total_queries = len(queries)
    ndcg_100 = 0
    for qid in queries:
        scored_passage = get_score_passages_by_qid(qid, all_scored_passage)
        ndcg_passage = calculate_ndcg_at_rank(scored_passage)
        ndcg_100 += ndcg_passage[99]["ndcg_value"]
    ndcg_whole_query = (ndcg_100 / total_queries)
    return ndcg_whole_query
