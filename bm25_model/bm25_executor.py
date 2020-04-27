import bm25_model.bm25 as bm25_model
import file_util.file_processor as file_processor


def get_queries(lines):
    qid_query_pair = {}
    for line in lines:
        elements = line.split("\t")
        qid_query_pair[elements[0]] = elements[2]
    return qid_query_pair


def get_relevant_pid_passage_pair(pid_passage_relevance):
    relevant_pid_passage_pair = {}
    for p in pid_passage_relevance:
        if float(p["relevancy"]) == 1.0:
            relevant_pid_passage_pair[p["pid"]] = p["passage"]
    return relevant_pid_passage_pair


def get_number_of_relevant_passage(pid_passage_relevance):
    R = 0;
    for ppr in pid_passage_relevance:
        if float(ppr["relevancy"]) == 1.0:
            R += 1
    return R


def get_pid_passage_pair(pid_passage_relevance):
    pid_passage_pair = {}
    for p in pid_passage_relevance:
        pid_passage_pair[p["pid"]] = p["passage"]
    return pid_passage_pair


def get_scored_passage_by_query(qid, query, all_pid_passage, relevant_pid_passage):
    scored_passage = bm25_model.get_bm25_scored_passage_by_qid(qid, query, all_pid_passage, relevant_pid_passage)
    return scored_passage


def run_bm25_model(validation_file_path, result_file_path):
    lines = file_processor.read_lines(validation_file_path)
    lines.pop(0)  # removing column header
    qid_queries = get_queries(lines)
    counter = 0;
    total_queries = len(qid_queries)
    for qid in qid_queries:
        counter += 1
        query = qid_queries[qid]
        pid_passage_relevance_tuple = file_processor.get_candidate_passages_relevance_by_qid(lines, qid)
        all_pid_passage = get_pid_passage_pair(pid_passage_relevance_tuple)
        relevant_pid_passage = get_relevant_pid_passage_pair(pid_passage_relevance_tuple)
        scored_passage = get_scored_passage_by_query(qid, query, all_pid_passage, relevant_pid_passage)
        print("Completed->  " + str(counter) + "  -> out of: " + str(total_queries))
        file_processor.write_ranked_passage(scored_passage, result_file_path)
