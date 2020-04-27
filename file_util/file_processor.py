def get_candidate_passages_by_qid(lines, qid):
    pid_passage_pair = {}
    for line in lines:
        elements = line.split("\t")
        if elements[0] == qid:
            pid_passage_pair[elements[1]] = elements[3]
    return pid_passage_pair


def get_candidate_passages_relevance_by_qid(lines, qid):
    pid_passage_relevance = []
    for line in lines:
        elements = line.split("\t")
        if elements[0] == qid:
            pid_passage_pair = {"pid": elements[1], "passage": elements[3], "relevancy": elements[4]}
            pid_passage_relevance.append(pid_passage_pair);
    return pid_passage_relevance


def get_test_queries(path):
    qid_query_pair = {}
    lines = read_lines(path)
    for line in lines:
        elements = line.split("\t")
        qid_query_pair[elements[0]] = elements[1]
    return qid_query_pair


def read_lines(path):
    file_reader = open(path, 'r')
    lines = file_reader.readlines()
    file_reader.close()
    return lines


def get_pid_passage_pair(pid_passage_relevance):
    pid_passage_pair = {}
    for p in pid_passage_relevance:
        pid_passage_pair[p["pid"]] = p["passage"]
    return pid_passage_pair


def read_n_lines(path, n):
    count = 0;
    list = []
    file_reader = open(path, 'r')
    while count < n:
        line = file_reader.readline()
        list.append(line)
        count += 1
    file_reader.close()
    return list


def write(file_path, contents):
    file_writer = open(file_path, 'w')
    file_writer.writelines((contents))
    file_writer.close()


def write_ranked_passage(ranked_passage_list, file_path):
    result_list = []
    for rp in ranked_passage_list:
        result = str(rp["qid"]) + "\t" + rp["assigment_name"] + "\t" + str(rp["pid"]) + "\t" + str(
            rp["rank"]) + "\t" + str(
            rp["score"]) + "\t" + rp["algorithm_name"] + "\t" + str(rp["relevancy"]) + "\n"
        result_list.append(result)
    file_writer = open(file_path, 'a')
    file_writer.writelines((result_list))
    file_writer.close()


def write_scored_passage(score_passages, file_path):
    result_list = []
    for rp in score_passages:
        result = str(rp["qid"]) + "\t" + rp["assigment_name"] + "\t" + str(rp["pid"]) + "\t" + str(
            rp["rank"]) + "\t" + str(
            rp["score"]) + "\t" + rp["algorithm_name"] + "\t" + str(rp["relevancy"]) + "\t" + "\n"
        result_list.append(result)
    file_writer = open(file_path, 'a')
    file_writer.writelines((result_list))
    file_writer.close()
