import index.text_pre_processor as text_processor


def build_inverted_index(pid_passage_pair):
    'Build the inverted index and for a term return list of candidate passage'
    inverted_index = {}
    candidate_passages_list = get_indexed_passage(pid_passage_pair)
    for candidate_passage in candidate_passages_list:
        for key in candidate_passage:
            if key not in inverted_index:
                list_of_candidate_passage = [candidate_passage.get(key)]
                inverted_index[key] = list_of_candidate_passage
            else:
                inverted_index[key].append(candidate_passage.get(key))
    return inverted_index


def get_indexed_passage(pid_passage_pair):
    indexed_passages = []
    key_list = list(pid_passage_pair.keys())
    for pid in key_list:
        passage = pid_passage_pair[pid]
        term_candidate_passage_pair = get_candidate_passage(pid, passage)
        indexed_passages.append(term_candidate_passage_pair)
    return indexed_passages


def get_candidate_passage(pid, passage):
    term_candidate_passage_pair = {}
    tokens_length = text_processor.remove_stopwords(passage)
    term_frequency_pair = text_processor.count_term_frequency(passage)
    for term in term_frequency_pair:
        frequency = term_frequency_pair[term]
        term_candidate_passage_pair[term] = {"pid": pid, "frequency": frequency, "length": len(tokens_length)}
    return term_candidate_passage_pair
