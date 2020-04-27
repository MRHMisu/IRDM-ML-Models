import file_util.file_processor as file_processor
import index.index_builder as index_builder

file_path = "../dataset/candidate_passages_top1000.tsv"

# passage collection
lines = file_processor.read_lines(file_path)

pid_passage_pair = file_processor.get_candidate_passages_by_qid(lines, "1113437")
index = index_builder.build_inverted_index(pid_passage_pair)
