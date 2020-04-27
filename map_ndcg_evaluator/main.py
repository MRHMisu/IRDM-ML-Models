import map_ndcg_evaluator.metric_calculator as evaluator


# model generated output file path
# change the path based on the model which is evaluated. Here NN.txt represents evaluating Neural Network model

result_file_path = "../result/NN.txt"

if __name__ == "__main__":
    map = evaluator.calculate_mean_average_precision(result_file_path)
    print("MAP:=", map)
    ndcg = evaluator.calculate_ndcg_whole_query_set(result_file_path)
    print("NDCG@100:=", ndcg)
