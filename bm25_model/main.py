import bm25_model.bm25_executor as bm25_executor

validation = "../dataset/validation_data.tsv"
bm25_output = "../result/BM25.txt"

if __name__ == "__main__":
    bm25_executor.run_bm25_model(validation, bm25_output)
