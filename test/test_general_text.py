import pytest
from dataflow.operators.process.GeneralText import NgramFilter, MinHashDeduplicator
from dataflow.operators.refine.GeneralText import HtmlUrlRemoverRefiner
from dataflow.utils.storage import FileStorage
from dataflow.llmserving import APILLMServing_request, LocalModelLLMServing

class TextPipeline():
    def __init__(self, llm_serving=None):
        
        self.storage = FileStorage(
            first_entry_file_name="../dataflow/example/GeneralTextPipeline/pt_input.jsonl",
            cache_path="./cache",
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl",
        )
        
        self.ngram_scorer_step_1 = NgramFilter(min_score=0.99, max_score=1.0, ngrams=5)
        self.html_remover_step_2 = HtmlUrlRemoverRefiner()
        self.minhash_deduplicator_step_3 = MinHashDeduplicator(num_perm=128, threshold=0.9, use_n_gram=True, ngram=5)
        
        if llm_serving is None:
            llm_serving = LocalModelLLMServing(
                # model_name_or_path="/data0/models/Qwen2.5-7B-Instruct", # set to your own model path
                model_name_or_path="/mnt/public/model/huggingface/Qwen2.5-7B-Instruct",
                tensor_parallel_size=4,
                max_tokens=8192,
                model_source="local"
            )
    def forward(self):
        
        self.ngram_scorer_step_1.run(
            storage = self.storage.step(),
            input_key = "raw_content",
            output_key = "ngram_score",
        )
        
        self.html_remover_step_2.run(
            storage = self.storage.step(),
            input_key='raw_content',
        )
        
        self.minhash_deduplicator_step_3.run(
            storage = self.storage.step(),
            input_key='raw_content',
            output_key='minhash_deduplicated_label',
        )
# @pytest.mark.gpu
# def test_text_pipeline_runs_without_errors():
#     try:
#         pipeline = TextPipeline()
#         pipeline.forward()
#     except Exception as e:
#         pytest.fail(f"TextPipeline execution failed with error: {e}")
if __name__ == "__main__":
    model = TextPipeline()
    model.forward()

