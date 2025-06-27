import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from dataflow.operators.generate.KnowledgeCleaning import (
    CorpusTextSplitter,
    KnowledgeExtractor,
    KnowledgeCleaner,
)
from dataflow.utils.storage import FileStorage
from dataflow.llmserving import LocalModelLLMServing

# 这里或许未来可以有个pipeline基类
class KBCleaningPipeline():
    def __init__(self):

        self.storage = FileStorage(
            first_entry_file_name="../dataflow/example/KBCleaningPipeline/pdf_test.json",
            cache_path="./cache",
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl",
        )

        # api_llm_serving = APILLMServing_request(
        #         api_url="http://123.129.219.111:3000/v1/chat/completions",
        #         model_name="gpt-4o",
        #         max_workers=100
        # )

        local_llm_serving = LocalModelLLMServing(
            model_name_or_path="/data0/models/Qwen2.5-7B-Instruct",
            max_tokens=1024,
            model_source="local"
        )

        self.knowledge_cleaning_step1 = KnowledgeExtractor(
            intermediate_dir="dataflow/example/KBCleaningPipeline/raw/"
        )

        self.knowledge_cleaning_step2 = CorpusTextSplitter(
            split_method="token",
            chunk_size=512,
            tokenizer_name="/data0/hzy/RARE/model_base/Qwen2.5-3B-Instruct",
        )

        self.knowledge_cleaning_step3 = KnowledgeCleaner(
            llm_serving=local_llm_serving,
        )
        # 未来或许可以维护一个类似nn.sequential的容器，方便添加并实例化多个算子
    def forward(self, raw_file):
        extracted=self.knowledge_cleaning_step1.run(
            storage=self.storage,
            raw_file=raw_file
        )
        
        self.knowledge_cleaning_step2.run(
            storage=self.storage.step(),
            input_file=extracted,
            output_key="raw_content",
        )

        self.knowledge_cleaning_step3.run(
            storage=self.storage.step(),
            input_key= "raw_content",
            output_key="cleaned",
        )

model = KBCleaningPipeline()
model.forward("/data0/hzy/test_mineru/muban.pdf")

