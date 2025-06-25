from dataflow.operators.generate.Text2SQL import *
from dataflow.utils.storage import FileStorage
from dataflow.llmserving import APILLMServing_request


class Text2SQLPipeline():
    def __init__(self):

        self.storage = FileStorage(
            first_entry_file_name="/mnt/public/data/scy/DataFlow-Preview/cache/dataflow_cache_step_2.jsonl",
            cache_path="./cache",
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl",
        )

        api_llm_serving = APILLMServing_request(
                api_url="http://123.129.219.111:3000/v1/chat/completions",
                model_name="gpt-4o",
                max_workers=100
        )
        
        self.sql_filter_step1 = SQLFilter(
            llm_serving=api_llm_serving
        )

        self.sql_difficulty_classifier_step2 = SQLDifficultyClassifier(
            llm_serving=api_llm_serving
        )

        self.schema_linking_step3 = SchemaLinking()

        self.database_schema_extractor_step4 = DatabaseSchemaExtractor()

        self.extra_knowledge_generator_step5 = ExtraKnowledgeGenerator(
            llm_serving=api_llm_serving
        )

        self.question_refiner_step6 = QuestionRefiner(
            llm_serving=api_llm_serving
        )

        self.prompt_generator_step7 = PromptGenerator(
            llm_serving=api_llm_serving
        )

        self.text2sql_difficulty_classifier_step8 = Text2SQLDifficultyClassifier(
            llm_serving=api_llm_serving
        )
        
        
    def forward(self):

        self.sql_filter_step1.run(
            storage=self.storage.step(),
            input_key="instruction",
            output_key="sql_filter_output"
        )

        self.sql_difficulty_classifier_step2.run(
            storage=self.storage.step(),
            input_key="sql_filter_output",
            output_key="sql_difficulty_classifier_output"
        )

        self.schema_linking_step3.run(
            storage=self.storage.step(),
            input_key="sql_difficulty_classifier_output",
            output_key="schema_linking_output"
        )

        self.database_schema_extractor_step4.run(
            storage=self.storage.step(),
            input_key="schema_linking_output",
            output_key="database_schema_extractor_output"
        )

        self.extra_knowledge_generator_step5.run(
            storage=self.storage.step(),
            input_key="database_schema_extractor_output",
            output_key="extra_knowledge_generator_output"
        )

        self.question_refiner_step6.run(
            storage=self.storage.step(),
            input_key="extra_knowledge_generator_output",
            output_key="question_refiner_output"
        )

        self.prompt_generator_step7.run(
            storage=self.storage.step(),
            input_key="question_refiner_output",
            output_key="prompt_generator_output"
        )

        self.text2sql_difficulty_classifier_step8.run(
            storage=self.storage.step(),
            input_key="prompt_generator_output",
            output_key="text2sql_difficulty_classifier_output"
        )
        
        
model = Text2SQLPipeline()
model.forward()

