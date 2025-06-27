from dataflow.prompts.kbcleaning import KnowledgeCleanerPrompt
import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC

@OPERATOR_REGISTRY.register()
class KnowledgeCleaner(OperatorABC):
    '''
        KnowledgeCleaner is a class that cleans knowledge for RAG to make them more accurate, reliable and readable.
    '''
    def __init__(self, llm_serving: LLMServingABC):
        self.logger = get_logger()
        self.prompts = KnowledgeCleanerPrompt(lang="zh")    
        self.llm_serving = llm_serving
    
    @staticmethod
    def get_desc(self, lang):
        if lang == "zh":
            return (
                "哈哈哈"
            )
        elif lang == "en":
            return (
                "hahahah"
            )
        else:
            return "hahaha"

    def _validate_dataframe(self, dataframe: pd.DataFrame):
        required_keys = [self.input_key]
        forbidden_keys = [self.output_key]

        missing = [k for k in required_keys if k not in dataframe.columns]
        conflict = [k for k in forbidden_keys if k in dataframe.columns]

        if missing:
            raise ValueError(f"Missing required column(s): {missing}")
        if conflict:
            raise ValueError(f"The following column(s) already exist and would be overwritten: {conflict}")

    def _reformat_prompt(self, dataframe):
        """
        Reformat the prompts in the dataframe to generate questions.
        """
        raw_contents = dataframe[self.input_key].tolist()
        inputs = [self.prompts.Classic_COT_Prompt(raw_content) for raw_content in raw_contents]

        return inputs

    def run(
        self, 
        storage: DataFlowStorage, 
        input_key:str = "raw_content", 
        output_key:str = "cleaned"
        ):
        '''
        Runs the knowledge cleaning process, reading from the input key and saving results to output key.
        '''
        self.input_key, self.output_key = input_key, output_key
        dataframe = storage.read("dataframe")
        self._validate_dataframe(dataframe)
        formatted_prompts = self._reformat_prompt(dataframe)
        cleaned = self.llm_serving.generate_from_input(formatted_prompts)

        dataframe[self.output_key] = cleaned
        output_file = storage.write(dataframe)
        self.logger.info(f"Results saved to {output_file}")

        return [output_key]