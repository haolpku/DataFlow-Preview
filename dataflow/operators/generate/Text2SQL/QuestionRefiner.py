from typing import Dict, Union
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataflow.utils.registry import GENERATOR_REGISTRY
from dataflow.generator.utils.Prompts import QuestionRefinePrompt
from dataflow.generator.utils.APIGenerator_request import APIGenerator_request
from dataflow.generator.utils.LocalModelGenerator import LocalModelGenerator
from dataflow.generator.utils.APIGenerator_aisuite import APIGenerator_aisuite
from dataflow.utils.utils import get_logger
from dataflow.data import MyScaleStorage

@GENERATOR_REGISTRY.register()
class QuestionRefiner:
    def __init__(self, args: Dict):        
        self.config = args
        self.prompt = QuestionRefinePrompt()
        self.model_generator = self.__init_model__()

        if "db_name" in args.keys():
            self.storage = MyScaleStorage(args['db_port'], args['db_name'], args['table_name'])
            self.input_file = None
            self.output_file= None
        else:
            self.input_file = args.get("input_file")
            self.output_file= args.get("output_file")
        self.eval_stage = args.get('eval_stage', 2)
        self.stage = args.get('stage', 0)
        self.pipeline_id = args.get('pipeline_id', 'default_pipeline')

        self.input_key = args.get("input_key", "data")
        self.output_refined_question_key = args.get('output_refined_question_key')
        self.input_db_key = args.get('input_db_key', 'id')
        self.num_threads = args.get('num_threads', 5)
        self.max_retries = args.get('max_retries', 3)
        self.input_question_key = args.get('input_question_key', 'question')
        self.read_max_score = args.get("read_max_score")
        self.read_min_score = args.get("read_min_score")
        self.logger = get_logger()

    @staticmethod
    def get_desc(lang):
        if lang == "zh":
            return (
                "该算子用于对已有的自然语言问题进行润色改写。\n\n"
                "输入参数：\n"
                "- input_question_key: 问题键\n"
                "- num_threads: 多线程并行数\n\n"
                "输出参数：\n"
                "- output_refined_question_key: 生成的润色后问题的key"
            )
        elif lang == "en":
            return (
                "This operator is used to refine and rewrite existing natural language questions.\n\n"
                "Input parameters:\n"
                "- input_question_key: Question key\n"
                "- num_threads: Number of parallel threads\n\n"
                "Output parameters:\n"
                "- output_refined_question_key: The key for the generated refined question"
            )
        else:
            return "AnswerExtraction_qwenmatheval performs mathematical answer normalization and standardization."


    def __init_model__(self) -> Union[LocalModelGenerator, APIGenerator_aisuite, APIGenerator_request]:
        generator_type = self.config["generator_type"].lower()
        if generator_type == "local":
            return LocalModelGenerator(self.config)
        elif generator_type == "aisuite":
            return APIGenerator_aisuite(self.config)
        elif generator_type == "request":
            return APIGenerator_request(self.config)
        raise ValueError(f"Invalid generator type: {generator_type}")

    def _load_input(self):
        if hasattr(self, 'storage'):
            value_list = self.storage.read_json(
                [self.input_key], eval_stage=self.eval_stage, syn='', format='PT', maxmin_scores=[dict(zip(['min_score', 'max_score'], list(_))) for _ in list(zip(self.read_min_score, self.read_max_score))], stage=self.stage, pipeline_id=self.pipeline_id, category="text2sql_data"
            )
            return pd.DataFrame([
                {**item['data'], 'id': str(item['id'])}
                for item in value_list
            ])
        else:
            return pd.read_json(self.input_file, lines=True)

    def _write_output(self, save_path, dataframe, extractions):
        if hasattr(self, 'storage'):
            output_rows = dataframe.where(pd.notnull(dataframe), None).to_dict(orient="records")
            self.storage.write_data(output_rows, format="SFT_Single", Synthetic='syn_q', stage=self.stage+1)
        else:
            dataframe.to_json(save_path, orient="records", lines=True)

    def _generate_prompt(self, item: Dict) -> str:
        return self.prompt.question_refine_prompt(item['question'])
    
    def _parse_response(self, response: str, original_question: str) -> str:
        if not response:
            return original_question
            
        response_upper = response.upper()
        if "RESULT: NO" in response_upper:
            return original_question
            
        try:
            result_line = next(
                line for line in response.split('\n') 
                if line.upper().startswith("RESULT:")
            )
            return result_line.split("RESULT:", 1)[1].strip()
        except (StopIteration, IndexError):
            self.logger.warning(f"Unexpected response format: {response[:200]}...")
            return original_question

    def _process_item_with_retry(self, item: Dict, retry_count: int = 2) -> Dict:
        try:
            prompt = self._generate_prompt(item)
            response = self.model_generator.generate_text_from_input([prompt])
            parsed_response = self._parse_response(response[0], item['question'])
            
            return {
                **item,
                self.output_refined_question_key: parsed_response
            }
        
        except Exception as e:
            if retry_count < self.max_retries:
                self.logger.warning(f"Retrying {item.get('id')} (attempt {retry_count + 1}): {str(e)}")
                return self._process_item_with_retry(item, retry_count + 1)
            # self.logger.error(f"Failed after {self.max_retries} retries for item: {e}")

            return {
                **item,
                self.output_refined_question_key: item['question']
            }

    def run(self) -> None:
        self.logger.info("Starting QuestionRefiner...")
        items = self._load_input().to_dict('records')
                        
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = {
                executor.submit(self._process_item_with_retry, item): item['id']
                for item in tqdm(items, desc="Submitting tasks", unit="item")
            }

            results = []
            with tqdm(total=len(items), desc="Processing items", unit="item") as pbar:
                for future in as_completed(futures):
                    item_id = futures[future]
                    try:
                        results.append(future.result())
                    except Exception as e:
                        self.logger.error(f"Fatal error for id={item_id}: {e}")
                        original_item = next((item for item in items if item['id'] == item_id), None)
                        if original_item:
                            results.append(
                                {
                                    **original_item, 
                                    self.output_refined_question_key: original_item[self.input_question_key]
                                }
                            )
                        
                    pbar.update(1)

        id_to_index = {item['id']: idx for idx, item in enumerate(items)}
        results.sort(key=lambda x: id_to_index[x['id']]) 
        self._write_output(self.output_file, pd.DataFrame(results), None)
