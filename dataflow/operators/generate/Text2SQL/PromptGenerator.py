from typing import Dict, Union, Optional, Tuple
from dataflow.generator.utils.Prompts import FinalPromptGeneration
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from tqdm import tqdm
from dataflow.generator.utils.Prompts import Text2SQLCotPrompt
from dataflow.generator.utils.LocalModelGenerator import LocalModelGenerator
from dataflow.generator.utils.APIGenerator_aisuite import APIGenerator_aisuite
from dataflow.generator.utils.APIGenerator_request import APIGenerator_request
from dataflow.utils.registry import GENERATOR_REGISTRY
from dataflow.utils.utils import get_logger
from dataflow.data import MyScaleStorage
import sqlite3
import os
import re

@GENERATOR_REGISTRY.register()
class PromptGenerator:
    def __init__(self, args: Dict):
        self.config = args
        self.prompt = FinalPromptGeneration()
        self.cot_output = Text2SQLCotPrompt()
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
        self.input_question_key = args.get('input_question_key', 'question')
        self.input_sql_key = args.get('input_sql_key', 'SQL')
        self.input_evidence_key = args.get('input_evidence_key', 'evidence')
        self.input_schema_key = args.get('input_schema_key', 'schema')
        self.num_threads = args.get('num_threads', 5)
        self.timeout = args.get('timeout', 60)
        self.db_root_path = args.get("db_root_path")  
        self.input_dbid_key = args.get("input_dbid_key")
        self.read_max_score = args.get("read_max_score")
        self.read_min_score = args.get("read_min_score")
        self.output_sft_prompt_key = args.get("output_sft_prompt_key")
        self.output_rl_prompt_key = args.get("output_rl_prompt_key")
        self.output_cot_key = args.get("output_cot_key")
        self.logger = get_logger()

    @staticmethod
    def get_desc(lang):
        if lang == "zh":
            return (
                "该算子用于构建完整的提示词和思维链推理过程。\n\n"
                "输入参数：\n"
                "- input_question_key: 问题键（如：question）\n"
                "- input_sql_key: SQL语句键（如：SQL）\n"
                "- input_schema_key: 数据库DDL信息键（如：ddl）\n"
                "- input_evidence_key: 输入中额外知识的键（如：evidence）\n"
                "- prompt_type: 提示词格式（如：omni-sql）\n"
                "- output_sft_prompt_key: SFT提示词输出键（如：sft_prompt）\n"
                "- output_rl_prompt_key: RL提示词输出键（如：rl_prompt）\n"
                "- output_cot_key: 思维链推理输出键（如：sft_output）\n"
                "- input_key: 输入数据主键（如：data）\n"
                "- input_dbid_key: 数据库ID键（如：db_id）\n"
                "- db_root_path: 数据库根目录（如：/mnt/public/data/.../dev_databases）\n"
                "- num_threads: 多线程并行数\n\n"
                "输出参数：\n"
                "- output_sft_prompt_key: SFT提示词\n"
                "- output_rl_prompt_key: RL提示词\n"
                "- output_cot_key: 思维链推理输出"
            )
        elif lang == "en":
            return (
                "This operator is used to construct complete prompts and chain-of-thought reasoning processes.\n\n"
                "Input parameters:\n"
                "- input_question_key: Key for the question (e.g., 'question')\n"
                "- input_sql_key: Key for the SQL statement (e.g., 'SQL')\n"
                "- input_schema_key: Key for the database DDL information (e.g., 'ddl')\n"
                "- input_evidence_key: Key for additional knowledge in the input (e.g., 'evidence')\n"
                "- prompt_type: Prompt format (e.g., 'omni-sql')\n"
                "- output_sft_prompt_key: Output key for SFT prompt (e.g., 'sft_prompt')\n"
                "- output_rl_prompt_key: Output key for RL prompt (e.g., 'rl_prompt')\n"
                "- output_cot_key: Output key for chain-of-thought reasoning (e.g., 'sft_output')\n"
                "- input_key: Main key for input data (e.g., 'data')\n"
                "- input_dbid_key: Key for database ID (e.g., 'db_id')\n"
                "- db_root_path: Root path of the databases (e.g., '/mnt/public/data/.../dev_databases')\n"
                "- num_threads: Number of parallel threads\n\n"
                "Output parameters:\n"
                "- output_sft_prompt_key: SFT prompt\n"
                "- output_rl_prompt_key: RL prompt\n"
                "- output_cot_key: Chain-of-thought reasoning output"
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
                [self.input_key], eval_stage=self.eval_stage, syn='syn_q', format='SFT_Single', maxmin_scores=[dict(zip(['min_score', 'max_score'], list(_))) for _ in list(zip(self.read_min_score, self.read_max_score))], stage=self.stage, pipeline_id=self.pipeline_id, category="text2sql_data"
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
            self.storage.write_data(output_rows, format="SFT_Single", Synthetic='syn_qa', stage=self.stage+1)
        else:
            dataframe.to_json(save_path, orient="records", lines=True)

    def generate_prompt(self, item: Dict, prompt_type: str) -> str:
        generated_prompt = None
        if prompt_type == 'dail-sql':
            generated_prompt = self.prompt.dial_sql_cot_prompt(
                    question=item.get(self.input_question_key),
                    sql=item.get(self.input_sql_key),
                    schema=item.get(self.input_schema_key),
                    evidence=item.get(self.input_evidence_key)
            )
        elif prompt_type == 'omni-sql':
            generated_prompt = self.prompt.omni_sql_cot_prompt(
                    question=item.get(self.input_question_key),
                    sql=item.get(self.input_sql_key),
                    schema=item.get(self.input_schema_key),
                    evidence=item.get(self.input_evidence_key)
            )
        return generated_prompt
    
    def generate_cot_synthesis_prompts(self, item: Dict, is_backup=False) -> str:
        if not is_backup:
            cot_synthesis_prompt = self.cot_output.text2sql_cot_prompt(
                item.get(self.input_schema_key),
                item.get(self.input_question_key),
                item.get(self.input_sql_key)
            )
        else:
            cot_synthesis_prompt = self.cot_output.text2sql_cot_prompt_backup(
                item.get(self.input_schema_key),
                item.get(self.input_question_key),
                item.get(self.input_sql_key)
            )
    
        return cot_synthesis_prompt
    
    def execute_sql(self, sql, db_path, timeout=10):
        conn = None
        try:
            conn = sqlite3.connect(db_path)
            conn.execute("PRAGMA busy_timeout = 5000")
            cursor = conn.cursor()
            cursor.execute(sql)
            result = cursor.fetchall()
            return result
        except sqlite3.Error as e:
            self.logger.error(f"SQL执行错误: {e}")
            return None
        finally:
            if conn:
                conn.close()
    
    def extract_sql(self, response):
        pattern = r"```sql\s*(.*?)\s*```"
        
        sql_blocks = re.findall(pattern, response, re.DOTALL)

        if sql_blocks:
            last_sql = sql_blocks[-1].strip()
            return last_sql
        else:
            return ""
    
    def _parse_response(self, response: str, gold_sql: str, db_path) -> Tuple[Optional[str], bool]:
        generated_sql = self.extract_sql(response)
        if not generated_sql:
            return None, False
        
        try:
            gen_result = self.execute_sql(generated_sql, db_path)
            gold_result = self.execute_sql(gold_sql, db_path)
            
            if gen_result is None or gold_result is None:
                return generated_sql, False
                
            return generated_sql, gen_result == gold_result 
        except Exception as e:
            self.logger.warning(f"SQL执行失败: {e}")
            return generated_sql, False

    def _parse_backup_response(self, response: str) -> Tuple[Optional[str], bool]:
        response = response.strip()
        if not response:
            return None, False

        lower_response = response.lower()
        keywords = ["let"] 
        
        for keyword in keywords:
            idx = lower_response.find(keyword)
            if idx != -1:
                return response[idx:], True
        
        return None, False

    def _process_item_with_retry(self, item: Dict, retry_count: int = 0, max_retries: int = 3) -> str:
        db_id = item.get(self.input_dbid_key)
        gold_sql = item.get(self.input_sql_key)
        db_path = os.path.join(self.db_root_path.rstrip('/'), db_id, f"{db_id}.sqlite")
        prompt = self.generate_cot_synthesis_prompts(item, False)
        
        while retry_count <= max_retries:
            try:
                response = self.model_generator.generate_text_from_input([prompt])
                parsed_response, flag = self._parse_response(response[0], gold_sql, db_path)
                
                if flag:
                    return parsed_response if parsed_response else ""
                
                retry_count += 1
            except Exception as e:
                self.logger.warning(f"Attempt {retry_count} failed: {e}")
                retry_count += 1

        try:
            backup_prompt = self.generate_cot_synthesis_prompts(item, True)
            backup_response = self.model_generator.generate_text_from_input([backup_prompt])
            parsed_backup_response, success = self._parse_backup_response(backup_response[0])
            return parsed_backup_response if success and parsed_backup_response else ""
        except Exception as e:
            self.logger.error(f"Backup processing failed: {e}")
            return ""

    
    def _process_item(self, item: Dict) -> Dict: 
        sft_prompt = self.generate_prompt(item, prompt_type="omni-sql")
        rl_prompt = self.generate_prompt(item, prompt_type="dail-sql")
        cot_output = self._process_item_with_retry(item)

        return {
            **item,
            self.output_sft_prompt_key: sft_prompt if sft_prompt else '',
            self.output_rl_prompt_key: rl_prompt if rl_prompt else '',
            self.output_cot_key: cot_output if cot_output else ''
        }

    def run(self) -> None:
        self.logger.info("Starting prompt generation...")
        items = self._load_input().to_dict('records')
             
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = {
                executor.submit(self._process_item, item): item['id']
                for item in tqdm(items, desc="Submitting tasks", unit="item")
            }

            results = []
            with tqdm(total=len(futures), desc="Processing", unit="item") as pbar:
                for future in as_completed(futures):
                    item_id = futures[future]
                    try:
                        results.append(future.result())
                    except Exception as e:
                        self.logger.error(f"Error processing id={item_id}: {e}")
                        original_item = next((item for item in items if item['id'] == item_id), None)
                        if original_item:
                            results.append({
                                **original_item, 
                                self.output_sft_prompt_key: "", 
                                self.output_rl_prompt_key: "",
                                self.output_cot_key: ""
                            })
                    
                    pbar.update(1)

        id_to_index = {item['id']: idx for idx, item in enumerate(items)}
        results.sort(key=lambda x: id_to_index[x['id']]) 
        self._write_output(self.output_file, pd.DataFrame(results), None)
