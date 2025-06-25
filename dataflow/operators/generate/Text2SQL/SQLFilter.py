from dataflow.generator.utils.Prompts import TextSQLConsistencyPrompt
from dataflow.generator.utils.LocalModelGenerator import LocalModelGenerator
from dataflow.generator.utils.APIGenerator_aisuite import APIGenerator_aisuite
from dataflow.generator.utils.APIGenerator_request import APIGenerator_request
from dataflow.utils.utils import get_logger
from func_timeout import func_timeout, FunctionTimedOut
from tqdm import tqdm
import sqlite3
import sys
import re
import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
from dataflow.utils.registry import GENERATOR_REGISTRY
from dataflow.data import MyScaleStorage

@GENERATOR_REGISTRY.register()
class SQLFilter:
    def __init__(self, args: dict):
        '''
        Initialize the SQLFilter with the provided configuration.
        '''
        self.config = args
        self.prompt = TextSQLConsistencyPrompt()
        self.model_generator = self.__init_model__()

        # self.storage = MyScaleStorage(args['db_port'], args['db_name'], args['table_name'])

        if "db_name" in args.keys():
            self.storage = MyScaleStorage(args['db_port'], args['db_name'], args['table_name'])
            self.input_file = None
            self.output_file= None
        else:
            self.input_file = args.get("input_file")
            self.output_file= args.get("output_file")

        # Extract the configurations from the provided dictionary
        self.db_root_path = self.config.get("db_root_path")
        self.num_cpus = self.config.get("num_cpus", 1)
        self.meta_time_out = self.config.get("meta_time_out", 120.0)
        self.input_key = self.config.get("input_key", "data")
        
        self.output_note_file = self.config.get("output_note_file")
        self.input_sql_key = self.config.get("input_sql_key", "SQL")
        self.input_dbid_key = self.config.get("input_dbid_key", "db_id")
        self.input_question_key = self.config.get("input_question_key", "question")
        self.input_evidence_key = self.config.get("input_evidence_key", "")
        self.output_consistency_key = self.config.get("output_consistency_key", "consistency")
        self.output_reason_key = self.config.get("output_reason_key", "consistency_reason")
        self.output_goldcorrect_key = self.config.get("output_goldcorrect_key", "is_correct")
        self.output_goldresult_key = self.config.get("output_goldresult_key", "exec_result")
        self.output_classification_key = self.config.get("output_classification_key", "classification_result")
        self.logger = get_logger()
        self.eval_stage = self.config.get('eval_stage', 2)
        self.stage = args.get('stage', 0)
        self.pipeline_id = args.get('pipeline_id', 'default_pipeline')
        
    @staticmethod
    def get_desc(lang):
        if lang == "zh":
            return (
                "该算子用于筛选SQL执行异常和模型判定SQL与自然语言问题是否一致。\n\n"
                "输入参数：\n"
                "- input_sql_key: SQL语句键\n"
                "- input_question_key: 问题键\n"
                "- input_evidence_key: 证据键，对问题的补充说明（可选）\n"
                "- input_dbid_key: 数据库ID键，数据库名\n"
                "- db_root_path: 数据库目录路径\n"
                "- num_cpus: 并行线程数\n"
                "- meta_time_out: gold sql执行超时时间\n\n"
                "输出参数：\n"
                "- output_goldcorrect_key：gold sql是否正常执行 key，只在note文件中输出\n"
                "- output_goldresult_key：gold sql执行结果，只在note文件中输出\n"
                "- output_consistency_key：question和sql是否一致，只在note中输出\n"
                "- output_reason_key：一致性的理由，只在note中输出"
            )
        elif lang == "en":
            return (
                "This operator filters SQL execution errors and checks the consistency between SQL and natural language questions using a model.\n\n"
                "Input parameters:\n"
                "- input_sql_key: Key for SQL statements\n"
                "- input_question_key: Key for questions\n"
                "- input_evidence_key: Key for evidence, additional explanation for the question (optional)\n"
                "- input_dbid_key: Key for database IDs, which is the database name\n"
                "- db_root_path: Path to the database directory\n"
                "- num_cpus: Number of parallel threads\n"
                "- meta_time_out: Timeout for executing gold SQL statements\n"  
                "\nOutput parameters:\n"
                "- output_goldcorrect_key: Key indicating whether the gold SQL executed successfully, only in the note file\n"
                "- output_goldresult_key: Execution results of the gold SQL, only in the note file\n"
                "- output_consistency_key: Key indicating whether the question and SQL are consistent, only in the note file\n"
                "- output_reason_key: Key for the reason of consistency, only in the note file"
            )
        else:
            return "AnswerExtraction_qwenmatheval performs mathematical answer normalization and standardization."

    def _load_input(self):
        if hasattr(self, 'storage'):
            value_list = self.storage.read_json(
                [self.input_key], eval_stage=self.eval_stage, syn='', format='PT', stage=self.stage, pipeline_id=self.pipeline_id, category="text2sql_data"
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
            output_rows = [
                {
                    "id": row.get("id"),
                    "score": 1 if row.get(self.output_consistency_key) and row.get(self.output_goldcorrect_key) else 0,
                    "info": row.get(self.output_reason_key, "")
                }
                for row in output_rows
            ]
            self.storage.write_eval(output_rows, algo_name="SQLFilter", score_key="score", stage=self.stage+1)
        else:
            output_dir = os.path.dirname(save_path)
            os.makedirs(output_dir, exist_ok=True)
            dataframe.to_json(self.output_file, orient="records", lines=True, force_ascii=False)

    @staticmethod
    def execute_sql(sql, db_path):
        '''
        Execute the given SQL statement and return the results.
        '''
        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql)
        result = cursor.fetchall()
        return result
    
    @staticmethod
    def execute_model(ground_truth, db_place, idx, meta_time_out, logger):
        '''
        Execute SQL model with timeout and error handling.
        '''
        is_correct = True
        try:
            results = func_timeout(meta_time_out, SQLFilter.execute_sql,
                        args=(ground_truth, db_place))
            return {"idx": idx,"is_correct": is_correct, "results": results}
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt")
            sys.exit(0)
        except FunctionTimedOut:
            logger.info(f"timeout when execute idx {idx}")
            result = (f'timeout')
            is_correct = False
            return {"idx": idx,"is_correct": is_correct, "results": result}
        except Exception as e:
            logger.info(f"error: {e} when execute idx {idx}")
            result = (f'error:{e}')  # possibly len(query) > 512 or not executable
            is_correct = False
            return {"idx": idx,"is_correct": is_correct, "results": result}
        
    
    def run_sqls_parallel(self, datas, db_root_path, num_cpus, meta_time_out, exec_result=[]):
        '''
        Execute the given SQL statements in parallel and return sorted results.
        '''
        pbar = tqdm(total=len(datas), desc="Executing SQLs")

        def wrap_task(ground_truth, db_place, idx, timeout):
            try:
                return SQLFilter.execute_model(ground_truth, db_place, idx, timeout, self.logger)
            except Exception as e:
                self.logger.error(f"Error executing SQL idx={idx}: {e}")
                return {"idx": idx, "error": str(e)}

        with ThreadPoolExecutor(max_workers=num_cpus) as executor:
            futures = []

            for i, data_pair in enumerate(datas):
                ground_truth = data_pair[self.input_sql_key]
                db_id = data_pair[self.input_dbid_key].replace('\n', '')
                db_id = re.sub(r'[^A-Za-z0-9_]', '', db_id)
                db_place = os.path.join(db_root_path.rstrip('/'), db_id, f"{db_id}.sqlite")

                future = executor.submit(wrap_task, ground_truth, db_place, i, meta_time_out)
                futures.append(future)

            for future in as_completed(futures):
                try:
                    result = future.result()
                    exec_result.append(result)
                except Exception as e:
                    self.logger.error(f"Error retrieving result from future: {e}")
                    exec_result.append({"idx": -1, "error": str(e)})
                pbar.update()

        pbar.close()
        return sorted(exec_result, key=lambda x: x['idx'])
    
    
    def __init_model__(self):
        '''
        Initialize the model generator based on the configuration.
        '''
        generator_type = self.config.get("generator_type", "local").lower()
        if generator_type == "local":
            return LocalModelGenerator(self.config)
        elif generator_type == "aisuite":
            return APIGenerator_aisuite(self.config)
        elif generator_type == "request":
            return APIGenerator_request(self.config)
        else:
            self.logger.error(f"Invalid generator type: {generator_type}")
            raise ValueError(f"Invalid generator type: {generator_type}")
        
    def _reformat_prompt(self, dataframe):
        '''
        Reformat the prompts in the dataframe to generate questions.
        '''
        formatted_prompts = []
        if self.input_evidence_key == "":
            for index, row in dataframe.iterrows():
                sql = row[self.input_sql_key]
                question = row[self.input_question_key]
                used_prompt = self.prompt.text_sql_consistency_prompt(question, sql)
                formatted_prompts.append(used_prompt.strip())
        else:
            dataframe["evidence_question"] = dataframe[self.input_evidence_key] + "\n" + dataframe[self.input_question_key]
            for index, row in dataframe.iterrows():
                sql = row[self.input_sql_key]
                question = row[self.input_question_key]
                evidence = row[self.input_evidence_key]
                used_prompt = self.prompt.text_sql_consistency_prompt(question, sql, evidence)
                formatted_prompts.append(used_prompt.strip())
        return formatted_prompts
    
    def process_single_question(self, question):
        try:
            # 注意：这里传入的是单个元素的列表，因为 generate_text_from_input 是 batch 接口
            result = self.model_generator.generate_text_from_input([question])
            return result[0] if result else ""
        except Exception as e:
            self.logger.error(f"Error processing question: {e}")
            return ""

    def run(self):
        '''
        Main execution method, used to read the input file, execute SQL statements, and save the results to the output file.
        '''
        # Read input file: only accept jsonl format
        dataframe = self._load_input()

        # Ensure the input and output keys are correctly set
        self._validate_dataframe(dataframe)

        # Execute gold sqls from datas
        selected_columns = [self.input_sql_key, self.input_dbid_key]
        datas = dataframe[selected_columns].to_dict('records')
        self.logger.info(f"Original data volume: {len(datas)}")
        exec_result = self.run_sqls_parallel(datas, self.db_root_path, self.num_cpus, self.meta_time_out, exec_result=[])

        # Reformat the prompts for question generation
        formatted_prompts = self._reformat_prompt(dataframe)

        # Generate responses using the model
        # responses = self.model_generator.generate_text_from_input(formatted_prompts)
        responses = []
        with ThreadPoolExecutor(max_workers=self.num_cpus) as executor:
            futures = {
                executor.submit(self.process_single_question, question): idx
                for idx, question in enumerate(tqdm(formatted_prompts, desc="Submitting tasks"))
            }

            results_buffer = [None] * len(formatted_prompts)

            for future in tqdm(as_completed(futures), total=len(formatted_prompts), desc="Collecting responses"):
                idx = futures[future]
                try:
                    result = future.result()
                    results_buffer[idx] = result
                except Exception as e:
                    self.logger.error(f"Error retrieving future result: {e}")
                    results_buffer[idx] = ""

        responses = results_buffer

        dataframe_note = copy.deepcopy(dataframe)
        dataframe_del = copy.deepcopy(dataframe) 

        # dataframe[self.output_classification_key] = [
        #     1 if item["is_correct"] else 0 
        #     for item in exec_result
        # ]

        dataframe_note[self.output_goldcorrect_key] = [item["is_correct"] for item in exec_result]
        dataframe_note[self.output_goldresult_key] = [item["results"] for item in exec_result]

        # Build a mask of which rows are correct
        is_correct_mask = dataframe_note[self.output_goldcorrect_key]

        # Remove rows from dataframe_del where is_correct is False
        dataframe_del = dataframe_del[is_correct_mask].reset_index(drop=True)

        for (idx, row), response in zip(dataframe.iterrows(), responses):
            try:
                conclusion = None
                response_lower = response.lower()
                
                if "conclusion:" in response_lower:
                    conclusion_part = response_lower.split("conclusion:")[1].strip()
                    if "analysis:" in response_lower:
                        analysis_part = response_lower.split("conclusion")[0].split("analysis:")[1].strip()
                    else:
                        analysis_part = ""
                else:
                    raise ValueError("Response does not contain 'conclusion:'")
                
                if "no" in conclusion_part:
                    conclusion = False
                    # row[self.output_classification_key] = 0
                elif "yes" in conclusion_part:
                    conclusion = True
                else:
                    raise ValueError("Could not determine conclusion from response")
                
                dataframe_note.at[idx, self.output_consistency_key] = conclusion
                dataframe_note.at[idx, self.output_reason_key] = analysis_part.strip()
                
            except Exception as e:
                self.logger.warning(f"Failed to judge the consistency of the SQL: {e}")
                dataframe_note.at[idx, self.output_consistency_key] = False
                dataframe_note.at[idx, self.output_reason_key] = f"Failed to judge: {str(e)}"

        # Ensure output directory exists
        # output_dir = os.path.dirname(self.output_file)
        # os.makedirs(output_dir, exist_ok=True)
        # output_note_dir = os.path.dirname(self.output_note_file)
        # os.makedirs(output_note_dir, exist_ok=True)
        self.logger.info(f"Filtered data volume: {len(dataframe_del)}")

        # Save DataFrame to JSON file
        self._write_output(self.output_file, dataframe_note, None)
        # dataframe_del.to_json(self.output_file, orient="records", lines=True, force_ascii=False)
        # dataframe_note.to_json(self.output_note_file, orient="records", lines=True, force_ascii=False)


    def _validate_dataframe(self, dataframe: pd.DataFrame):
        '''
        Helper method to validate the input dataframe columns.
        '''
        # Check if the input sql key exists in the dataframe
        if self.input_sql_key not in dataframe.columns:
            self.logger.error(f"input_sql_key: {self.input_sql_key} not found in the dataframe.")
            raise ValueError(f"input_sql_key: {self.input_sql_key} not found in the dataframe.")
        
        # Check if the input dbid key exists in the dataframe
        if self.input_dbid_key not in dataframe.columns:
            self.logger.error(f"input_dbid_key: {self.input_dbid_key} not found in the dataframe.")
            raise ValueError(f"input_dbid_key: {self.input_dbid_key} not found in the dataframe.")
        
        # Check if the input question key exists in the dataframe
        if self.input_question_key not in dataframe.columns:
            self.logger.error(f"input_question_key: {self.input_question_key} not found in the dataframe.")
            raise ValueError(f"input_question_key: {self.input_question_key} not found in the dataframe.")
        
        # Check if the input evidence key exists in the dataframe
        if self.input_evidence_key != "" and self.input_evidence_key not in dataframe.columns:
            self.logger.error(f"input_evidence_key: {self.input_evidence_key} not found in the dataframe.")
            raise ValueError(f"input_evidence_key: {self.input_evidence_key} not found in the dataframe.")
        
        # Check if the output key already exists in the dataframe
        if self.output_goldcorrect_key in dataframe.columns:
            self.logger.error(f"Found {self.output_goldcorrect_key} in the dataframe, which would overwrite an existing column. Please use a different output_key.")
            raise ValueError(f"Found {self.output_goldcorrect_key} in the dataframe, which would overwrite an existing column. Please use a different output_key.")

        # Check if the output result key already exists in the dataframe
        if self.output_goldresult_key in dataframe.columns:
            self.logger.error(f"Found {self.output_goldresult_key} in the dataframe, which would overwrite an existing column. Please use a different output_key.")
            raise ValueError(f"Found {self.output_goldresult_key} in the dataframe, which would overwrite an existing column. Please use a different output_key.")

        # Check if the output consistency key already exists in the dataframe
        if self.output_consistency_key in dataframe.columns:
            self.logger.error(f"Found {self.output_consistency_key} in the dataframe, which would overwrite an existing column. Please use a different output_key.")
            raise ValueError(f"Found {self.output_consistency_key} in the dataframe, which would overwrite an existing column. Please use a different output_key.")

        # Check if the output reason key already exists in the dataframe
        if self.output_reason_key in dataframe.columns:
            self.logger.error(f"Found {self.output_reason_key} in the dataframe, which would overwrite an existing column. Please use a different output_key.")
            raise ValueError(f"Found {self.output_reason_key} in the dataframe, which would overwrite an existing column. Please use a different output_key.")
