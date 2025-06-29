from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig, XLMRobertaXLModel
from transformers.trainer_utils import set_seed
import torch
import torch.nn as nn
import numpy as np
import random
import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage

class SchemaItemClassifier(nn.Module):
    def __init__(self, model_name_or_path, mode):
        super(SchemaItemClassifier, self).__init__()
        if mode in ["eval", "test"]:
            config = AutoConfig.from_pretrained(model_name_or_path)
            self.plm_encoder = XLMRobertaXLModel(config)
        elif mode == "train":
            self.plm_encoder = XLMRobertaXLModel.from_pretrained(model_name_or_path)
        else:
            raise ValueError()

        self.plm_hidden_size = self.plm_encoder.config.hidden_size

        # column cls head
        self.column_info_cls_head_linear1 = nn.Linear(self.plm_hidden_size, 256)
        self.column_info_cls_head_linear2 = nn.Linear(256, 2)
        
        # column bi-lstm layer
        self.column_info_bilstm = nn.LSTM(
            input_size = self.plm_hidden_size,
            hidden_size = int(self.plm_hidden_size/2),
            num_layers = 2,
            dropout = 0,
            bidirectional = True
        )

        # linear layer after column bi-lstm layer
        self.column_info_linear_after_pooling = nn.Linear(self.plm_hidden_size, self.plm_hidden_size)

        # table cls head
        self.table_name_cls_head_linear1 = nn.Linear(self.plm_hidden_size, 256)
        self.table_name_cls_head_linear2 = nn.Linear(256, 2)
        
        # table bi-lstm pooling layer
        self.table_name_bilstm = nn.LSTM(
            input_size = self.plm_hidden_size,
            hidden_size = int(self.plm_hidden_size/2),
            num_layers = 2,
            dropout = 0,
            bidirectional = True
        )
        # linear layer after table bi-lstm layer
        self.table_name_linear_after_pooling = nn.Linear(self.plm_hidden_size, self.plm_hidden_size)

        # activation function
        self.leakyrelu = nn.LeakyReLU()
        self.tanh = nn.Tanh()

        # table-column cross-attention layer
        self.table_column_cross_attention_layer = nn.MultiheadAttention(embed_dim = self.plm_hidden_size, num_heads = 8)

        # dropout function, p=0.2 means randomly set 20% neurons to 0
        self.dropout = nn.Dropout(p = 0.2)

    def table_column_cross_attention(
        self,
        table_name_embeddings_in_one_db, 
        column_info_embeddings_in_one_db, 
        column_number_in_each_table
    ):
        table_num = table_name_embeddings_in_one_db.shape[0]
        table_name_embedding_attn_list = []
        for table_id in range(table_num):
            table_name_embedding = table_name_embeddings_in_one_db[[table_id], :]
            column_info_embeddings_in_one_table = column_info_embeddings_in_one_db[
                sum(column_number_in_each_table[:table_id]) : sum(column_number_in_each_table[:table_id+1]), :]
            
            table_name_embedding_attn, _ = self.table_column_cross_attention_layer(
                table_name_embedding,
                column_info_embeddings_in_one_table,
                column_info_embeddings_in_one_table
            )

            table_name_embedding_attn_list.append(table_name_embedding_attn)
        
        table_name_embeddings_in_one_db = table_name_embeddings_in_one_db + torch.cat(table_name_embedding_attn_list, dim = 0)

        table_name_embeddings_in_one_db = torch.nn.functional.normalize(table_name_embeddings_in_one_db, p=2.0, dim=1)

        return table_name_embeddings_in_one_db

    def table_column_cls(
        self,
        encoder_input_ids,
        encoder_input_attention_mask,
        batch_aligned_column_info_ids,
        batch_aligned_table_name_ids,
        batch_column_number_in_each_table
    ):
        batch_size = encoder_input_ids.shape[0]
        
        encoder_output = self.plm_encoder(
            input_ids = encoder_input_ids,
            attention_mask = encoder_input_attention_mask,
            return_dict = True
        ) 

        batch_table_name_cls_logits, batch_column_info_cls_logits = [], []

        for batch_id in range(batch_size):
            column_number_in_each_table = batch_column_number_in_each_table[batch_id]
            sequence_embeddings = encoder_output["last_hidden_state"][batch_id, :, :] 

            aligned_table_name_ids = batch_aligned_table_name_ids[batch_id]
            aligned_column_info_ids = batch_aligned_column_info_ids[batch_id]

            table_name_embedding_list, column_info_embedding_list = [], []

            for table_name_ids in aligned_table_name_ids:
                table_name_embeddings = sequence_embeddings[table_name_ids, :]
                
                output_t, (hidden_state_t, cell_state_t) = self.table_name_bilstm(table_name_embeddings)
                table_name_embedding = hidden_state_t[-2:, :].view(1, self.plm_hidden_size)
                table_name_embedding_list.append(table_name_embedding)
            table_name_embeddings_in_one_db = torch.cat(table_name_embedding_list, dim = 0)
            table_name_embeddings_in_one_db = self.leakyrelu(self.table_name_linear_after_pooling(table_name_embeddings_in_one_db))
            
            for column_info_ids in aligned_column_info_ids:
                column_info_embeddings = sequence_embeddings[column_info_ids, :]
                
                output_c, (hidden_state_c, cell_state_c) = self.column_info_bilstm(column_info_embeddings)
                column_info_embedding = hidden_state_c[-2:, :].view(1, self.plm_hidden_size)
                column_info_embedding_list.append(column_info_embedding)
            column_info_embeddings_in_one_db = torch.cat(column_info_embedding_list, dim = 0)
            column_info_embeddings_in_one_db = self.leakyrelu(self.column_info_linear_after_pooling(column_info_embeddings_in_one_db))

            table_name_embeddings_in_one_db = self.table_column_cross_attention(
                table_name_embeddings_in_one_db, 
                column_info_embeddings_in_one_db, 
                column_number_in_each_table
            )
            
            table_name_embeddings_in_one_db = self.table_name_cls_head_linear1(table_name_embeddings_in_one_db)
            table_name_embeddings_in_one_db = self.dropout(self.leakyrelu(table_name_embeddings_in_one_db))
            table_name_cls_logits = self.table_name_cls_head_linear2(table_name_embeddings_in_one_db)

            column_info_embeddings_in_one_db = self.column_info_cls_head_linear1(column_info_embeddings_in_one_db)
            column_info_embeddings_in_one_db = self.dropout(self.leakyrelu(column_info_embeddings_in_one_db))
            column_info_cls_logits = self.column_info_cls_head_linear2(column_info_embeddings_in_one_db)

            batch_table_name_cls_logits.append(table_name_cls_logits)
            batch_column_info_cls_logits.append(column_info_cls_logits)

        return batch_table_name_cls_logits, batch_column_info_cls_logits

    def forward(
        self,
        encoder_input_ids,
        encoder_attention_mask,
        batch_aligned_column_info_ids,
        batch_aligned_table_name_ids,
        batch_column_number_in_each_table,
    ):  
        batch_table_name_cls_logits, batch_column_info_cls_logits \
            = self.table_column_cls(
                encoder_input_ids,
                encoder_attention_mask,
                batch_aligned_column_info_ids,
                batch_aligned_table_name_ids,
                batch_column_number_in_each_table
        )

        return {
            "batch_table_name_cls_logits" : batch_table_name_cls_logits, 
            "batch_column_info_cls_logits": batch_column_info_cls_logits
        }
    
class SchemaItemClassifierInference():
    def __init__(self, model_save_path):
        set_seed(42)
        self.tokenizer = AutoTokenizer.from_pretrained(model_save_path, add_prefix_space = True)
        self.model = SchemaItemClassifier(model_save_path, "test")
        self.model.load_state_dict(torch.load(model_save_path + "/dense_classifier.pt", map_location=torch.device('cpu')), strict=False)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()
    
    def prepare_inputs_and_labels(self, sample):
        table_names = [table["table_name"] for table in sample["schema"]["schema_items"]]
        column_names = [table["column_names"] for table in sample["schema"]["schema_items"]]
        column_num_in_each_table = [len(table["column_names"]) for table in sample["schema"]["schema_items"]]

        column_name_word_indices, table_name_word_indices = [], []
        
        input_words = [sample["text"]]
        for table_id, table_name in enumerate(table_names):
            input_words.append("|")
            input_words.append(table_name)
            table_name_word_indices.append(len(input_words) - 1)
            input_words.append(":")
            
            for column_name in column_names[table_id]:
                input_words.append(column_name)
                column_name_word_indices.append(len(input_words) - 1)
                input_words.append(",")
            
            input_words = input_words[:-1]

        tokenized_inputs = self.tokenizer(
            input_words, 
            return_tensors="pt", 
            is_split_into_words = True,
            padding = "max_length",
            max_length = 512,
            truncation = True
        )

        column_name_token_indices, table_name_token_indices = [], []
        word_indices = tokenized_inputs.word_ids(batch_index = 0)

        for column_name_word_index in column_name_word_indices:
            column_name_token_indices.append([token_id for token_id, word_index in enumerate(word_indices) if column_name_word_index == word_index])

        for table_name_word_index in table_name_word_indices:
            table_name_token_indices.append([token_id for token_id, word_index in enumerate(word_indices) if table_name_word_index == word_index])

        encoder_input_ids = tokenized_inputs["input_ids"]
        encoder_input_attention_mask = tokenized_inputs["attention_mask"]

        if torch.cuda.is_available():
            encoder_input_ids = encoder_input_ids.cuda()
            encoder_input_attention_mask = encoder_input_attention_mask.cuda()

        return encoder_input_ids, encoder_input_attention_mask, \
            column_name_token_indices, table_name_token_indices, column_num_in_each_table
    
    def get_schema(self, tables_and_columns):
        schema_items = []
        table_names = list(dict.fromkeys([t for t, c in tables_and_columns]))
        for table_name in table_names:
            schema_items.append(
                {
                    "table_name": table_name,
                    "column_names":  [c for t, c in tables_and_columns if t == table_name]
                }
            )

        return {"schema_items": schema_items}

    def get_sequence_length(self, text, tables_and_columns, tokenizer):
        table_names = [t for t, c in tables_and_columns]
        table_names = list(dict.fromkeys(table_names))

        column_names = []
        for table_name in table_names:
            column_names.append([c for t, c in tables_and_columns if t == table_name])

        input_words = [text]
        for table_id, table_name in enumerate(table_names):
            input_words.append("|")
            input_words.append(table_name)
            input_words.append(":")
            for column_name in column_names[table_id]:
                input_words.append(column_name)
                input_words.append(",")
            input_words = input_words[:-1]

        tokenized_inputs = tokenizer(input_words, is_split_into_words = True)

        return len(tokenized_inputs["input_ids"])

    def split_sample(self, sample, tokenizer):
        text = sample["text"]

        table_names = []
        column_names = []
        for table in sample["schema"]["schema_items"]:
            table_names.append(table["table_name"] + " ( " + table["table_comment"] + " ) " \
                if table["table_comment"] != "" else table["table_name"])
            column_names.append([column_name + " ( " + column_comment + " ) " \
                if column_comment != "" else column_name \
                    for column_name, column_comment in zip(table["column_names"], table["column_comments"])])

        splitted_samples = []
        recorded_tables_and_columns = []

        for table_idx, table_name in enumerate(table_names):
            for column_name in column_names[table_idx]:
                if self.get_sequence_length(text, recorded_tables_and_columns + [[table_name, column_name]], tokenizer) < 500:
                    recorded_tables_and_columns.append([table_name, column_name])
                else:
                    splitted_samples.append(
                        {
                            "text": text,
                            "schema": self.get_schema(recorded_tables_and_columns)
                        }
                    )
                    recorded_tables_and_columns = [[table_name, column_name]]

        splitted_samples.append(
            {
                "text": text,
                "schema": self.get_schema(recorded_tables_and_columns)
            }
        )

        return splitted_samples

    def merge_pred_results(self, sample, pred_results):
        table_names = []
        column_names = []
        for table in sample["schema"]["schema_items"]:
            table_names.append(table["table_name"] + " ( " + table["table_comment"] + " ) " \
                if table["table_comment"] != "" else table["table_name"])
            column_names.append([column_name + " ( " + column_comment + " ) " \
                if column_comment != "" else column_name \
                    for column_name, column_comment in zip(table["column_names"], table["column_comments"])])

        merged_results = []
        for table_id, table_name in enumerate(table_names):
            table_prob = 0
            column_probs = []
            for result_dict in pred_results:
                if table_name in result_dict:
                    if table_prob < result_dict[table_name]["table_prob"]:
                        table_prob = result_dict[table_name]["table_prob"]
                    column_probs += result_dict[table_name]["column_probs"]

            merged_results.append(
                {
                    "table_name": table_name,
                    "table_prob": table_prob,
                    "column_names": column_names[table_id],
                    "column_probs": column_probs
                }
            )

        return merged_results

    def lista_contains_listb(self, lista, listb):
        for b in listb:
            if b not in lista:
                return 0

        return 1

    def predict_one(self, sample):
        encoder_input_ids, encoder_input_attention_mask, column_name_token_indices,\
            table_name_token_indices, column_num_in_each_table = self.prepare_inputs_and_labels(sample)

        with torch.no_grad():
            model_outputs = self.model(
                encoder_input_ids,
                encoder_input_attention_mask,
                [column_name_token_indices],
                [table_name_token_indices],
                [column_num_in_each_table]
            )

        table_logits = model_outputs["batch_table_name_cls_logits"][0]
        table_pred_probs = torch.nn.functional.softmax(table_logits, dim = 1)[:, 1].cpu().tolist()
            
        column_logits = model_outputs["batch_column_info_cls_logits"][0]
        column_pred_probs = torch.nn.functional.softmax(column_logits, dim = 1)[:, 1].cpu().tolist()

        splitted_column_pred_probs = []
        for table_id, column_num in enumerate(column_num_in_each_table):
            splitted_column_pred_probs.append(column_pred_probs[sum(column_num_in_each_table[:table_id]): sum(column_num_in_each_table[:table_id]) + column_num])
        column_pred_probs = splitted_column_pred_probs

        result_dict = dict()
        for table_idx, table in enumerate(sample["schema"]["schema_items"]):
            result_dict[table["table_name"]] = {
                "table_name": table["table_name"],
                "table_prob": table_pred_probs[table_idx],
                "column_names": table["column_names"],
                "column_probs": column_pred_probs[table_idx],
            }

        return result_dict

    def predict(self, test_sample):
        splitted_samples = self.split_sample(test_sample, self.tokenizer)
        pred_results = []
        for splitted_sample in splitted_samples:
            pred_results.append(self.predict_one(splitted_sample))
        
        return self.merge_pred_results(test_sample, pred_results)
    
    def evaluate_coverage(self, dataset, logger):
        max_k = 100
        total_num_for_table_coverage, total_num_for_column_coverage = 0, 0
        table_coverage_results = [0]*max_k
        column_coverage_results = [0]*max_k

        for data in dataset:
            indices_of_used_tables = [idx for idx, label in enumerate(data["table_labels"]) if label == 1]
            pred_results = self.predict(data)
            # print(pred_results)
            table_probs = [res["table_prob"] for res in pred_results]
            for k in range(max_k):
                indices_of_top_k_tables = np.argsort(-np.array(table_probs), kind="stable")[:k+1].tolist()
                if self.lista_contains_listb(indices_of_top_k_tables, indices_of_used_tables):
                    table_coverage_results[k] += 1
            total_num_for_table_coverage += 1

            for table_idx in range(len(data["table_labels"])):
                indices_of_used_columns = [idx for idx, label in enumerate(data["column_labels"][table_idx]) if label == 1]
                if len(indices_of_used_columns) == 0:
                    continue
                column_probs = pred_results[table_idx]["column_probs"]
                for k in range(max_k):
                    indices_of_top_k_columns = np.argsort(-np.array(column_probs), kind="stable")[:k+1].tolist()
                    if self.lista_contains_listb(indices_of_top_k_columns, indices_of_used_columns):
                        column_coverage_results[k] += 1

                total_num_for_column_coverage += 1

        logger.info(f"total_num_for_table_coverage:{total_num_for_table_coverage}")
        logger.info(f"table_coverage_results:{table_coverage_results}")
        logger.info(f"total_num_for_column_coverage:{total_num_for_column_coverage}")
        logger.info(f"column_coverage_results:{column_coverage_results}")


@OPERATOR_REGISTRY.register()
class SchemaLinking(OperatorABC):
    def __init__(self, table_info_file: str,
            model_path: str,
            selection_mode: str = "eval",                       
            num_top_k_tables: int = 5,                           
            num_top_k_columns: int = 5
        ):
        self.input_table_file = table_info_file
        self.model_path = model_path
        self.selection_mode = selection_mode
        self.num_top_k_tables = num_top_k_tables
        self.num_top_k_columns = num_top_k_columns
        self.logger = get_logger()

    @staticmethod
    def get_desc(lang):
        if lang == "zh":
            return (
                "该算子用于提取出数据库模式链接。\n\n"
                "输入参数：\n"
                "- input_table_file：输入文件路径，数据库表信息\n"
                "- input_sql_key：SQL语句键\n"
                "- input_table_names_original_key：table file中原始表名字段\n"
                "- input_table_names_statement_key：table file中表名说明字段\n"
                "- input_column_names_original_key：table file中原始列名字段\n"
                "- input_column_names_statement_key：table file中列名说明字段\n"
                "- num_top_k_tables：保留的最大表数量\n"
                "- num_top_k_columns：每个表保留的最大列数量\n"
                "- selection_mode：模型链接模式，eval或train\n"
                "- model_path：模式链接模型路径，只在eval模式下需要\n"
                "- input_question_key： question key，只在train模式下需要\n"
                "- input_dbid_key：db_id key，数据库名，只在train模式下需要\n"
                "注意：eval模式需要下载sic_merged模型（quark netdisk or google drive）并在参数中指明模型路径\n\n"
                "输出参数：\n"
                "- output_key：筛选提取的数据库模式信息，保留的表名和列名"
            )
        elif lang == "en":
            return (
                "This operator extracts the database schema linking.\n\n"
                "Input parameters:\n"
                "- input_table_file: Input file path, database table information\n"
                "- input_sql_key: SQL statement key\n"
                "- input_table_names_original_key: Original table name field in the table file\n"
                "- input_table_names_statement_key: Table name description field in the table file\n"
                "- input_column_names_original_key: Original column name field in the table file\n"
                "- input_column_names_statement_key: Column name description field in the table file\n"
                "- num_top_k_tables: Maximum number of tables to retain\n"
                "- num_top_k_columns: Maximum number of columns to retain for each table\n"
                "- selection_mode: Model linking mode, eval or train\n"
                "- model_path: Path to the schema item classifier model, required only in eval mode\n"
                "- input_question_key: Question key, required only in train mode\n"
                "- input_dbid_key: db_id key, database name, required only in train mode\n"
                "Note: In eval mode, you need to download the sic_merged model (from Quark Netdisk or Google Drive) and specify the model path in the parameters.\n\n"
                "Output parameters:\n"
                "- output_key: Extracted database schema information, retaining table names and column names."    
            )
        else:
            return "AnswerExtraction_qwenmatheval performs mathematical answer normalization and standardization."
        

    def _process_data(self, questions_df, schemas_df):
        schemas_dict = {row[self.input_dbid_key]: row for _, row in schemas_df.iterrows()}
        
        processed_data = []
        
        for _, row in questions_df.iterrows():
            db_id = row[self.input_dbid_key]
            schema = schemas_dict.get(db_id, {})
            
            schema_items = []
            
            table_names_orig = schema.get(self.input_table_names_original_key, [])
            table_names = schema.get(self.input_table_names_statement_key, [])
            
            for table_idx, (table_orig, table_name) in enumerate(zip(table_names_orig, table_names)):
                table_comment = table_name if table_name != table_orig else ""
                
                columns = []
                column_comments = []
                
                for col_info_orig, col_info in zip(
                    schema.get(self.input_column_names_original_key, []),
                    schema.get(self.input_column_names_statement_key, [])
                ):
                    if col_info_orig[0] == table_idx:  
                        col_orig = col_info_orig[1]
                        col_name = col_info[1]
                        
                        col_comment = col_name if col_name != col_orig else ""
                        
                        columns.append(col_orig)
                        column_comments.append(col_comment)
                
                schema_items.append({
                    "table_name": table_orig,
                    "table_comment": table_comment,
                    "column_names": columns,
                    "column_comments": column_comments
                })
            
            if self.selection_mode == "train" or self.input_sql_key != "":
                processed_data.append({
                    "text": row[self.input_question_key],
                    "sql": row[self.input_sql_key],
                    "schema": {
                        "schema_items": schema_items
                    }
                })
            else:
                processed_data.append({
                    "text": row[self.input_question_key],
                    "schema": {
                        "schema_items": schema_items
                    }
                })
        
        return processed_data

    def find_used_tables_and_columns(self, dataset):
        for data in dataset:
            sql = data["sql"].lower()
            data["table_labels"] = []
            data["column_labels"] = []
            
            for table_info in data["schema"]["schema_items"]:
                table_name = table_info["table_name"]
                data["table_labels"].append(1 if table_name.lower() in sql else 0)
                data["column_labels"].append([1 if column_name.lower() in sql else 0 \
                    for column_name in table_info["column_names"]])
        return dataset
    
    def filter_func(self, dataset, dataset_type, sic, num_top_k_tables = 5, num_top_k_columns = 5):
        for data in tqdm(dataset, desc = "filtering schema items for the dataset"):
            filtered_schema = dict()
            filtered_schema["schema_items"] = []

            table_names = [table["table_name"] for table in data["schema"]["schema_items"]]
            table_comments = [table["table_comment"] for table in data["schema"]["schema_items"]]
            column_names = [table["column_names"] for table in data["schema"]["schema_items"]]
            column_comments = [table["column_comments"] for table in data["schema"]["schema_items"]]

            if dataset_type == "eval":
                # predict scores for each tables and columns
                pred_results = sic.predict(data)
                # remain top_k1 tables for each database and top_k2 columns for each remained table
                table_probs = [pred_result["table_prob"] for pred_result in pred_results]
                table_indices = np.argsort(-np.array(table_probs), kind="stable")[:num_top_k_tables].tolist()
            elif dataset_type == "train":
                table_indices = [table_idx for table_idx, table_label in enumerate(data["table_labels"]) if table_label == 1]
                if len(table_indices) < num_top_k_tables:
                    unused_table_indices = [table_idx for table_idx, table_label in enumerate(data["table_labels"]) if table_label == 0]
                    table_indices += random.sample(unused_table_indices, min(len(unused_table_indices), num_top_k_tables - len(table_indices)))
                random.shuffle(table_indices)

            for table_idx in table_indices:
                if dataset_type == "eval":
                    column_probs = pred_results[table_idx]["column_probs"]
                    column_indices = np.argsort(-np.array(column_probs), kind="stable")[:num_top_k_columns].tolist()
                elif dataset_type == "train":
                    column_indices = [column_idx for column_idx, column_label in enumerate(data["column_labels"][table_idx]) if column_label == 1]
                    if len(column_indices) < num_top_k_columns:
                        unused_column_indices = [column_idx for column_idx, column_label in enumerate(data["column_labels"][table_idx]) if column_label == 0]
                        column_indices += random.sample(unused_column_indices, min(len(unused_column_indices), num_top_k_columns - len(column_indices)))
                    random.shuffle(column_indices)

                filtered_schema["schema_items"].append(
                    {
                        "table_name": table_names[table_idx],
                        "table_comment": table_comments[table_idx],
                        "column_names": [column_names[table_idx][column_idx] for column_idx in column_indices],
                        "column_comments": [column_comments[table_idx][column_idx] for column_idx in column_indices]
                    }
                )

            data["schema"] = filtered_schema

            if dataset_type == "train":
                del data["table_labels"]
                del data["column_labels"]
            
        return dataset

    def run(self, storage: DataFlowStorage,
            input_sql_key: str = "SQL",
            input_dbid_key: str = "db_id",
            input_question_key: str = "question",
            input_table_names_original_key: str = "table_names_original",
            input_table_names_statement_key: str = "table_names",
            input_column_names_original_key: str = "column_names_original",    
            input_column_names_statement_key: str = "column_names",
            output_schema_key: str = "selected_schema"
        ):
        self.input_question_key = input_question_key
        self.input_dbid_key = input_dbid_key
        self.input_sql_key = input_sql_key
        self.input_table_names_original_key = input_table_names_original_key
        self.input_table_names_statement_key = input_table_names_statement_key
        self.input_column_names_original_key = input_column_names_original_key
        self.input_column_names_statement_key = input_column_names_statement_key
        self.output_schema_key = output_schema_key

        questions_df = storage.read("dataframe")
        self._validate_questions_dataframe(questions_df)
        schemas_df = pd.read_json(self.input_table_file, lines=True)
        self._validate_schemas_dataframe(schemas_df)
        
        processed_data = self._process_data(questions_df, schemas_df)
        
        sic = None
        if self.selection_mode == "eval":
            sic = SchemaItemClassifierInference(self.model_path)
        if self.selection_mode == "train" or self.input_sql_key != "":
            processed_data = self.find_used_tables_and_columns(processed_data)
        if self.selection_mode == "eval" and self.input_sql_key != "":
            self.logger.info("开始评估模式覆盖度...")
            sic.evaluate_coverage(processed_data, self.logger)
            self.logger.info("覆盖度评估完成\n")
        
        filtered_data = self.filter_func(
            processed_data,
            dataset_type=self.selection_mode,
            sic = sic,
            num_top_k_tables=self.num_top_k_tables,
            num_top_k_columns=self.num_top_k_columns
        )
        
        selected_schemas = []
        for data in filtered_data:
            selected_tables = []
            for table in data["schema"]["schema_items"]:
                selected_tables.append({
                    "table_name": table["table_name"],
                    "columns": table["column_names"]
                })
            selected_schemas.append(selected_tables)
        
        questions_df[self.output_schema_key] = selected_schemas

        output_file = storage.write(questions_df)
        self.logger.info(f"Extracted answers saved to {output_file}")

        return [self.output_schema_key]

        
    def _validate_questions_dataframe(self, dataframe: pd.DataFrame):
        if self.input_question_key not in dataframe.columns:
            raise ValueError(f"input_question_key: {self.input_question_key} not found in the dataframe.")
        
        if self.input_dbid_key not in dataframe.columns:
            raise ValueError(f"input_dbid_key: {self.input_dbid_key} not found in the dataframe.")

        if self.input_sql_key not in dataframe.columns:
            raise ValueError(f"selection_mode is {self.selection_mode}, input_sql_key: {self.input_sql_key} not found in the dataframe.")
        
        if self.output_schema_key in dataframe.columns:
            raise ValueError(f"Found {self.output_schema_key} in the dataframe, which would overwrite an existing column. Please use a different output_key.")
        
    def _validate_schemas_dataframe(self, dataframe: pd.DataFrame):
        if self.input_dbid_key not in dataframe.columns:
            raise ValueError(f"input_dbid_key: {self.input_dbid_key} not found in the dataframe.")  

        if self.input_table_names_original_key not in dataframe.columns:
            raise ValueError(f"input_table_names_original_key: {self.input_table_names_original_key} not found in the dataframe.")

        if self.input_table_names_statement_key not in dataframe.columns:
            raise ValueError(f"input_table_names_statement_key: {self.input_table_names_statement_key} not found in the dataframe.")

        if self.input_column_names_original_key not in dataframe.columns:
            raise ValueError(f"input_column_names_original_key: {self.input_column_names_original_key} not found in the dataframe.")

        if self.input_column_names_statement_key not in dataframe.columns:
            raise ValueError(f"input_column_names_statement_key: {self.input_column_names_statement_key} not found in the dataframe.")
        
