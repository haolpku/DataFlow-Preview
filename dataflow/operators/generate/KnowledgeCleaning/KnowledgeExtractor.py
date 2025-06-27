import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC

import os
from pathlib import Path
from mineru.data.data_reader_writer import FileBasedDataWriter
from mineru.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
from mineru.backend.pipeline.pipeline_middle_json_mkcontent import union_make as pipeline_union_make
from mineru.backend.pipeline.model_json_to_middle_json import result_to_middle_json as pipeline_result_to_middle_json
from mineru.utils.enum_class import MakeMode
# from magic_doc.docconv import DocConverter
import chonkie

@OPERATOR_REGISTRY.register()
class KnowledgeExtractor(OperatorABC):
    '''
    Answer Generator is a class that generates answers for given questions.
    '''
    def __init__(self, **kwargs):
        self.logger = get_logger()
        self.intermediate_dir=kwargs.get("intermediate_dir", "intermediate")
        
    @staticmethod
    def get_desc(lang="en"):
        """
        返回算子功能描述 (根据run()函数的功能实现)
        """
        if lang == "zh":
            return (
                "知识提取器，支持从PDF/DOC/PPT等文档中提取结构化内容，"
                "输出Markdown格式的文本数据。"
                "处理流程包括：\n"
                "1. PDF文件：使用MinerU解析引擎提取文本、表格和公式\n"
                "2. DOC/PPT文件：通过DocConverter转换为Markdown\n"
                "3. HTML文件：提取正文内容（待实现）"
            )
        else:  # 默认英文
            return (
                "Knowledge Extractor that supports content extraction "
                "from PDF/DOC/PPT files to structured Markdown.\n"
                "Processing pipeline:\n"
                "1. PDF: Uses MinerU engine to extract text/tables/formulas\n"
                "2. DOC/PPT: Converts to Markdown via DocConverter\n"
                "3. HTML: Content extraction (TODO)"
            )
        
    def _parse_pdf_to_md(
        self,
        input_pdf_path: str, 
        output_dir: str,      
        lang: str = "ch",     
        parse_method: str = "auto"  # 解析方法：auto/txt/ocr
    ):
        """
        将PDF转换为Markdown（仅使用Pipeline后端）
        """
        # 读取PDF文件
        pdf_bytes = Path(input_pdf_path).read_bytes()
        pdf_name = Path(input_pdf_path).stem

        # 解析PDF
        infer_results, all_image_lists, all_pdf_docs, _, ocr_enabled_list = pipeline_doc_analyze(
            [pdf_bytes], [lang], parse_method=parse_method
        )

        # 准备输出目录
        image_dir = os.path.join(output_dir, f"{pdf_name}_images")
        os.makedirs(image_dir, exist_ok=True)
        image_writer = FileBasedDataWriter(image_dir)
        md_writer = FileBasedDataWriter(output_dir)

        # 生成中间结果和Markdown
        middle_json = pipeline_result_to_middle_json(
            infer_results[0], all_image_lists[0], all_pdf_docs[0], 
            image_writer, lang, ocr_enabled_list[0], True
        )
        md_content = pipeline_union_make(middle_json["pdf_info"], MakeMode.MM_MD, os.path.basename(image_dir))
        # 保存Markdown
        md_writer.write_string(f"{pdf_name}.md", md_content)
        print(f"Markdown saved to: {os.path.join(output_dir, f'{pdf_name}.md')}")
        return os.path.join(output_dir,f"{pdf_name}.md")

    def _parse_doc_to_md(self, input_file: str, output_file: str):
        """
           support conversion of doc/ppt/pptx/pdf files to markdowns
        """
        converter = DocConverter(s3_config=None)
        markdown_content, time_cost = converter.convert(input_file, conv_timeout=300)
        print("time cost: ", time_cost)
        with open(output_file, "w") as f:
            f.write(markdown_content)
        return output_file

    def run(self, storage:DataFlowStorage ,raw_file, lang="ch"):
        self.logger.info("starting to extract...")
        raw_file_name=os.path.splitext(os.path.basename(raw_file))[0]
        raw_file_suffix=os.path.splitext(raw_file)[1]
        if(raw_file_suffix==".pdf"):
            # optional: 是否从本地加载OCR模型
            os.environ['MINERU_MODEL_SOURCE'] = "local"
            output_file=self._parse_pdf_to_md(
                raw_file,
                self.intermediate_dir,
                lang,
                "txt"
            )
        elif(raw_file_suffix in [".doc", ".docx", ".pptx", ".ppt"]):
            output_file=os.path.join(self.intermediate_dir,f"{raw_file_name}.md")
            output_file=self._parse_doc_to_md(raw_file, output_file)
        elif(raw_file_suffix == ".html"):
            # TODO: Implement the logic to extract knowledge from HTML files
            # ...
            pass
        else:
            raise Exception("Unsupported file type: " + raw_file_suffix)
        
        self.logger.info(f"Primary extracted result written to: {output_file}")
        return output_file
