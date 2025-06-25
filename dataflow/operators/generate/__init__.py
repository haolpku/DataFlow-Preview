import sys
from dataflow.utils.registry import LazyLoader
from .Reasoning import *

cur_path = "dataflow/operators/generate/"
_import_structure = {
    "AnswerGenerator": (cur_path + "Reasoning/AnswerGenerator.py", "AnswerGenerator"),
    "QuestionCategoryClassifier": (cur_path + "Reasoning/QuestionCategoryClassifier.py", "QuestionCategoryClassifier"),
    "QuestionDifficultyClassifier": (cur_path + "Reasoning/QuestionDifficultyClassifier.py", "QuestionDifficultyClassifier"),
    "QuestionGenerator": (cur_path + "Reasoning/QuestionGenerator.py", "QuestionGenerator"),
    "AnswerExtraction_QwenMathEval": (cur_path + "Reasoning/AnswerExtraction_QwenMathEval.py", "AnswerExtraction_QwenMathEval"),
    "PseudoAnswerGenerator": (cur_path + "Reasoning/PseudoAnswerGenerator.py", "PseudoAnswerGenerator"),
    "DatabaseSchemaExtractor": (cur_path + "Text2SQL/DatabaseSchemaExtractor.py", "DatabaseSchemaExtractor"),
    "ExtraKnowledgeGenerator": (cur_path + "Text2SQL/ExtraKnowledgeGenerator.py", "ExtraKnowledgeGenerator"),
    "PromptGenerator": (cur_path + "Text2SQL/PromptGenerator.py", "PromptGenerator"),
    "QuestionRefiner": (cur_path + "Text2SQL/QuestionRefiner.py", "QuestionRefiner"),
    "SchemaLinking": (cur_path + "Text2SQL/SchemaLinking.py", "SchemaLinking"),
    "SQLDifficultyClassifier": (cur_path + "Text2SQL/SQLDifficultyClassifier.py", "SQLDifficultyClassifier"),
    "SQLFilter": (cur_path + "Text2SQL/SQLFilter.py", "SQLFilter"),
    "Text2SQLDifficultyClassifier": (cur_path + "Text2SQL/Text2SQLDifficultyClassifier.py", "Text2SQLDifficultyClassifier"),

}

sys.modules[__name__] = LazyLoader(__name__, "dataflow/operators/generate/", _import_structure)
