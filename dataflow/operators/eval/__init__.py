import sys
from .GeneralText import *
from dataflow.utils.registry import LazyLoader

cur_path = "dataflow/operators/eval/"
_import_structure = {  
    "NgramScorer": (cur_path + "GeneralText/NgramScorer.py", "NgramScorer"),
}

sys.modules[__name__] = LazyLoader(__name__, "dataflow/operators/eval/", _import_structure)
