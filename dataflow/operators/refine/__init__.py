import sys
from dataflow.utils.registry import LazyLoader
from .GeneralText import *
cur_path = "dataflow/operators/refine/"
_import_structure = {
    "HtmlUrlRemoverRefiner": (cur_path + "GeneralText" + "HtmlUrlRemoverRefiner.py", "HtmlUrlRemoverRefiner")
}

sys.modules[__name__] = LazyLoader(__name__, "dataflow/operators/refine/", _import_structure)
