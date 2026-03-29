from lmms_eval.api.filter import Filter
from lmms_eval.tasks._task_utils.eval_utils import extract_final_boxed_content


class BoxedFilter(Filter):
    def apply(self, resps, docs):
        def filter_set(inst):
            return [extract_final_boxed_content(resp) for resp in inst]

        return [filter_set(resp) for resp in resps]
