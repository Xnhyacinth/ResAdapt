from lmms_eval.tasks._task_utils.eval_utils import extract_final_boxed_content

from lmms_eval.tasks.mvbench import utils as _utils


def mvbench_process_results(doc, results):
    return _utils.mvbench_process_results(doc, [extract_final_boxed_content(results[0])])

