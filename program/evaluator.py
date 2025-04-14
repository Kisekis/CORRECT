import argparse
from evaluate.base_evaluator import Evaluator
from evaluate.f1_evaluator import F1Evaluator
from evaluate.cwe_scaling_evaluator import CWEScalingEvaluator
from evaluate.abnormal_evaluator import AbnormalEvaluator
from evaluate.feedback_evaluator import FeedbackEvaluator
from evaluate.scaling_evaluator import ScalingEvaluator


def get_evaluator(evaluator_name):
    evaluator_map = {
        "base": Evaluator,
        "abnormal": AbnormalEvaluator,
        "f1": F1Evaluator,
        "cwe_scaling": CWEScalingEvaluator,
        "feedback": FeedbackEvaluator,
        "scaling": ScalingEvaluator,
    }
    return evaluator_map.get(evaluator_name, AbnormalEvaluator)()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluator")
    parser.add_argument(
        "--evaluator",
        type=str,
        default="abnormal",
        help="Evaluator type to use (venn/unknown/f1/cwe/rationale/sankey/f1432/change/cwe_scaling/dataset/feedback)",
    )
    args = parser.parse_args()

    evaluator = get_evaluator(args.evaluator)
    evaluator.evaluate()
