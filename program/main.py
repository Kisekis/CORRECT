import argparse
from common.logger import logger
from common.dataloader import DataLoader
from analyze.get_analyzer import get_analyzer
from mode import (
    mode_no_context,
    mode_context,
    mode_normal,
    mode_hard,
)
from program.mode import mode_par_scaling, mode_seq_scaling


def process_no_context(
    loader: DataLoader,
    detection_model: str,
    timestamp: str = None,
    res_timestamp: str = None,
):
    analyzer = get_analyzer(detection_model)
    mode_no_context.process_batch(
        loader,
        analyzer,
        res_folder="../results/no_context",
        times=1,
        timestamp=timestamp,
        res_timestamp=res_timestamp,
    )


def process_context(
    loader: DataLoader,
    detection_model: str,
    timestamp: str = None,
    res_timestamp: str = None,
):
    analyzer = get_analyzer(detection_model)
    mode_context.process_batch(
        loader,
        analyzer,
        res_folder="../results/context",
        times=8,
        timestamp=timestamp,
        res_timestamp=res_timestamp,
    )


def process_normal(loader: DataLoader, detection_model: str, evaluator_model: str):
    detector = get_analyzer(detection_model)
    evaluator = get_analyzer(evaluator_model)
    mode_normal.process_batch(
        loader, detector, evaluator, res_folder="../results/normal", times=1
    )


def process_hard(
    loader: DataLoader,
    detection_model: str,
    evaluator_model: str,
    timestamp: str = None,
    res_timestamp: str = None,
    re_evaluate: bool = False,
    existing_timestamp: str = None,
):
    detector = get_analyzer(detection_model)
    evaluator = get_analyzer(evaluator_model)
    mode_hard.process_batch(
        loader,
        detector,
        evaluator,
        res_folder="../results/hard",
        times=1,
        timestamp=timestamp,
        res_timestamp=res_timestamp,
        re_evaluate=re_evaluate,
        existing_timestamp=existing_timestamp,
    )


def process_seq_scaling(
    loader: DataLoader,
    detection_model: str,
    timestamp: str = None,
    res_timestamp: str = None,
):
    analyzer = get_analyzer(detection_model)
    mode_seq_scaling.process_batch(
        loader,
        analyzer,
        res_folder="../results/seq_scaling",
        times=1,
        timestamp=timestamp,
        res_timestamp=res_timestamp,
    )


def process_par_scaling(
    loader: DataLoader,
    evaluator_model: str,
    timestamp: str = None,
    res_timestamp: str = None,
):
    evaluator = get_analyzer(evaluator_model)
    mode_par_scaling.process_batch(
        loader,
        evaluator,
        res_folder="../results/par_scaling",
        timestamp=timestamp,
        res_timestamp=res_timestamp,
    )


def main():
    parser = argparse.ArgumentParser(description="data processing")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=[
            "no_context",
            "context",
            "normal",
            "hard",
            "seq_scaling",
            "par_scaling",
        ],
        help="mode",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        default="CIVDataset-Small",
        help="dataset name",
    )
    parser.add_argument(
        "--res_timestamp",
        type=str,
        required=False,
        default=None,
        help="result timestamp",
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        required=False,
        default=None,
        help="dataset timestamp",
    )
    parser.add_argument(
        "--detection_model",
        type=str,
        required=False,
        default="r1-7b",
        help="detection model",
    )
    parser.add_argument(
        "--existing_timestamp",
        type=str,
        required=False,
        default=None,
        help="existing result timestamp",
    )
    parser.add_argument(
        "--majority_file",
        type=str,
        required=False,
        default=None,
        help="majority file",
    )

    args = parser.parse_args()

    dataset_name = args.dataset

    loader = DataLoader()
    loader.load_data(dataset_name)
    loader.print_data_info()

    if args.mode == "no_context":
        process_no_context(
            loader,
            args.detection_model,
            timestamp=args.timestamp,
            res_timestamp=args.res_timestamp,
        )
    elif args.mode == "context":
        process_context(
            loader,
            args.detection_model,
            timestamp=args.timestamp,
            res_timestamp=args.res_timestamp,
        )
    elif args.mode == "normal":
        process_normal(
            loader, detection_model=args.detection_model, evaluator_model="deepseek"
        )
    elif args.mode == "hard":
        process_hard(
            loader,
            detection_model=args.detection_model,
            evaluator_model="4o",
            timestamp=args.timestamp,
            res_timestamp=args.res_timestamp,
            re_evaluate=False,
            existing_timestamp=args.existing_timestamp,
        )
    elif args.mode == "seq_scaling":
        process_seq_scaling(
            loader,
            detection_model=args.detection_model,
            timestamp=args.timestamp,
            res_timestamp=args.res_timestamp,
        )
    elif args.mode == "par_scaling":
        process_par_scaling(
            loader,
            evaluator_model="4o",
            timestamp=args.timestamp,
            res_timestamp=args.res_timestamp,
        )


if __name__ == "__main__":
    main()
