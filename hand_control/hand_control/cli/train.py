#!/usr/bin/env python

import argparse

from hand_control import ClassificationModel
from hand_control.models import __default_model__


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.description = (
        "Train a custom hand pose classification model using a labeled dataset. "
        "The trained model can be used by the main hand control application."
    )

    parser.add_argument(
        "path_to_model",
        type=str,
        help="Path where the trained model will be saved",
    )
    parser.add_argument(
        "path_to_data",
        type=str,
        help="Path to the dataset directory",
    )
    parser.add_argument(
        "-H",
        "--hidden_layers",
        nargs="+",
        type=int,
        default=[50, 25, 10],
        help="Hidden layer dimensions (e.g. -H 50 25 10)",
    )
    parser.add_argument(
        "-l",
        "--learning_rate",
        type=float,
        default=0.01,
        help="Learning rate",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=15,
        help="Number of training epochs",
    )
    parser.add_argument(
        "-t",
        "--test_size",
        type=float,
        default=0.3,
        help="Fraction of data used for validation",
    )

    args = parser.parse_args()

    model = ClassificationModel()
    model.read_dataset(args.path_to_data)
    model.preprocess()
    model.train(
        hidden_layers=args.hidden_layers,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        test_size=args.test_size,
    )
    model.save(args.path_to_model)


if __name__ == "__main__":
    main()
