import argparse
import pandas as pd

from titanic_project.features.features_computer import TitanicFeaturesComputer


def build_parser():
    parser = argparse.ArgumentParser(description="Create Titanic features")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main():
    args = build_parser().parse_args()

    df = pd.read_csv(args.input)

    features_computer = TitanicFeaturesComputer()
    df = features_computer.process(df)

    df.to_csv(args.output, index=False)

    print(f"Saved features to {args.output}")


if __name__ == "__main__":
    main()