import argparse
import pandas as pd

from titanic_project.preprocessing.preprocessor import TitanicPreprocessor


def build_parser():
    parser = argparse.ArgumentParser(description="Preprocess Titanic data")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main():
    args = build_parser().parse_args()

    df = pd.read_csv(args.input)

    preprocessor = TitanicPreprocessor()
    df = preprocessor.process(df)

    df.to_csv(args.output, index=False)

    print(f"Saved processed data to {args.output}")


if __name__ == "__main__":
    main()