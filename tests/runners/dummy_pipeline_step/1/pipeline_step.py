"""Dummy pipeline step matching the default from clarifai pipeline init."""

import argparse

from clarifai.utils.logging import logger


def main():
    parser = argparse.ArgumentParser(description='dummy-pipeline-step processing step.')
    parser.add_argument('--input_text', type=str, required=True, help='Text input for processing')

    args = parser.parse_args()

    # TODO: Implement your pipeline step logic here
    logger.info(f"dummy-pipeline-step processed: {args.input_text}")


if __name__ == "__main__":
    main()
