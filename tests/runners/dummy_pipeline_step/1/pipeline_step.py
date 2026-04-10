"""Dummy pipeline step matching the default clarifai pipelinestep init template."""

import argparse


def main():
    parser = argparse.ArgumentParser(description='Dummy pipeline step.')
    parser.add_argument('--param_a', type=str, required=True, help='First parameter')
    parser.add_argument('--param_b', type=str, default='default_b', help='Second parameter')

    args = parser.parse_args()

    print(f"Pipeline step started: param_a={args.param_a}, param_b={args.param_b}")
    print("Pipeline step completed successfully")


if __name__ == "__main__":
    main()
