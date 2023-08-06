import inspect
from json import loads
from argparse import ArgumentParser

from alea.runner import Runner
from alea.configuration import Configuration


def main():
    signatures = inspect.signature(Runner.__init__)
    args = list(signatures.parameters.keys())
    parser = ArgumentParser(description='command line running of run_toymcs')

    # skip the first one because it is self(Runner itself)
    for arg in args[1:]:
        parser.add_argument(f'--{arg}',
            type=str,
            required=True,
            # TODO: add the help message
            help=None)

    parsed_args = parser.parse_args()
    args = dict()
    for arg, value in parsed_args.__dict__.items():
        args.update({arg: Configuration.str_to_arg(value, signatures[arg].annotation)})

    runner = Runner(**args)

    runner.run()


if __name__ == '__main__':
    main()
