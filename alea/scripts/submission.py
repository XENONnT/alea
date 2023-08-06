from argparse import ArgumentParser

from alea.submitters import SubmitterMidway, SubmitterOSG, SubmitterLocal


def main():
    parser = ArgumentParser('Submitter script for toyMCs in alea')
    # Exclusive groups on whether submitting or debugging
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--submit',
        action='store_true',
        help='Submit to OSG/Midway')
    group.add_argument(
        '--debug',
        action='store_true',
        help='Only 1 job will be prepared and its script will be printed out')
    # Exclusive groups on submission destination
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        '--local',
        action='store_true',
        help='Executes the defined jobs locally')
    group.add_argument(
        '--midway',
        action='store_true',
        help='Prepare submission for Midway')
    group.add_argument(
        '--osg',
        action='store_true',
        help='Write out files for submission to OSG')
    # Required arguments
    parser.add_argument(
        '--runner_config',
        type=str,
        required=True,
        help='YAML runner config file')
    parser.add_argument(
        '--outputfolder',
        type=str,
        required=False,
        default=None,
        help='Overwriting the outputfolder, usually defined in runner config file')
    parser.add_argument(
        '--computation',
        type=str,
        required=True,
        choices=[
            'discovery_power',
            'threshold',
            'sensitivity',
        ],
        help='Type of computation, defined in YAML runner config file')
    parsed_args = parser.parse_args()
    # parsed_args.submit, parsed_args.debug

    kwargs = dict()
    if parsed_args.outputfolder is not None:
        kwargs.update({'outputfolder': parsed_args.outputfolder})

    if sum([parsed_args.submit, parsed_args.debug]) > 1:
        raise ValueError(
            'You can only choose one of the following: --submit, --debug')

    if sum([parsed_args.midway, parsed_args.osg, parsed_args.local]) > 1:
        raise ValueError(
            'You can only choose one of the following: --midway, --osg, --local')

    if parsed_args.midway:
        submitter = SubmitterMidway(
            runner_config=parsed_args.runner_config,
            computation=parsed_args.computation,
            debug=parsed_args.debug, **kwargs)
    elif parsed_args.osg:
        submitter = SubmitterOSG(
            runner_config=parsed_args.runner_config,
            computation=parsed_args.computation,
            debug=parsed_args.debug, **kwargs)
    elif parsed_args.local:
        submitter = SubmitterLocal(
            runner_config=parsed_args.runner_config,
            computation=parsed_args.computation,
            debug=parsed_args.debug, **kwargs)

    submitter.submit()

if __name__ == '__main__':
    main()
