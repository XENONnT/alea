from argparse import ArgumentParser


def main():
    parser = ArgumentParser("Submitter script for toyMCs in alea")
    # Required arguments
    parser.add_argument("running_config", type=str, help="YAML runner config file")
    parser.add_argument(
        "--computation",
        type=str,
        required=True,
        help=(
            "Type of computation, defined in YAML runner config file\n"
            "May be any key in computation_options.\n"
            'If the key is "threshold" no jobs are submitted,\n'
            "and the runner instead tries to collate already existing files\n"
            "to compute a Neyman threshold.\n"
            "Otherwise it submits toyMCs according to the config file.\n"
            "Other common names are (you have to set the config accordingly):\n"
            '    * "discovery_power": you wish to compute discovery significance '
            "for different signal sizes\n"
            '    * "sensitivity": you wish to compute the distribution of confidence'
            "intervals under the null hypothesis:\n"
            '    * "coverage": you wish to compute upper limits for a range of signal sizes\n'
            '    * "neyman": you wish to compute the test statistic for testing'
            "a range of true signal sizes\n"
            '    * "pull": you wish to check the distribution of fitted nuisance parameters'
            "as function of the true value\n"
            '    * "trialcorrection": you wish to compute discovery significance'
            "for the same dataset with all signal hypotheses\n"
        ),
    )
    # Exclusive groups on submission destination
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--local", action="store_true", help="Executes the defined jobs locally")
    group.add_argument("--slurm", action="store_true", help="Prepare submission for slurm")
    group.add_argument(
        "--htcondor", action="store_true", help="Write out files for submission to htcondor"
    )
    # Optional arguments
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Only 1 job will be prepared and its script will be printed out",
    )
    parser.add_argument(
        "--outputfolder",
        type=str,
        required=False,
        default=None,
        help="Overwriting the outputfolder, usually defined in runner config file",
    )
    parser.add_argument(
        "--resubmit",
        action="store_true",
        help="Force resubmit no matter if the output folder exists",
    )
    parsed_args = parser.parse_args()

    if parsed_args.computation == "threshold":
        if not parsed_args.local:
            raise ValueError("Threshold computation can only be done locally.")

    if parsed_args.local:
        from alea.submitters.local import SubmitterLocal, NeymanConstructor

        submitter_class = (
            SubmitterLocal if parsed_args.computation != "threshold" else NeymanConstructor
        )
    elif parsed_args.slurm:
        from alea.submitters.slurm import SubmitterSlurm

        submitter_class = SubmitterSlurm
    elif parsed_args.htcondor:
        from alea.submitters.htcondor import SubmitterHTCondor

        submitter_class = SubmitterHTCondor
    else:
        raise ValueError(
            "No submission destination specified. "
            "Please specify one of the following: --local, --slurm, --htcondor"
        )

    kwargs = {
        "computation": parsed_args.computation,
        "debug": parsed_args.debug,
        "resubmit": parsed_args.resubmit,
    }
    # if outputfolder is provided, overwrite the one in runner config file
    if parsed_args.outputfolder:
        kwargs["outputfolder"] = parsed_args.outputfolder

    submitter = submitter_class.from_config(parsed_args.running_config, **kwargs)

    submitter.submit()


if __name__ == "__main__":
    main()
