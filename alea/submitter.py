import os
import logging
import inspect
import shlex
import shutil
from argparse import ArgumentParser
from copy import deepcopy
from json import dumps, loads
from typing import List, Dict, Any, Optional

from tqdm import tqdm

from alea.model import StatisticalModel
from alea.runner import Runner
from alea.utils import (
    get_file_path,
    load_yaml,
    compute_variations,
    add_i_batch,
    can_assign_to_typing,
)


class Submitter:
    """Submitter base class that generate the submission script from the configuration. It is
    initialized by the configuration file, and the configuration file should contain the arguments
    of __init__ method of the Submitter.

    Attributes:
        statistical_model (str): the name of the statistical model
        statistical_model_config (str): the configuration file of the statistical model
        poi (str): the parameter of interest
        computation_dict (dict): the dictionary of the computation,
            with keys to_zip, to_vary and in_common
        debug (bool): whether to run in debug mode.
            If True, only one job will be submitted or one runner will be returned.
            And its script will be printed.
        resubmit (bool): whether to resubmit the jobs that have not finished.
            If True, will submit all the jobs, even if the output file exists.

    Args:
        statistical_model (str): the name of the statistical model
        statistical_model_config (str): the configuration file of the statistical model
        poi (str): the parameter of interest
        computation_options (dict): the configuration of the computation
        computation (str, optional (default='discovery_power')): the name of the computation,
            it should be a key of computation_options
        outputfolder (str, optional (default=None)): the output folder
        debug (bool, optional (default=False)): whether to run in debug mode
        loglevel (str, optional (default='INFO')): the log level

    Keyword Args:
        kwargs: the arguments of __init__ method of the Submitter,
            containing configurations of clusters

    Caution:
        All the source of template should be from the same folder.
        All the output, including toydata and fitting results, should be in the same folder.

    """

    config_file_path: str
    template_path: str
    combine_n_jobs: int = 1
    first_i_batch: int = 0
    allowed_special_args: List[str] = []
    logging = logging.getLogger("submitter_logger")

    def __init__(
        self,
        statistical_model: str,
        statistical_model_config: str,
        poi: str,
        computation_options: dict,
        computation: str = "discovery_power",
        outputfolder: Optional[str] = None,
        fit_strategy: Optional[dict] = None,
        debug: bool = False,
        resubmit: bool = False,
        loglevel: str = "INFO",
        **kwargs,
    ):
        """Initializes the submitter."""
        if type(self) == Submitter:  # noqa: E721
            raise RuntimeError(
                "You cannot instantiate the Submitter class directly, "
                "you must use a subclass where the submit method are implemented"
            )
        loglevel = getattr(logging, loglevel.upper() if not debug else "DEBUG")
        self.logging.setLevel(loglevel)

        # find the path of template, requires users install alea-inference properly
        self.run_toymc = shutil.which("alea-run_toymc")
        if self.run_toymc is None:
            raise RuntimeError(
                "Excecutable alea-run_toymc is not found, "
                "please make sure you have installed alea-inference correctly, "
                "and appended alea/bin or .local/bin(pip install direction) to your $PATH."
            )

        self.statistical_model = statistical_model
        self.statistical_model_config = statistical_model_config
        self.poi = poi
        self.outputfolder = outputfolder
        self.fit_strategy = fit_strategy

        self.computation = computation
        self.computation_dict = computation_options[self.computation]
        self.debug = debug
        self.resubmit = resubmit

        # Find statistical model config file
        if not os.path.exists(self.statistical_model_config):
            self.statistical_model_config = os.path.join(
                os.path.dirname(get_file_path(self.config_file_path)), self.statistical_model_config
            )
        if not (
            os.path.exists(self.statistical_model_config)
            and os.path.isfile(self.statistical_model_config)
        ):
            raise FileNotFoundError(
                f"statistical_model_config {self.statistical_model_config} "
                "is not a valid filename or does not exist, "
                "presumably it should be in the same folder as "
                f"config_file_path {self.config_file_path}."
            )

        # Initialize the statistical model
        statistical_model_class = StatisticalModel.get_model_from_name(self.statistical_model)
        self.model = statistical_model_class.from_config(
            self.statistical_model_config, template_path=self.template_path
        )

        # Get fittable and not fittable parameters, for parameters classification later
        self.parameters_fittable = self.model.parameters.fittable + ["poi_expectation"]
        self.parameters_not_fittable = self.model.parameters.not_fittable

    @classmethod
    def from_config(cls, config_file_path: str, **kwargs) -> "Submitter":
        """Initializes the submitter from a yaml config file.

        Args:
            config_file_path (str): Path to the yaml config file.

        Returns:
            BlueiceExtendedModel: Statistical model.

        """
        config = load_yaml(config_file_path)
        cls.config_file_path = config_file_path
        return cls(**{**config, **kwargs})

    @property
    def outputfolder(self) -> Optional[str]:
        return self._outputfolder

    @outputfolder.setter
    def outputfolder(self, outputfolder: Optional[str]):
        if outputfolder is None:
            # default output folder is the current working directory
            raise ValueError("outputfolder is not provided")
        else:
            self._outputfolder = os.path.abspath(outputfolder)
        if not os.path.exists(self._outputfolder):
            os.makedirs(self._outputfolder, exist_ok=True)

    @staticmethod
    def arg_to_str(value, annotation) -> str:
        """Convert the argument to string for the submission script.

        Args:
            value: the value of the argument, can be various type
            annotation: the annotation of the argument

        Returns:
            str: the string of the argument

        Caution:
            Currently we only support str, int, float, bool, dict and list.
            The float will be rounded to 4 digits after the decimal point.

        """
        if value is None:
            return "None"
            # raise ValueError('provides argument can not be None')
        if can_assign_to_typing(str, annotation):
            return value
        elif can_assign_to_typing(int, annotation):
            return "{:d}".format(value)
        elif can_assign_to_typing(float, annotation):
            # currently we only support 4 digits after the decimal point
            rounded_value = round(value, 4)
            if rounded_value != value:
                logging.warn(
                    f"Value {value} of {annotation} will be rounded to four "
                    f"digit precision for the submission script.",
                )
            return "{:.4f}".format(value)
        elif can_assign_to_typing(bool, annotation):
            return str(value)
        elif can_assign_to_typing(dict, annotation) or can_assign_to_typing(list, annotation):
            # the replacement is needed because the json.dumps adds spaces
            return dumps(value).replace(" ", "")
        else:
            raise ValueError(
                f"Unknown annotation type: {annotation}, "
                "it can only be str, int, float, bool, dict or list, "
                "or the typing relatives of them."
            )

    @staticmethod
    def str_to_arg(value: str, annotation):
        """Convert the string to argument for the submission script.

        Args:
            value: the string of the argument
            annotation: the annotation of the argument

        Returns:
            the value of the argument, can be various type

        """
        if value == "None":
            return None
        if can_assign_to_typing(str, annotation):
            return value
        elif can_assign_to_typing(int, annotation):
            return int(value)
        elif can_assign_to_typing(float, annotation):
            return float(value)
        elif can_assign_to_typing(bool, annotation):
            if value == "True":
                return True
            elif value == "False":
                return False
            else:
                raise ValueError(f"Unknown value type: {value}, it can only be True or False")
        elif can_assign_to_typing(dict, annotation) or can_assign_to_typing(list, annotation):
            # the replacement is needed because the json.dumps adds spaces
            return loads(value)
        else:
            raise ValueError(
                f"Unknown annotation type: {annotation}, "
                "it can only be str, int, float, bool, dict or list, "
                "or the typing relatives of them."
            )

    def merged_arguments_generator(self):
        """Generate the merged arguments for Runner from to_zip, to_vary and in_common."""
        _, default_args, _ = Runner.runner_arguments()

        to_zip = self.computation_dict.get("to_zip", {})
        to_vary = self.computation_dict.get("to_vary", {})
        in_common = self.computation_dict.get("in_common", {})
        allowed_keys = ["to_zip", "to_vary", "in_common"]
        if set(self.computation_dict.keys()) - set(allowed_keys):
            raise ValueError(
                "Keys in computation_options should be to_zip, to_vary or in_common, "
                "unknown computation options: {}".format(
                    set(self.computation_dict.keys()) - set(allowed_keys)
                )
            )

        merged_args_list = compute_variations(to_zip=to_zip, to_vary=to_vary, in_common=in_common)

        common_runner_args = {
            "statistical_model": self.statistical_model,
            "statistical_model_config": self.statistical_model_config,
            "poi": self.poi,
            "fit_strategy": self.fit_strategy,
        }

        if set(merged_args_list[0].keys()) & set(common_runner_args.keys()):
            raise ValueError(
                "You specified the following arguments in computation_options, "
                "but they are already specified in the submitter: "
                f"{set(merged_args_list[0].keys()) & set(common_runner_args.keys())}."
            )

        if self.debug:
            print("\n\n" + f"Will submit {len(merged_args_list)} argument combinations:")
            for merged_args in merged_args_list:
                print(merged_args)
        for merged_args in tqdm(merged_args_list):
            runner_args = deepcopy(default_args)
            # update defaults with merged_args and common_runner_args
            runner_args.update(merged_args)
            runner_args.update(common_runner_args)

            # update n_mc if n_batch is provided
            self.update_n_batch(runner_args)
            # update folder and i_batch
            self.update_output_toydata(runner_args, self.outputfolder)
            # update generate_values and nominal_values for runner
            self.update_runner_args(
                runner_args, self.parameters_fittable, self.parameters_not_fittable
            )
            # update the path of limit_threshold
            self.update_limit_threshold(runner_args, self.outputfolder)
            # update template_path and limit_threshold in statistical_model_args if needed
            self.update_statistical_model_args(runner_args, self.template_path)
            # check if all arguments are supported
            self.check_redunant_arguments(runner_args, self.allowed_special_args)

            yield runner_args

    def filename_kwargs(self, runner_args: dict) -> dict:
        """Get the filename_kwargs from runner_args.

        Args:
            runner_args (dict): the arguments of Runner

        Returns:
            dict: the keyword arguments for the filename

        """
        needed_kwargs = {
            "i_batch": runner_args["i_batch"],
            "confidence_level": runner_args["confidence_level"],
            **runner_args["nominal_values"],
            **runner_args["generate_values"],
        }
        return needed_kwargs

    def computation_tickets_generator(self):
        """Get the submission script for the current configuration. It generates the submission
        script for each combination of the computation options.

        For Runner from to_zip, to_vary and in_common:
            - First, generate the combined computational options directly.
            - Second, update the input and output folder of the options.
            - Thrid, collect the non-fittable(settable) parameters into nominal_values.
            - Then, collect the fittable parameters into generate_values.
            - Finally, it generates the submission script for each combination.

        Yields:
            (str, str): the submission script and name output_filename

        """

        _, _, annotations = Runner.runner_arguments()

        for runner_args in self.merged_arguments_generator():
            for i_batch in range(self.first_i_batch, runner_args.get("n_batch", 1)):
                i_args = deepcopy(runner_args)
                i_args["i_batch"] = i_batch

                for name in ["output_filename", "toydata_filename", "limit_threshold"]:
                    if i_args.get(name, None) is not None:
                        # Note: here the later format will overwrite the previous one,
                        # so generate_values have the highest priority.
                        needed_kwargs = self.filename_kwargs(i_args)
                        try:
                            i_args[name] = i_args[name].format(**needed_kwargs)
                        except KeyError:
                            raise KeyError(
                                f"Keys for {i_args[name]} are not in provided arguments "
                                f"{needed_kwargs}, please check the {name}."
                            )

                script = Submitter.script_from_runner_kwargs(annotations, i_args)
                script = f"python3 {self.run_toymc} " + " ".join(
                    map(shlex.quote, script.split(" "))
                )

                if not self.already_done(i_args):
                    yield script, i_args["output_filename"]

    def already_done(self, i_args: dict) -> bool:
        """Check if the job is already done, considering the modes of toydata and output."""
        toydata_mode = i_args["toydata_mode"]
        toydata_filename = i_args["toydata_filename"]
        only_toydata = i_args["only_toydata"]
        output_filename = i_args["output_filename"]
        # these check might need change if we support more modes
        if (toydata_mode == "generate_and_store") and (toydata_filename is None):
            raise ValueError(
                "toydata_filename should be provided when toydata_mode is generate_and_store."
            )
        if (not only_toydata) and (output_filename is None):
            raise ValueError("output_filename should be provided when only_toydata is False.")

        is_done = True
        if self.resubmit or (self.computation == "threshold"):
            is_done = False
        if (toydata_mode == "generate_and_store") and (not os.path.exists(toydata_filename)):
            is_done = False
        if (not only_toydata) and (not os.path.exists(output_filename)):
            is_done = False
        return is_done

    def combined_tickets_generator(self):
        """Get the combined submission script for the current configuration. ``self.combine_n_jobs``
        jobs will be combined into one submission script.

        Yields:
            (str, str): the combined submission script and name output_filename

        Note:
            User can add ``combine_n_jobs: 10`` in ``local_configurations``,
            ``slurm_configurations`` or ``htcondor_configurations`` to combine 10 jobs into
            one submission script. User will need this feature when the number of jobs pending
            for submission is too large.

        """

        _script = ""
        n_combined = 0
        n_submitted = 0
        for script, last_output_filename in self.computation_tickets_generator():
            if n_combined == 0:
                _script += script
            else:
                _script += " && " + script
            n_combined += 1
            if n_combined == self.combine_n_jobs:
                yield _script, last_output_filename
                n_submitted += 1
                n_combined = 0
                _script = ""
        else:
            if n_combined > 0:
                yield _script, last_output_filename
                n_submitted += 1
        print(f"Total {n_submitted} jobs submitted.")

    @staticmethod
    def update_n_batch(runner_args):
        """Update n_mc if n_batch is provided.

        Distribute n_mc into n_batch, so that each batch will run n_mc/n_batch times.

        """
        if "n_mc" not in runner_args:
            logging.warn("n_mc is not provided, it will be set to the default value of Runner")
            return
        if "n_batch" in runner_args:
            if runner_args["n_mc"] % runner_args["n_batch"] != 0:
                raise ValueError("n_mc must be divisible by n_batch")
            runner_args["n_mc"] = runner_args["n_mc"] // runner_args["n_batch"]

    @staticmethod
    def update_output_toydata(runner_args, outputfolder: str):
        for f in ["output_filename", "toydata_filename"]:
            if (f in runner_args) and (runner_args[f] is not None):
                if ("n_batch" in runner_args) and (runner_args["n_batch"] != 1):
                    runner_args[f] = os.path.join(outputfolder, add_i_batch(runner_args[f]))
                else:
                    runner_args[f] = os.path.join(outputfolder, runner_args[f])

    @staticmethod
    def update_runner_args(
        runner_args: Dict[str, Dict[str, Any]],
        parameters_fittable: List[str],
        parameters_not_fittable: List[str],
    ):
        """Update the runner arguments' generate_values and nominal_values. If the argument is
        fittable, it will be added to generate_values, otherwise it will be added to nominal_values.

        Args:
            runner_args (dict): the arguments of Runner

        """
        if runner_args["generate_values"] is None:
            runner_args["generate_values"] = {}
        if runner_args["nominal_values"] is None:
            runner_args["nominal_values"] = {}
        kw_to_pop = []
        for k, v in runner_args.items():
            if k in parameters_fittable:
                runner_args["generate_values"][k] = v
                kw_to_pop.append(k)
            elif k in parameters_not_fittable:
                runner_args["nominal_values"][k] = v
                kw_to_pop.append(k)
        for k in kw_to_pop:
            runner_args.pop(k)
        if set(runner_args["generate_values"].keys()) - set(parameters_fittable):
            raise ValueError(
                f'The generate_values {runner_args["generate_values"]} '
                f"should be a subset of the fittable parameters "
                f"{parameters_fittable} in the statistical model."
            )
        if not all([isinstance(v, (float, int)) for v in runner_args["generate_values"].values()]):
            raise ValueError(
                f"The generate_values {runner_args['generate_values']} "
                "should be all float or int."
            )
        if not all([isinstance(v, (float, int)) for v in runner_args["nominal_values"].values()]):
            raise ValueError(
                f"The nominal_values {runner_args['nominal_values']} should be all float or int."
            )

    @staticmethod
    def update_limit_threshold(runner_args, outputfolder: str):
        if "limit_threshold" in runner_args:
            runner_args["limit_threshold"] = os.path.join(
                outputfolder, runner_args["limit_threshold"]
            )

    @staticmethod
    def update_statistical_model_args(
        runner_args: Dict[str, Dict[str, Any]], template_path: Optional[str] = None
    ):
        """Update template_path in the statistical model arguments.

        Args:
            runner_args (dict): the arguments of Runner

        """
        if runner_args["statistical_model_args"] is None:
            runner_args["statistical_model_args"] = {}
        if template_path is not None:
            runner_args["statistical_model_args"]["template_path"] = template_path
        if "limit_threshold" in runner_args:
            runner_args["statistical_model_args"]["limit_threshold"] = runner_args.pop(
                "limit_threshold"
            )
        if "limit_threshold_interpolation" in runner_args:
            runner_args["statistical_model_args"]["limit_threshold_interpolation"] = (
                runner_args.pop("limit_threshold_interpolation")
            )
        if "asymptotic_dof" in runner_args:
            runner_args["statistical_model_args"]["asymptotic_dof"] = runner_args.pop(
                "asymptotic_dof"
            )

    @staticmethod
    def check_redunant_arguments(runner_args, allowed_special_args: List[str] = []):
        signatures = inspect.signature(Runner.__init__)
        args = list(signatures.parameters.keys())[1:] + ["n_batch"] + allowed_special_args
        intended_args = set(runner_args.keys())
        allowed_args = set(args)
        if not intended_args.issubset(allowed_args):
            raise ValueError(
                f"Not all arguments are supported, "
                f"arguments {allowed_args} are acceptable, "
                f"and the following arguments is unknown: "
                f"{intended_args - allowed_args}."
            )

    def submit(self, *arg, **kwargs):
        """Submit the jobs to the destinations."""
        raise NotImplementedError("You must write a submit function your submitter class")

    def all_runner_kwargs(self):
        """Parse all the runner arguments from the submission script."""
        kwargs_list = []
        for _, (script, _) in enumerate(self.computation_tickets_generator()):
            kwargs = Submitter.runner_kwargs_from_script(shlex.split(script)[2:])
            kwargs_list.append(kwargs)
        return kwargs_list

    @staticmethod
    def runner_kwargs_from_script(sys_argv: Optional[List[str]] = None):
        """Parse kwargs of a Runner from a string of arguments(script).

        Args:
            sys_argv (list, optional (default=None)): string of arguments, with the format of
                ['--arg1', 'value1', '--arg2', 'value2', ...]. The arguments must be the same as
                the arguments of Runner.__init__.

        """
        signatures = inspect.signature(Runner.__init__)
        args = list(signatures.parameters.keys())[1:]
        parser = ArgumentParser(description="Command line running of alea-run_toymc")

        # skip the first one because it is self(Runner itself)
        for arg in args:
            parser.add_argument(f"--{arg}", type=str, required=True, help=None)

        parsed_args = parser.parse_args(args=sys_argv)
        kwargs = {}
        for arg, value in parsed_args.__dict__.items():
            kwargs.update({arg: Submitter.str_to_arg(value, signatures.parameters[arg].annotation)})
        return kwargs

    @staticmethod
    def script_from_runner_kwargs(annotations, kwargs) -> str:
        """Generate the submission script from the runner arguments."""
        script_array = []
        for arg, annotation in annotations.items():
            script_array.append(f"--{arg}")
            script_array.append(Submitter.arg_to_str(kwargs[arg], annotation))
        script = " ".join(script_array)
        return script
