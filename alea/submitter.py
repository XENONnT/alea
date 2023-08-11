import os
import logging
import inspect
import shlex
from copy import deepcopy
from json import dumps, loads

from tqdm import tqdm

from alea.model import StatisticalModel
from alea.runner import Runner
from alea.utils import load_yaml, compute_variations, add_i_batch


class Submitter():
    """
    Submitter base class that generate the submission script from the configuration.
    It is initialized by the configuration file, and the configuration file should contain
    the arguments of __init__ method of the Submitter.

    Attributes:
        statistical_model (str): the name of the statistical model
        statistical_model_config (str): the configuration file of the statistical model
        poi (str): the parameter of interest
        computation_options (dict): the configuration of the computation
        computation (dict): the dictionary of the computation,
            with keys to_zip, to_vary and in_common
        debug (bool): whether to run in debug mode.
            If True, only one job will be submitted, and its script will be printed.

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

    logging = logging.getLogger('submitter_logger')

    def __init__(
            self,
            statistical_model: str,
            statistical_model_config: str,
            poi: str,
            computation_options: dict,
            computation: str = 'discovery_power',
            outputfolder: str = None,
            debug: bool = False,
            loglevel: str = 'INFO',
            **kwargs,
        ):
        """Initializes the submitter."""
        if type(self) == Submitter:
            raise RuntimeError(
                "You cannot instantiate the Submitter class directly, "
                "you must use a subclass where the submit method are implemented")
        loglevel = getattr(logging, loglevel.upper())
        self.logging.setLevel(loglevel)

        self.statistical_model = statistical_model
        self.statistical_model_config = statistical_model_config
        self.poi = poi
        self.outputfolder = outputfolder

        self.computation = computation_options[computation]
        self.debug = debug

        statistical_model_class = StatisticalModel.get_model_from_name(self.statistical_model)
        self.model = statistical_model_class.from_config(
            self.statistical_model_config, inputfolder=self.inputfolder)
        self.parameters_fittable = self.model.parameters.fittable
        self.parameters_not_fittable = self.model.parameters.not_fittable
        self.parameters_with_uncertainty = self.model.parameters.with_uncertainty

    @property
    def outputfolder(self) -> str:
        return self._outputfolder

    @outputfolder.setter
    def outputfolder(self, outputfolder):
        if outputfolder is None:
            # default output folder is the current working directory
            raise ValueError("outputfolder is not provided")
        else:
            self._outputfolder = os.path.abspath(outputfolder)
        if not os.path.exists(self._outputfolder):
            os.makedirs(self._outputfolder, exist_ok=True)

    @classmethod
    def from_config(cls, config_file_path: str, **kwargs) -> "Submitter":
        """Initializes the submitter from a yaml config file.

        Args:
            config_file_path (str): Path to the yaml config file.

        Returns:
            BlueiceExtendedModel: Statistical model.
        """
        config = load_yaml(config_file_path)
        return cls(**{**config, **kwargs})

    @staticmethod
    def arg_to_str(value, annotation) -> str:
        """
        Convert the argument to string for the submission script

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
            return 'None'
            # raise ValueError('provides argument can not be None')
        if annotation is str:
            return value
        elif annotation is int:
            return '{:d}'.format(value)
        elif annotation is float:
            # currently we only support 4 digits after the decimal point
            return '{:.4f}'.format(value)
        elif annotation is bool:
            return str(value)
        elif annotation is dict or annotation is list:
            # the replacement is needed because the json.dumps adds spaces
            return dumps(value).replace(' ', '')
        else:
            raise ValueError(
                f'Unknown annotation type: {annotation},'
                + ' it can only be str, int, float, bool, dict or list')

    @staticmethod
    def str_to_arg(value: str, annotation):
        """
        Convert the string to argument for the submission script

        Args:
            value: the string of the argument
            annotation: the annotation of the argument

        Returns:
            the value of the argument, can be various type
        """
        if value == 'None':
            return None
        if annotation is str:
            return value
        elif annotation is int:
            return int(value)
        elif annotation is float:
            return float(value)
        elif annotation is bool:
            if value == 'True':
                return True
            elif value == 'False':
                return False
            else:
                raise ValueError(
                    f'Unknown value type: {value}, '
                    'it can only be True or False')
        elif annotation is dict or annotation is list:
            # the replacement is needed because the json.dumps adds spaces
            return loads(value)
        else:
            raise ValueError(
                f'Unknown annotation type: {annotation}, '
                'it can only be str, int, float, bool, dict or list')

    def computation_tickets_generator(self):
        """
        Get the submission script for the current configuration.
        It generates the submission script for each combination of the computation options
        for Runner from to_zip, to_vary and in_common.
            - First, generate the combined computational options directly.
            - Second, update the input and output folder of the options.
            - Thrid, collect the non-fittable(settable) parameters into nominal_values.
            - Then, collect the fittable parameters into generate_values.
            - Finally, it generates the submission script for each combination.

        Yields:
            (str, str): the submission script and name output_file

        Todo:
            Add support for inputfolder
        """

        to_zip = self.computation.get('to_zip', {})
        to_vary = self.computation.get('to_vary', {})
        in_common = self.computation.get('in_common', {})
        allowed_keys = ['to_zip', 'to_vary', 'in_common']
        if set(self.computation.keys()) - set(allowed_keys):
            raise ValueError(
                'Keys in computation_options should be to_zip, to_vary or in_common, '
                'unknown computation options: {}'.format(
                    set(self.computation.keys()) - set(allowed_keys)))

        merged_args_list = compute_variations(to_zip=to_zip, to_vary=to_vary, in_common=in_common)

        # find run toyMC default args and annotations:
        # reference: https://docs.python.org/3/library/inspect.html#inspect.getfullargspec
        args, varargs, varkw, defaults, kwonlyargs, kwonlydefaults, annotations = inspect.getfullargspec(Runner.__init__)  # noqa
        # skip the first one because it is self(Runner itself)
        default_args = dict(zip(args[1:], defaults))

        common_runner_args = {
            'statistical_model': self.statistical_model,
            'statistical_model_config': self.statistical_model_config,
            'poi': self.poi,
        }

        for merged_args in tqdm(merged_args_list):
            function_args = deepcopy(default_args)
            # update defaults with merged_args and common_runner_args
            function_args.update(merged_args)
            function_args.update(common_runner_args)

            # update folder and i_batch
            for f in ['output_file', 'toydata_file']:
                if f in function_args:
                    function_args[f] = os.path.join(
                        self.outputfolder, add_i_batch(function_args[f]))

            # check if there are redundant arguments
            redundant_mask = function_args['parameter_definition'] is not None
            redundant_mask |= function_args['likelihood_config'] is not None
            if redundant_mask:
                raise ValueError(
                    'Please put the parameter_definition and likelihood_config '
                    'into statistical model configuration file.')

            # check if all arguments are supported
            intended_args = set(function_args.keys()) - {'n_batch'}
            acceptable_args = set(args[1:] + self.parameters_not_fittable + ['poi_expectation'])
            if not acceptable_args.issuperset(intended_args):
                logging.warning(
                    f'Not all arguments are supported, '
                    f'default arguments will be used for the following arguments: '
                    f'{acceptable_args - intended_args} '
                    f'and the following arguments will be ignored: '
                    f'{intended_args - acceptable_args}.')

            # distribute n_mc into n_batch, so that each batch will run n_mc/n_batch times
            if function_args['n_mc'] % function_args['n_batch'] != 0:
                raise ValueError('n_mc must be divisible by n_batch')
            function_args['n_mc'] = function_args['n_mc'] // function_args['n_batch']

            # only update folder
            if 'limit_threshold' in function_args:
                function_args['limit_threshold'] = os.path.join(
                    self.outputfolder, function_args['limit_threshold'])

            # update generate_values and nominal_values for runner
            self.update_runner_args(function_args)
            # update poi according to poi_expectation
            self.update_poi(function_args)

            allowed_keys = list(annotations.keys()) + ['poi_expectation', 'n_batch']
            if set(function_args.keys()) - set(allowed_keys):
                raise ValueError(
                    f'Keys in function_args should a subset of {allowed_keys}, '
                    'unknown computation options: {}'.format(
                        set(function_args.keys()) - set(allowed_keys)))

            n_batch = function_args['n_batch']
            for i_batch in range(n_batch):
                function_args['i_batch'] = i_batch

                for name in ['output_file', 'toydata_file', 'limit_threshold']:
                    if function_args.get(name, None) is not None:
                        function_args[name] = function_args[name].format(
                            **{**function_args['generate_values'],
                               **function_args['nominal_values'],
                               **function_args})

                script_array = []
                for arg, annotation in annotations.items():
                    script_array.append(f'--{arg}')
                    script_array.append(self.arg_to_str(function_args[arg], annotation))
                script = ' '.join(script_array)

                script = f'alea-run_toymc ' + ' '.join(map(shlex.quote, script.split(' ')))

                yield script, function_args['output_file']

    def update_poi(self, function_args):
        """
        Update the poi according to poi_expectation.
        First, it will check if poi_expectation is provided, if not so, it will do nothing.
        Second, it will check if poi is provided, if so, it will raise error.
        Third, it will check if poi ends with _rate_multiplier, if not so, it will raise error.
        Finally, it will update poi to the correct value according to poi_expectation.

        Args:
            function_args (dict): the arguments of Runner

        Caution:
            The expectation is evaluated under nominal_values in each batch.
        """
        if 'poi_expectation' not in function_args:
            return
        if function_args['poi'] in function_args:
            raise ValueError(
                f'You can not specify both {function_args["poi"]} '
                'along with poi_expectation, '
                'because it will be updated according to poi_expectation.')
        if not function_args['poi'].endswith('_rate_multiplier'):
            raise ValueError(
                f'poi {function_args["poi"]} should end with _rate_multiplier, '
                'if poi_expectation is provided, because you want to update '
                'the generate_values according to the expectations.')
        expectation_values = self.model.get_expectation_values(
            **{**function_args['generate_values'], **function_args['nominal_values']})
        component = function_args['poi'].replace('_rate_multiplier', '')
        poi_expectation = function_args['poi_expectation']
        nominal_expectation = expectation_values[component]
        ratio = poi_expectation / nominal_expectation
        # update poi to the correct value
        function_args['generate_values'][function_args['poi']] = ratio

    def update_runner_args(self, function_args):
        """
        Update the runner argumentsgenerate_values and nominal_values.
        If the argument is fittable, it will be added to generate_values,
        otherwise it will be added to nominal_values.

        Args:
            function_args (dict): the arguments of Runner
        """
        if function_args['nominal_values'] is None:
            function_args['generate_values'] = {}
        if function_args['nominal_values'] is not None:
            raise ValueError(
                'nominal_values should not be provided directly, '
                'it will be automatically deduced from the statistical model.')
        function_args['nominal_values'] = {}
        kw_to_pop = []
        for k, v in function_args.items():
            if k in self.parameters_fittable:
                function_args['generate_values'][k] = v
                kw_to_pop.append(k)
            elif k in self.parameters_not_fittable:
                function_args['nominal_values'][k] = v
                kw_to_pop.append(k)
        for k in kw_to_pop:
            function_args.pop(k)
        if set(function_args['generate_values'].keys()) - set(self.parameters_fittable):
            raise ValueError(
                f'The generate_values {function_args["generate_values"]} '
                f'should be a subset of the fittable parameters '
                f'{self.parameters_fittable} in the statistical model.')

    def submit(self):
        """Submit the jobs to the destinations."""
        raise NotImplementedError(
            "You must write a submit function your submitter class")
