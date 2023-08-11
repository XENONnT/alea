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
    All the source of template should be from the same folder.
    All the output, like toydata and fitting results, should be in the same folder.
    """
    logging = logging.getLogger('submitter_logger')

    def __init__(
            self,
            statistical_model,
            statistical_model_config,
            poi,
            computation_options,
            computation='discovery_power',
            outputfolder=None,
            debug: bool = False,
            loglevel='INFO',
            **kwargs):
        if isinstance(self, Submitter):
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
        if value is None:
            return 'None'
            # raise ValueError('provides argument can not be None')
        if annotation is str:
            return value
        elif annotation is int:
            return '{:d}'.format(value)
        elif annotation is float:
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
    def str_to_arg(value, annotation) -> str:
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
        """Get the submission script for the current configuration"""

        statistical_model_class = StatisticalModel.get_model_from_name(self.statistical_model)
        # TODO: add support and test for inputfolder
        self.model = statistical_model_class.from_config(
            self.statistical_model_config, inputfolder=self.inputfolder)

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
            acceptable_args = set(args[1:])
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

            n_batch = function_args['n_batch']
            for i_batch in range(n_batch):
                function_args['i_batch'] = i_batch

                for name in ['output_file', 'toydata_file', 'limit_threshold']:
                    if function_args.get(name, None) is not None:
                        function_args[name] = function_args[name].format(**function_args)

                script_array = []
                for arg, annotation in annotations.items():
                    script_array.append(f'--{arg}')
                    script_array.append(self.arg_to_str(function_args[arg], annotation))
                script = ' '.join(script_array)

                script = f'alea-run_toymc ' + ' '.join(map(shlex.quote, script.split(' ')))

                yield script, function_args['output_file']

    def submit(self):
        raise NotImplementedError(
            "You must write a submit function your submitter class")
