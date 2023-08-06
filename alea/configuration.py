import os
import inspect
import logging
from copy import deepcopy
from json import dumps, loads

from tqdm import tqdm
from utilix import batchq

from alea.runner import Runner
from alea.utils import load_yaml, compute_variations, add_i_batch


class Configuration():
    logging = logging.getLogger('configuration_logger')
    wildcards_for_threshold = [
        'signal_rate_multiplier',
        'signal_expectation', 
        'n_mc',
        'n_batch',
    ]
    batchq_arguments = [
        'hours', 'mem_per_cpu',
        'container', 'bind',
        'partition', 'qos', 'cpus_per_task', 'exclude_nodes',
    ]

    def __init__(
            self,
            runner_config: str,
            computation: str = 'discovery_power',
            model_config: str = None,
            loglevel='INFO',
            **kwargs):
        loglevel = getattr(logging, loglevel.upper())
        self.logging.setLevel(loglevel)
        self._check_batchq_arguments()

        self.runner_config = load_yaml(runner_config)
        if model_config is None:
            if 'statistical_model_config' not in self.runner_config:
                raise ValueError(
                    'statistical_model_config must be provided in the runner_config '
                    'if it is not specified by the model_config argument')
            self.model_config = load_yaml(self.runner_config['statistical_model_config'])
        else:
            self.model_config = load_yaml(model_config)

        self.computation = computation

        outputfolder = kwargs.get('outputfolder', self.runner_config['outputfolder'])
        if not os.path.exists(outputfolder):
            os.makedirs(outputfolder, exist_ok=True)
        self.runner_config['outputfolder'] = outputfolder

        self.computation_config = self.runner_config['computation'][self.computation]

        self.parameters_to_vary = self.computation_config.get('parameters_to_vary', {})
        self.parameters_in_common = self.computation_config.get('parameters_in_common', {})
        self.parameters_to_zip = self.computation_config.get('parameters_to_zip', {})
        self.parameters_as_wildcards = self.computation_config.get('parameters_as_wildcards', {})

    @property
    def server_arguments(self):
        {
            'midway': {
                'midway_path': self.runner_config.get('midway_path', None),
                'midway_parameters': self.runner_config.get('midway_parameters', None),
            },
            'OSG': {
                'OSG_path': self.runner_config.get('OSG_path', None),
                'OSG_parameters': self.runner_config.get('OSG_parameters', None),
            },
        }

    def _check_batchq_arguments(self):
        """Check if the self.batchq_arguments are valid"""
        args = set(inspect.signature(batchq.submit_job).parameters)
        if not set(self.batchq_arguments).issubset(args):
            raise ValueError(
                f'The following arguments are not supported of utilix.batchq.submit_job: '
                f'{set(self.batchq_arguments) - set(args)}.')

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
                    f'Unknown value type: {value},'\
                    + ' it can only be True or False')
        elif annotation is dict or annotation is list:
            # the replacement is needed because the json.dumps adds spaces
            return loads(value)
        else:
            raise ValueError(
                f'Unknown annotation type: {annotation},'
                + ' it can only be str, int, float, bool, dict or list')

    def get_submission_script(self):
        """Get the submission script for the current configuration"""
        common_runner_args = {
            'statistical_model': self.runner_config['statistical_model'],
            'statistical_model_config': self.runner_config['statistical_model_config'],
            'poi': self.runner_config['poi'],
        }
        merged_args_list = compute_variations(
            parameters_to_zip=self.parameters_to_zip,
            parameters_to_vary=self.parameters_to_vary,
            parameters_in_common=self.parameters_in_common)

        # find run toyMC default args and annotations:
        # reference: https://docs.python.org/3/library/inspect.html#inspect.getfullargspec
        args, varargs, varkw, defaults, kwonlyargs, kwonlydefaults, annotations = inspect.getfullargspec(Runner.__init__)  # noqa
        # skip the first one because it is self(Runner itself)
        default_args = dict(zip(args[1:], defaults))

        for merged_args in tqdm(merged_args_list):
            function_args = deepcopy(default_args)
            # update defaults with merged_args and common_runner_args
            function_args.update(merged_args)
            function_args.update(common_runner_args)

            redundant_mask = function_args['parameter_definition'] is not None
            redundant_mask |= function_args['likelihood_config'] is not None
            if redundant_mask:
                raise ValueError(
                    'Please put the parameter_definition and likelihood_config '
                    'into statistical model configuration file.')

            intended_args = set(function_args.keys()) - {'n_batch'}
            acceptable_args = set(args[1:])
            if not acceptable_args.issuperset(intended_args):
                logging.warn(
                    f'Not all arguments are supported, '
                    f'default arguments will be used for the following arguments: '
                    f'{acceptable_args - intended_args} '
                    f'and the following arguments will be ignored: '
                    f'{intended_args - acceptable_args}.')

            # distribute n_mc into n_batch, so that each batch will run n_mc/n_batch times
            if function_args['n_mc'] % function_args['n_batch'] != 0:
                raise ValueError('n_mc must be divisible by n_batch')
            function_args['n_mc'] = function_args['n_mc'] // function_args['n_batch']

            batched_filenames = {
                'output_file': None,
                'toydata_file': None,
            }
            for name in batched_filenames.keys():
                # so that we'll index by i_batch
                if function_args[name] is not None:
                    batched_filenames[name] = add_i_batch(function_args[name])

            n_batch = function_args['n_batch']
            for i_batch in range(n_batch):
                function_args['i_batch'] = i_batch

                for name, file_pattern in batched_filenames.items():
                    if function_args[name] is not None:
                        function_args[name] = file_pattern.format(**function_args)

                for f in list(batched_filenames.keys()) + ['limit_threshold']:
                    if (f in function_args) and (function_args[f] is not None):
                        function_args[f] = os.path.join(
                            self.runner_config['outputfolder'], function_args[f])

                script_array = []
                for arg, annotation in annotations.items():
                    if arg == 'parameter_definition':
                        print('here')
                    script_array.append(f'--{arg}')
                    script_array.append(self.arg_to_str(function_args[arg], annotation))
                script = ' '.join(script_array)

                batchq_kwargs = {}
                for arg in self.batchq_arguments:
                    if arg in function_args:
                        batchq_kwargs[arg] = function_args[arg]
                yield script, function_args['output_file'], batchq_kwargs
