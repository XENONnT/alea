import logging

from alea.configuration import Configuration


class Submitter():
    logging = logging.getLogger('submitter_logger')

    def __init__(
            self,
            runner_config: str,
            computation: str,
            debug: bool=False,
            loglevel='INFO',
            **kwargs):
        if type(self) == Submitter:
            raise RuntimeError(
                "You cannot instantiate the Submitter class directly, "
                "you must use a subclass where the submit method are implemented")
        loglevel = getattr(logging, loglevel.upper())
        self.logging.setLevel(loglevel)

        self.debug = debug
        self.computation = computation
        self.configuration = Configuration(runner_config, computation, **kwargs)

    def submit(self):
        raise NotImplementedError(
            "You must write a submit function your submitter class")
