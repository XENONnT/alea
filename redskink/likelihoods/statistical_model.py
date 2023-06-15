import inspect
from inference_interface import toydata_from_file, toydata_to_file

class StatisticalModel:
    """
    Class that defines a statistical model.
    The statisical model contains two parts that you must define yourself:
        - a likelihood function, ll(self, parameter_1, parameter_2... parameter_n):
            a function of a set of named parameters
            returns a float expressing the loglikelihood for observed data
            given these parameters
        - a data generation method generate_data(self, parameter_1, parameter_2... parameter_n):
            a function of the same set of named parameters
            returns a full data set:

    Methods:
         __init__
         required to implement:

         ll
         generate_data

         optional to implement:
         get_mus
         get_likelihood_term_names

         Implemented here:
         set_data
         get_data
         store_data
         fit
         get_confidence_interval
         get_expectations
         get_parameter_list
         print_config


    Other members:
        _data = None
        _config = {}
        _confidence_level = 0.9
        _confidence_interval_kind = "upper,lower,unified"
        _fit_guess = {}
        _fixed_parameters = []
    """
    def ll(self, **kwargs) -> float:
        return NotImplementedError("You must write a likelihood function for your statistical model or use a subclass where it is written for you")
        mua = self._data[0]["mua"]
    def generate_data(self, **kwargs):
        return NotImplementedError("You must write a data-generation method for your statistical model or use a subclass where it is written for you")
        ret = np.array(,dtype=("n",int),("b_meas",int))
        data_shape = []

    def __init__(self,
                 data = None,
                 config: dict = dict(),
                 confidence_level: float = 0.9,
                 confidence_interval_kind:str = "unified",
                 fit_guess:dict = dict(),
                 fixed_parameters:dict = dict(),
                 default_values = dict(),
                 **kwargs):
        self._data = data
        self._config = config
        self._confidence_level = confidence_level
        self._confidence_interval_kind = confidence_interval_kind
        self._fit_guess = fit_guess
        self._fixed_parameters = fixed_parameters

        self._parameter_list = set(inspect.signature(self.ll))
        if self._parameter_list != set(inspect.signature(self.generate_data())):
            raise AssertionError("ll and generate_data must have the same signature (parameters)")

    def set_data(self, data):
        """
        Simple setter for a data-set-- mainly here so it can be over-ridden for special needs.
        Data-sets are expected to be in the form of a list of one or more structured arrays-- representing the data-sets of one or more likelihood terms.
        """
        self._data = data

    def get_data(self):
        """
        Simple getter for a data-set-- mainly here so it can be over-ridden for special needs.
        Data-sets are expected to be in the form of a list of one or more structured arrays-- representing the data-sets of one or more likelihood terms.
        """
        return self._data

    def store_data(self, file_name, data_list, data_name_list=None, metadata = None):
        """
        Store a list of datasets (each on the form of a list of one or more structured arrays)
        Using inference_interface, but included here to allow over-writing.
        structure would be: [[datasets1],[datasets2]... [datasetsn]]
        where each of datasets is a list of structured arrays
        if you specify, it is set, if not it will read from self.get_likelihood_term_names
        if not defined, it will be ["0","1"..."n-1"]
        """
        if data_name_list is None:
            try:
                data_name_list = self.get_likelihood_term_names()
            except NotImplementedError:
                data_name_list = ["{:d}" for range(len(data_list[0]))]

        kw = dict(metadata = metadata) if metadata is not None
        toydata_to_file(file_name, data_list, data_name_list, **kw)



    def fit(self, ):
    def get_confidence_interval(self) -> float, float:
    def get_expectations(self):
        return NotImplementedError("get_expectation is optional to implement")

    def get_likelihood_term_names(self):
        """
        It may be convenient to partition the likelihood in several terms,
        you can implement this function to give them names (list of strings)
        """
        return NotImplementedError("get_likelihood_term_names is optional to implement")

    def get_likelihood_term_from_name(self, likelihood_name):
        """
        returns the index of a likelihood term if the likelihood has several names
        """
        try:
            if hasattr(self, likelihood_names):
                likelihood_names = self.likelihood_names
            else:
                likelihood_names = self.get_likelihood_term_names()
            return {n:i for i,n in enumerate(likelihood_names)}[likelihood_name]
        except e:
            print e
            return None



    def get_parameter_list(self):
        """returns a set of all parameters that the generate_data and likelihood accepts"""
        return self._parameter_list

    def print_config(self):
        for k,i in self.config:
            print(k,i)

