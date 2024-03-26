# hyperparameters: configuration
# inputs: templates in the specified local directory
# outputs: all *.h5 files in the output folder


from alea.submitter import Submitter


class SubmitterHTCondor(Submitter):
    """Submitter for htcondor cluster.
    """
    def __init__(self, *args, **kwargs):
        self.name = self.__class__.__name__
        self.htcondor_configurations = kwargs.get("slurm_configurations", {})
        self.template_path = self.htcondor_configurations.pop("template_path", None)
        super().__init__(*args, **kwargs)
    
    def _validate_x509_proxy(self):
        raise NotImplementedError
    
    def _generate_sc(self):
        raise NotImplementedError
    
    def _generate_workflow(self):
        raise NotImplementedError
    
    def _plan_and_submit(self):
        raise NotImplementedError  

    def submit_workflow(self):
        raise NotImplementedError

