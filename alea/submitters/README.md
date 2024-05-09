# Submitters
## HTCondor Submitter Tips

This submitter will only work on OSG, assuming you are submitting from `ap23`.

### Setup
Please make sure you run this setup before submitting jobs, otherwise you will run into `Pegasus` related issues.
```
. setup_pegasus.sh
```
### Configuration
Following this as an example
```
htcondor_configurations:
  template_path: "/home/yuanlq/alea_inputs/nt_cevns_templates/v7"
  template_tarball_filename: "nt_cevns_templates_v7.tar.gz"
  cluster_size: 1
  request_cpus: 1
  request_memory: 2000
  request_disk: 2000000
  dagman_maxidle: 100000
  dagman_retry: 2
  dagman_maxjobs: 100000
  pegasus_transfer_threads: 4
  singularity_image: "/cvmfs/singularity.opensciencegrid.org/xenonnt/montecarlo:2024.04.1"
  running_configuration_filename: "/home/yuanlq/software/alea/lqtests/lq_b8_cevns_running.yaml"
  wf_id: "lq_b8_cevns_30"
```
- `template_path`: where you put your input templates. Note that all files have to have unique names, and no subfolders is allowed in `template_path`.
- `template_tarball_filename`: a filename of your choice, which is a tarball for all templates in `temp;ate_path` tarred by the submitter. It will be uploaded to the grid.
- `cluster_size`: clustering multiple `alea-run_toymc` jobs into a single job. For example, now you expect to run 100 individual `alea-run_toymc` jobs, and you specified `cluster_size: 10`, there will be only 10 `alea-run_toymc` in the end, each containing 10 jobs to run in sequence. Unless you got crazy amount of jobs like >200, I don't recommend changing it from 1.
- `request_cpus`: number of CPUs for each job. The default 1 should be good.
- `request_memory`: requested memory for each job in unit of MB. Please don't put a number larger than what you need, because it will significantly reduce our available slots.
- `request_disk`: requested disk for each job in unit of KB. Please don't put a number larger than what you need, because it will significantly reduce our available slots.
- `dagman_maxidle`: maximum of jobs allowed to be idle. The default 100000 is good for most cases.
- `dagman_retry`: number of automatic retry for each job when failure happen for whatever reason. Note that everytime it retries, we will have new resources requirement `n_retry * request_memory` and `n_retry * request_disk` to get rid of failure due to resource shortage.
- `dagman_maxjobs`: maximum of jobs allowed to be running. The default 100000 is good for most cases.
- `pegasus_transfer_threads`: number of threads for transfering handled by `Pegasus`. The default 4 is good so in most cases you want to keep it.
- `singularity_image`: the jobs will be running in this singularity image.
- `wf_id`: name of user's choice for this workflow. If not specified it will put the datetime as `wf_id`.


### Usage
Make sure you configured the running config well, then you just simply pass `--htcondor` into your `alea-submission` command.

In the end of the return, it should give you something like this:
```
Worfklow written to

	/scratch/yuanlq/workflows/runs/lq_b8_cevns_30
```
Keep this directory in mind, since all logs will go there and we call it "run directory"

### Useful Commands
To check the progress, you want to do `condor_q`. For example
```
(XENONnT_development) yuanlq@ap23:~$ condor_q


-- Schedd: ap23.uc.osg-htc.org : <192.170.231.144:9618?... @ 05/09/24 16:45:39
OWNER  BATCH_NAME                      SUBMITTED   DONE   RUN    IDLE   HOLD  TOTAL JOB_IDS
yuanlq xenonnt-0.dag+12972724         5/9  16:16   4155    114    288      _   5439 12972725.0 ... 12973919.0
yuanlq xenonnt-0.dag+12972825         5/9  16:16  15350    108     84      1  16622 12972933.0 ... 12973920.0
yuanlq alea-workflow-0.dag+12973662   5/9  16:35      4      3      _      _     11 12973684.0 ... 12973686.0
```

Here you see `alea-workflow-0.dag+12973662` is running.

To cancel it, you want to do:
```
condor_rm 12973662
```

If you want to know more details, like checking why the job failed, just do this in your "run directory". This command should give you a summary of the workflow, including errors encountered if any.
```
pegasus-analyzer /scratch/yuanlq/workflows/runs/lq_b8_cevns_30
```

Let's say now the workflow is ended (you see nothing from `condor_q`). If it didn't finish successfully for weird error, a good thing to do is just to rerun it. However, keep in mind that the workflow itself will automatically retries up to `dagman_retry` times (defined in your running config). To rerun the failed jobs only, just do this.
```
pegasus-run /scratch/yuanlq/workflows/runs/lq_b8_cevns_30
```

To collect the final outputs, there are two ways
- Check your folder `/scratch/$USER/workflows/outputs/<wf_id>/`
- A redundant way is to get files from dCache, in which you have to use `gfal` command to approach. For example ```gfal-ls davs://xenon-gridftp.grid.uchicago.edu:2880/xenon/scratch/yuanlq/lq_b8_cevns_30/```