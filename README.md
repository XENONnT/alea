# alea
A tool to perform toyMC-based inference constructions

`alea` is a public version of previously private XENON likelihood definition and inference construction code (""binference"" that used the blueice repo https://github.com/JelleAalbers/blueice. 

Binference was developed for XENON1T WIMP searches by Knut Dundas Morå, and for the first XENONnT results by Robert Hammann, Knut Dundas Morå and Tim Wolf


Installation on Midway and OSG:

  * source analysis environment: (
    examples in this [repo](https://github.com/XENONnT/env_starter)
    or this command:

    `source  /cvmfs/xenon.opensciencegrid.org/releases/nT/development/setup.sh`
    )

  * cd binference
  * pip install -r requirements.txt --user
  * pip install . --user

# Try running it!
 * `cd ..`
 * `mkdir run_folder`
 * `cp ../binference/notebooks/example_batch_runs.ipynb .`
 * Start a jupyter notebook, and give the notebook a try!


## Example how to run the sensitivity on OSG

### 1. Installation
Install binference on OSG.
It is also highly recommended to set up SSH keys between midway and OSG and the other way around.

### 2. Run discovery_powers with the command:


```bash
python binferene/script/submission_script.py --config binference/configs/ll_nt_lowfield_v3_ER_rate190_AC17.yaml \
  --computation=discovery_powers \
  --OSG \
  --submit \
  --outputfolder_overwrite PATH_TO_YOUR_OUTPUTFOLDER
```
This command will submit to OSG with the specifications given in the config file. It is advisable to first try it with the `--local` option instead of `--OSG` and with the `--debug` option. For this kind of debugging `--submit` is not needed. If you chose to run on midway you can exchange the `--OSG` flag for `--midway`. This command (on OSG) will produce a number of `tar.gz`-files in the output folder.

### 3. Compute thresholds with the command:
From step one we are able to compute the Neyman thresholds. Specifications need to be given in the config file. To compute them you need to run:
```bash
python binferene/script/submission_script.py --config binference/configs/ll_nt_lowfield_v3_ER_rate190_AC17.yaml \
  --computation=threshold \
  --local \
  --unpack \
  --outputfolder_overwrite PATH_TO_YOUR_OUTPUTFOLDER
```
The `--unpack` option is important if you did not unpack the `tar.gz`-files by your self. The outcome of this step is a file containing the thresholds which is conveniently called `thresholds.h5`.

### 4. Compute sensitivity with the command:
To compute the sensitivity with the thresholds computed in 3., you need to specify the `limit_threshold` as the file in which the thresholds were written as it is done in the example.

To compute them you need to run:
```bash
python binferene/script/submission_script.py --config binference/configs/ll_nt_lowfield_v3_ER_rate190_AC17.yaml \
  --computation=sensitivity \
  --OSG \
  --submit \
  --outputfolder_overwrite PATH_TO_YOUR_OUTPUTFOLDER
```
The outcome of this step again will be a number of `tar.gz`-files. You can either unpack them manually or use the script which is under `binference/scripts/get_data_from_OSG.py`. Usually you want to run this script from midway to transfer your results there and make plots of the outputs.

### 5. Transfer outputs to midway
On midway (where you also need a binference installation, if you want to run this script), you need to run:

```bash
binference/scripts/get_data_from_OSG.py --OSG_path PATH_WITH_DATA_ON_OSG \
  --user YOUR_OSG_USERNAME
  (--midway_path PATH_WHERE_TO_STORE_ON_MIDWAY)
```
This command will NOT transfer anything. It will print out the commands that are going to be executed when you add the `--get` flag. After adding this flag, data will be unpacked on OSG and transfered to midway.



## OSG Tips

When submitting with the script to OSG you can check the status of your jobs with

`condor_q`

`condor_q --nobatch`

`condor_q -analyze JOB_ID`

When submitting the jobs a periodic release is implemented 5 times.
This should be sufficient. If it is not, you can try:

`condor_release JOB_ID`


## Resources:
* [Video for HTCondor](https://www.youtube.com/watch?v=oMAvxsFJaw4)

