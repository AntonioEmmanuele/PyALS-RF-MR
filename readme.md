# PyALS-RF Mirror — Experimental Evaluation Repository

## Repository and Goals

This repository is a mirror of the **PyALS-RF** project (https://github.com/SalvatoreBarone/pyALS-RF), created with the purpose of providing an immediate link between the experimental evaluation presented in  
*“Exploiting Modular Redundancy for Approximating Random Forest Classifiers”* and the employed tool.

**PyALS-RF** is a tool designed to investigate approximate Random Forest implementations on FPGA/ASIC.  
More specifically, the tool implements a behavioral simulation of the accelerator described in  
*“Implementing Hardware Decision Tree Prediction: A Scalable Approach”*  
([DOI: 10.1109/WAINA.2016.171](https://doi.org/10.1109/WAINA.2016.171)).

Using this behavioral simulation, the tool is capable of generating approximate classifiers by exploiting such behavior model.

---

## Installation

As this repository is only a mirror, we report here the installation guide.

Please note that **PyALS-RF** is still under development, as it is currently employed in different scientific works.  
For this reason, while there is a [Docker container](https://hub.docker.com/r/salvatorebarone/pyals-docker-image) available for an easier setup,  
the version of the tool provided in the container differs from the one used in the experimental evaluation.

The recommended approach to install the version of **PyALS-RF** contained in this repository  
is to pull the aforementioned container and attach the tool version from this repository as an additional folder inside it.  
This reflects the typical development setup adopted for the experiments in the paper.

---

### Installation Guide
#### Preliminary Step
As the tool performs several other functionalities, such as Approximate Logic Synthesis, it is required 

a `LutCatalog` for starting the container. 

To this aim, the folder `./pyALS-lut-catalog`, contains a well detailed guide detailing how to build a catalog.

In short, you need to : 

1. Install ```sqlite3'''. 

2. Run, withing the  `./pyALS-lut-catalog` folder, the command:
```
./import.sh
```

 

#### Installation Steps
1. **Install Docker** on your system (see [Get Docker](https://docs.docker.com/get-docker/)).

2. **Pull the container image**:
   ```bash
   docker pull salvatorebarone/pyals-docker-image:latest
   ```

3. **Run the script**

```bash
./runPyALSDock.sh -c ./pyALS-lut-catalog/lut_catalog.db -d ./pyALS_RF -s pathOfYourExperiments
```

This script executes the Docker container, attaching as a shared folder the version of **pyALS-RF** contained in this directory.  

**Important:** the version of the tool from this repository will now be found inside the folder `/home/user/pyALS-RF-dbg`,  
and **not** in `/home/user/pyALS` or `/home/user/pyALS-RF`, which are the default ones used inside the container.  

The script maps three different shared folders, located under `/home/user` inside the container:

- **shared (`pathOfYourExperiments`)** → the folder where experiments are executed and saved.  
  In our case, this will be the folder `examples`.  
- **lutCatalog (`./pyALS-lut-catalog/lut_catalog.db`)** → this file is required for running the container but is **not used** by the components executing the Modular Redundancy approximation.  
- **pyALS-RF-dbg (`./pyALS-RF`)** → the folder containing the tool version from this repository.  
  For this reason, the script maps it to `./pyALS-RF`.

If you want to map the experiment folder, in the the provided examples simply run: 

```bash
./runPyALSDock.sh -c ./pyALS-lut-catalog/lut_catalog.db -d ./pyALS_RF -s ./examples
```

In this case, the `shared` folder, will contain the content of the `examples` folder.

**If you want to install the tool directly on your machine**, follow the instructions present in the main repository (https://github.com/SalvatoreBarone/pyALS-RF). 

Beware, that at the time of the writing of this readme, we are currently developing the tool as attached to a docker repository. 

Once, the works will be done, the main image of the docker container will be updated. 

---

### Train a model

By leveraging scikit-learn as a back-end, the tool allows you to perform training of random forest classifiers, performed even with Quantization Aware Training. 

To train a model, you simply need one csv containing the columns of your model and labels, then the tool will automatically execute, and optionally quantize, training. 

The tool can perform hyperparameterization by adopting either a 5-fold grid-search or random-search.

All the datasets considered in the paper are trained using a grid search. 

Parameters employed are shown below: 

| **Hyperparameter**    | **Values**                                                                                  |
|------------------------|---------------------------------------------------------------------------------------------|
| `criterion`            | `["entropy", "gini"]`                                                                      |
| `max_depths`           | `[5, 7, 10, 15, 20]`                                                                       |
| `min_samples_split`    | `[2, 3, 5, 7, 10, 20, 30, 50, 100]`                                                        |
| `min_samples_leaf`     | `[1, 2, 3, 5, 7, 10, 20, 30, 50, 100]`                                                     |
| `ccp_alpha`            | `[0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.01, 0.05, 0.1]`                                  |

Regarding Quantization Aware Training, the tool simply quantizes, using a MinMax Scaler the input values for each column of the input dataset.

During inferences, the input thresholds of the model are simply reinterpreted as integers. 

For more details, please check “Dynamic Decision Tree Ensembles for Energy-Efficient Inference on IoT Edge Nodes” https://doi.org/10.1109/JIOT.2023.3286276.

To train a model, execute the previously discussed script. Once in the container, go to the directory /home/user/pyALS-RF-dbg and run this command: 

```bash
train <commandOption> <modelType> <hyperparameterizationType> <dataset> <datasetConfigJson> -n <NumberOfDesiredTrees> -f <trainTestSplitFraction> -q16 <quantizationFlag> -j <ncpus>
```
- **commandOption** : use always `autotune` for performing a grid search.
- **modelType**: type of model to train, use always `rf`.
- **hyperparameterizationType**: type of hyperparametization to employ before training. Use `grid` for 5-fold grid search and `random` for random search. In the case of a random search, 
      you can also set the number of iteration with -i <nIter>. For example, -i 500 for 500 iteration during a random seach.
- **dataset**: Path of the dataset file.
- **datasetConfigJson**: Path of the JSON configuration file, provided alongside the dataset. These files describe crucial aspects, like feature names, and separators. This files must be generated prior to importing the dataset in pyALS-RF. We provide a few examples in `examples/Datasets`.
- **ouputdir**: Output directory of the newly trained model.
- **numberOfDesiredTrees**: Number of decision trees desired in the ensemble.
- **trainTestSplitFraction**: fraction of training samples. 
- **quantizationFlag** : 1 for enabling 16 bit quantization, 0 otherwise, can be even left untouched.
- **ncpus**: Number of CPUs employed.

We provide an already **ready** script, in the `examples/scripts` folders. These are`examples/scripts/train/train.sh` and `examples/scripts/train/train_q16.sh`, and train models for the datasets Dry Bean and Statlog Segment, with number of trees equal at 20, for rapid tests.

```bash
./train_q16.sh
```

These take as input simply the name of the dataset and produce in output, in the folder `examples/trained_models<dataset_name>/rf_20` and 

`examples/trained_models_q16/<dataset_name>/rf_20` the non-quantized and quantized classifiers, respectively.

For example, the Statlog Segment, quantized at 16 bits, will be in the `examples/trained_models_q16/statlog_segment/rf_20`. 

We highlight that these script must be executed **inside the container**. They are under the folder  `shared/scripts/train/train_q16.sh-`

These model folders contain models exported in PMML and Joblib format, as well as the training and test set exported in csv. 

**Before running approximation**, you need to generate a configuration file for these models. These are different from the previous configuration files, and we **highly advise**, 

to generate  them using the script `examples/scripts/train/pyalsrf_config_generator.sh` and `examples/scripts/train/pyalsrf_config_generator_q16.sh`.

These scripts, take as input the name of the dataset trained, for instance: 

```bash
./pyalsrf_config_generator_q16.sh statlog_segment
```

By executing this script, **always inside the container**, a json5 file will be generated in model output directory. 

As an example, for the Statlog Segment dataset, the config file will be in `examples/trained_models_q16/statlog_segment/rf_20/config.json5`.

**To easily add a new dataset:**
- Import it into the `examples/datasets` folder.
- Create the config.json file and put it into `examples/datasets/<newDataset>/config.json`.
- Train the model using the previous discussed procedure. To this aim, we suggest to directly insert the dataset parameters into the provided training scripts. All the required parameters, involve
  the **datasets and configs** arrays, which simply the specification of the csv and .json config dataset path. 
- Run the generator script, `examples/scripts/train/pyalsrf_config_generator.sh`.
**If the outdir of the dataset is different from  `examples/trained_models_q16_test` or `examples/trained_models_test`, then also the configuration generator scripts must be modified** 

---

### Running Modular Redundancy approximation on trained classifiers

In order to approximate a trained model, inside the  `home/user/pyALS-RF-dbg/` folder of the container, is: 

```bash
./pyals-rf mr_heu -c <pathToJSON5Config> -q <quantizationType> -f <mrSetFraction> -r <rankingMethod> -m <mrOrder> -p <pruningDir> -c <csvDir> -j <nCPUs>
```
- **pathToJSON5Config**: Path to the configuration file of the trained model ( the one generated with the provided configuration generator scripts).
- **quantizationType**: Type of quantization applied to the during training. Depending on this parameter, internal thresholds of the model are reinterpreted as 16-bit integers or floats. Use `-q "int16"` if 16-bit QAT was applied.
- **mrSetFraction**:  Portion of the test set used to approximate the model. The remaining part is used to validate its accuracy.
- **rankingMethod**:  Method used to rank Decision Trees during approximation. Use "pertree_acc_heu" for the accuracy ranking and "pertree_margin_heu" for margin ranking.
- **pruningDir**:     Folder in which the set of removed leaves and approximate configuration (trees per class) of the model is stored alongside its approximate configuration.
- **csvDir**:         Folder in which the approximation report ( e.g. parameters such as the loss values) are stored.
- **nCPUs**:         Number of CPUs to use for executing the algorithm.

Even in this case, we provide the reader with additional scripts regarding our approach. 

In `examples/scripts/mr/launch.sh` `examples/scripts/mr/launch_q16.sh` there are two scripts to launch the modular redundancy approximation, with modular reduncancy order set to 9, for the 

Statlog and Dry Bean datasets, set to 20 trees for quantized and non quantized models. The scripts will simply launch a single repetition of the experiment. 

The output, for the non quantized model script, will be generated in the folder, `examples/mr_exp/pertree_acc_heu` and `examples/mr_exp/pertree_margin_heu`. In both folders, for both datasets,  a subfolder containing the name 

of the dataset will be created. The reader can observe the statistics of approximation in `examples/mr_exp/pertree_margin_heu/<dataset_name>/rf_<tree_num>/mr_report.csv`. 

Differently, the  configuration will be in `examples/mr_exp/pertree_margin_heu/<dataset_name>/rf_<tree_num>/mr_<selectedOrder>/cfg_1/`.

When running multiple repetition of the same experiment, different cfg folders will be generated.  

The same goes for the `examples/scripts/mr/launch_q16.sh` script in the examples/mrq16_exp` folder. 

The reader can add more datasets ( such as newly trained ones in the examples) and repetitions, by simply modifying these scripts  (or by executing the aforementioned command). 

---

### Generate per class-statistics and number of nodes from a set of Modular Redundancy approximation experiments

The command employed to generate per class statistics and number of nodes is : 

```bash
./pyals-rf mr_additional_estimations -c <pathToJSONConfig> -q <quantizationType> -j <nCPUs> -e <experimentsPath> -k <subpathK> -r <subpathRep> -l <kLowerBound> -u <kUpperBound> -s <kStep> -n <nRepetitions>
```

- **pathToJSONConfig** : Path to the JSON configuration file of the trained classifier (the same format as generated by the configuration scripts).

- **quantizationType** : Quantization type used for internal thresholds. Use "int16" if 16-bit Quantization-Aware Training (QAT) was applied.

- **nCPUs**: Number of CPUs to use during the estimation procedure. By default, all available CPUs are used.

- **experimentsPath**: Path to the directory containing the experimental results for a specific dataset.

- **subpathK**: Subdirectory naming pattern for modular-redundancy order experiments performed on the same dataset (default: "mr_"). For example, experiments for redundancy order 11 are stored under mr_11/.

- **subpathRep**: Subdirectory naming pattern for different repetitions of experiments under a given redundancy order (default: "cfg_").
For instance, mr_11/cfg_1 corresponds to the first repetition of the experiment with modular redundancy order 11.

- **kLowerBound**: Lower bound of the modular redundancy order range (default: 3).

- **kUpperBound**: Upper bound of the modular redundancy order range (default: 19).

- **kStep**: Step size for iterating through redundancy orders (default: 2).

- **nRepetitions**: Number of repetitions for each experiment at a fixed modular redundancy order (default: 30).

Example: 
```bash
./pyals-rf mr_additional_estimations  -c ../shared/trained_models_q16/statlog_segment/rf_20/config.json5   -e ../shared/mrq16_exp/pertree_acc_heu/statlog_segment/rf_20 -k mr_ -r cfg_ -l 9 -u 9 -s 2 -n 1 -j 20
```

This script will create, alongside the mr_report.csv file, two new files: `node_counts_report.csv` and `per_class_acc_report.csv`.  

---

### Generate HDL code

The command employed the HDL code of an exact and approximate classifier is:  

```bash
./pyals-rf generate_mr_hdl -c <pathToJSONConfig> -o <outpath> -q <quantizationType> -p <leafApproxPath> -a <approximateCfgPath> 
```

- **pathToJSONConfig** : Path to the JSON configuration file of the trained classifier (the same format as generated by the configuration scripts).

- **outpath**: Output path in which the code for the exact and approximate classifier will be generated. 

- **quantizationType** : Quantization type used for internal thresholds. Use "int16" if 16-bit Quantization-Aware Training (QAT) was applied.

- **leafApproxPath**: Path in which the pruning_conf.json5 file is generated, after mr_heu. In our scripts, it is under the cfgs folder.

- **approximateCfgPath**: Path to the per_class_cfg.json5 file of an approximate configuration. In our scripts, it is under the cfgs folder.

Example: 
```bash
./pyals-rf generate_mr_hdl -c ../shared/trained_models_q16/statlog_segment/rf_20/config.json5 -o ../shared/testHDL -q "int16" -p  ../shared/mrq16_exp/pertree_acc_heu/statlog_segment/rf_20/mr_9/cfg_1/pruning_conf.json5 -a ../shared/mrq16_exp/pertree_acc_heu/statlog_segment/rf_20/mr_9/cfg_1/per_class_cfg.json5
```

