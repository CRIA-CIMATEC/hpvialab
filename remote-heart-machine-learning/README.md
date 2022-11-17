<h3 align="center">Machine Learning Structure Template and Development Project</h3>

<div align="center">

[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![pipeline status](http://200.9.65.23/joseviniciusdantas/bias-mitigation/badges/develop/pipeline.svg)](http://200.9.65.23/joseviniciusdantas/bias-mitigation/commits/develop)
[![coverage report](http://200.9.65.23/joseviniciusdantas/bias-mitigation/badges/develop/coverage.svg)](http://200.9.65.23/joseviniciusdantas/bias-mitigation/commits/develop)
</div>

---

<p align="center"> We provide Python code templating from which developers can readily set up a model training environment with a particular focus on the analysis of fairness issues.
		<br>
</p>

## 📝 Contents table

- [Getting Started](#getting_started)
- [Prerequisites](#prerequisites)
- [Attention](#Attention)
- [Usage](#usage)
- [Running tests](#tests)


## 🏁 Getting Started <a name = "getting_started"></a>

These instructions will provide a copy of the project running on your local machine for development, testing and usage purposes.

The structure of folders was based on https://towardsdatascience.com/structuring-machine-learning-projects-be473775a1b6.

Folders / files and descriptions:
```
atum/      : Code template built on top of development code
atum_dev/      	    : Development code where new features are added and tested
```

Template structure:
```
atum/				    : Template folder.
  |cookiecutter.json		    : Cookiecutter configuration file with default arguments.
  |tests/			    : Folder containing scripts to test functions.
  |	conftest.py		    : Pytest config that contains a fixture.
  |	test_combinations.py		    : Test file used to execute several combinations of the package.
  |main.py : File containing functions relevant to the a package call.
  |pytest.ini : File containing pytest configurations.
  |{{cookiecutter.directory_name}}/ : Base directory that will be renamed after cookiecutter execution.
  |   data/			    : Folder with project data.
  |      interim/		    : Transformed intermediate data, not ready for modelling.
  |      processed/		    : Prepared data, ready for modelling.
  |      raw/			    : Immutable original data.
  |         api_conf.yaml    : API YAML with informations about the current experiment.
  | dist/			    : Folder containing the API library that needs to be installed.
  | results/			    : Folder containing results from the experiments organized by execution time.
  | src/			    : Folder containing project source code.
  |   data/			    : Folder containing scripts to download/generate/treat data.
  |      __init__.py		    : Package init that contains imports and the sub-parser method.
  |      load.py		    : File containing funtions to load datasets and store datasets info.
  |      torch_dataset.py	    : File containing a class used to use the dataset with pytorch.
  |   model/			    : Folder containing scripts to train and predict.
  |   	models_fair/			    : Folder containing scripts to train and predict models from FAIR paper.
  |   tests/			    : Folder containing scripts to test functions.
  |      conftest.py		    : Pytest config that contains a fixture.
  |      test_eg.py		    : Example of test file to use with pytest.
  |   main.py			    : Script responsible for centralizing project execution.
  |   Pipeline_report.ipynb	    : Notebook responsible for plotting the results of each experiment.
  |   pipeline_support.py	    : Script that contains functions to plot the results use by the notebook.
atum_dev/				    : Development project folder.
dist/				    : Package distributables.
papers/				    : Papers written about the experiments and the project structure.
atum.sh			    : Installation file.
makefile			    : Package makefile that creates the package distributables.
setup.cfg			    : Static configuration file.
setup.py			    : Dynamic configuration file to set the pipx env and create the CTL.
```

## Prerequisites <a name = "prerequisites"></a>
```
pillow==9.0.2
ipython==8.4.0
pandas==1.4.3
numpy==1.22.4
torchsummary==1.5.1
aif360==0.4.0
fairlearn==0.4.6
pytorch==1.12.1
cpuonly==2.0
scikit-learn==1.1.2
pyyaml==6.0
tensorflow==2.9.1
iteround==1.0.4
pytorch-ignite==0.4.9
pip==22.1
wrapper==0.0.7
aequitas==0.41.0
cookiecutter==2.1.1
natsort==8.1.0
whylogs==1.0.9
```

### Attention!

The following code was changed, [Issue 280 from AIF360]( https://github.com/Trusted-AI/AIF360/issues/280).
Line 147 of {path_to_env_name}/lib/python3.9/site-packages/aif360/sklearn/inprocessing/exponentiated_gradient_reduction.py:

```
X = X.drop(self.prot_attr)
```

To:

```
X = X.drop(self.prot_attr, axis=1)
```

A message ("get_disparity_predefined_group()") inside a function of the Aequitas library is being printed during training, to hide that, we comment the line 359 of the file "{path_to_env_name}/lib/python3.9/site-packages/aequitas/bias.py".

## 🚀 Usage <a name="usage"></a>

To install the command line tool, the environment and the package, use this command in the root folder:
```
source atum.sh env_name
```

To install the environment, enter this inside the terminal's "Bias_Mitigation" (root) folder:
```
conda create --name env_name
conda activate env_name
pip install .
ipython kernel install --name env_name --user
```

If it fails because of the wrapper, aequitas or aif360 library:
```
pip install dist/wrapper-0.0.10-py3-none-any.whl
pip install git+http://200.9.65.29/joseviniciusdantas/aequitas.git#egg=aequitas
pip install git+http://200.9.65.29/joseviniciusdantas/aif360.git#egg=aif360
```

To update the CTL automatically, enter this inside the terminal's "Bias_Mitigation" (root) folder:
```
pip install -e .
```

Examples:
```
atum create
atum run
atum create run --actions train_torch_classifier eval_results --dataset_name "Compas AIF360" --epochs 4 --result_folder_path current --metric "Predictive Equality" --batch_size 100 --proportion_type epoch --balance_start middle
```

Use the template to develop another project:
```
cookiecutter atum/
<Answer questionnaire>
cd <directory_name>
```

Execution of a basic classifier with compas (without charge description input):
```
python src/main.py --actions train_torch_classifier eval_results --dataset_name Compas --epochs 4 --result_folder_path current --metric "Predictive Equality" --batch_size 100 --proportion_type epoch --balance_start middle
```

Execution of a basic classifier with compas from AIF360 (with charge description input):
```
python src/main.py --actions train_torch_classifier eval_results --dataset_name "Compas AIF360" --epochs 4 --result_folder_path current --metric "Predictive Equality" --batch_size 100 --proportion_type epoch --balance_start middle
```

Execution of AdversarialDebiasing with compas from AIF360 (with charge description input):
```
python src/main.py --actions train_adversarial eval_results --dataset_name "Compas AIF360" --epochs 4 --result_folder_path current --metric "Predictive Equality" --batch_size 100 --proportion_type epoch --balance_start middle
```

Execution of PrejudiceRemover with compas from AIF360 (with charge description input):
```
python src/main.py --actions prejudice_remover eval_results --dataset_name "Compas AIF360" --epochs 4 --result_folder_path current --metric "Predictive Equality" --batch_size 100 --proportion_type epoch --balance_start middle
```

Execution of ExponentiatedGradientReduction with compas from AIF360 (with charge description input):
```
python src/main.py --actions gradient_reduction eval_results --dataset_name "Compas AIF360" --epochs 4 --result_folder_path current --metric "Predictive Equality" --batch_size 100 --proportion_type epoch --balance_start middle
```

Execution of FAIR with compas from AIF360 (with charge description input):
```
python src/main.py --actions fair eval_results --dataset_name "Compas AIF360" --epochs 4 --result_folder_path current --metric "Predictive Equality" --batch_size 100 --model_name FAIR_plus_scalar_class --loss_w_name exp_3
```

### 🔧 Running tests <a name = "tests"></a>

#### Test cookiecutter
```
cd atum
pytest
```

#### Test development code
```
cd atum_dev
pytest
```
