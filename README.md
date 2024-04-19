# Automatic Sensitivity Identification with Generative AI

## Project Description

Many collections of documents, such as emails, cannot be released to the public because they contain sensitive information. Previous literature has addressed the task of opening collections containing sensitive information by framing the problem as a classification task. We use a novel approach for sensitivity classification using zero-shot generative LLMs. We explore prompt engineering strategies to elicit better responses from these models.

## Link
https://github.com/ryangregson01/L5-project

This repository has been used for the entire development of the project. It includes code used in our experiment and during development where we trialled different approaches.

## Project Structure Overview:
* dissertation/ Source for the project dissertation
* notebooks/ The Jupyter notebooks used at the start of our project development, before we moved to more maintainable scripts. README discusses purpose of each notebook.
* scripts/ Source code for the project. README in folder describes purpose of each file.
* reports/ Reports saved from meetings. Moved to shared Notion site.
* results_notebook - has been kept outside of development notebook space. This was used to obtain all of our results and conclusions for our dissertation. Note some cells are not relevant as we only choose the most informative and useful analysis of our results to discuss.

## Running our code
* We used Python 3.10 and cuda > 12.0.
* Installing necessary requirements using requirements.txt. Note, if using our DSPy exploration scripts dspyrequirements.txt must be installed.
* You should fill out the config.py file if you wish to use Llama 2. You should also set your cache directory or remove this from models.py. (Note: This was a requirement using the undergraduate university machines due to personal storage limits.)
* In pipeling_config you can set sample size, and prompt_names to be used. Then within scripts run, 
  * where model name is a string that represents a key in the model_map in pipeline_config.
  * and prompt names are found in the getter function at the bottom of final_prompts.py.
```
python pipeline_config.py model_name
```
* Your results should write out to results/model_results/model_name/prompt_name
* Similarly, results_latex and see_eval can be used after a model_name(s) and prompt(s) of interest is set within these files. These print the results out into the shell.
