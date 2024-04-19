# Scripts

## Overview

* Scripts are used to run our experiments, entering at pipeline_config.py for a single model (or run_experiments.sh).
* The aware folder explores the knowledge our LLMs have of sensitive personal information.
* The dspy folder is where we experimented with DSPy. We find using our text in DSPy formats followed the same trends as our written strings; however, we did not have enough time to run complete experiments with DSPy. Furthermore, DSPy teleprompt optimisers were trialled but could not search the prompt-space without memory issues.
* The host folder allowed for a short cpu job on cluster. Note this has not been run. Furthermore, issues with Docker and ir_datasets occur, hence fetching SARA into a csv in this folder.
* We will briefly discuss our scripts:
* * Analysis active prompt - Used for Section 6 for the dissertation.
* * Compare ML - Used for results - uses traditional approaches and compares them against Mistral Base and Best. Used alongside ml.py.
* * For our experimental pipeline we use: dataset.py to retrieve SARA, preprocess_sara.py to clean SARA, models.py to load our LLM, model.py to make inferences on documents. Pipeline.py controls these files, and finally writes our to a JSON file in the results folder. Pipeline_config.py abstracts automation in the pipeline and is called to run the experiment (or started through run_experiments.sh).
* * enron_genre_errors, eval, logits_exploration, nameless_preprocess and person_hide helped us explore possibilities for the project but are not used within for final dissertation at all.
* * few is a script to find most similar documents for few-shot.
* * final_prompts contains our final prompts used in experiments. We move old prompts to the old prompts folder to prevent confusion.
* * Results latex and see eval let us evaluate the results of our prompting strategies.


