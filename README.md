nli2paraphrases
==============================

Source code repository accompanying the paper `Extracting and filtering paraphrases by bridging natural language 
inference and paraphrasing`.

Setup
-----
```shell
# Make sure to run this from the root of the project (top-level directory)
$ pip3 install -r requirements.txt
$ python3 setup.py install
```

Project Organization
------------

    ├── README.md          
    ├── experiments        <- Experiment scripts, through which training and extraction is done
    ├── models             <- Intended for storing fine-tuned models and configs
    ├── requirements.txt   
    ├── setup.py           
    ├── src                <- Core source code for this project
    │   ├── __init__.py    
    │   ├── data           <- data loading scripts
    │   ├── models         <- general scripts for training/using a NLI model
    │   └── visualization  <- visualization scripts for obtaining a nicer view of extracted paraphrases


Getting started
----------------
As an example, let us extract paraphrases from **SNLI**.

The training and extraction process largely follows the same track for other datasets (with some new or removed 
flags, run scripts with `--help` flag to see the specifics).

In the example, we first fine-tune a `roberta-base` NLI model on SNLI sequences (s1, s2).  
Then, we use the fine-tuned model to predict the reverse relation for entailment examples, and select only those 
examples for which entailment holds in both directions.
The extracted paraphrases are stored into `extract-argmax`.

This example assumes that you have access to a GPU. If not, you can force the scripts to use CPU by setting `--use_cpu`, 
although the whole process will be much slower.  

```shell
# Assuming the current position is in the root directory of the project
$ cd experiments/SNLI_NLI

# Training takes ~1hr30mins on Colab GPU (K80)
$ python3 train_model.py \
--experiment_dir="../models/SNLI_NLI/snli-roberta-base-maxlen42-2e-5" \
--pretrained_name_or_path="roberta-base" \
--model_type="roberta" \
--num_epochs=10 \
--max_seq_len=42 \
--batch_size=256 \
--learning_rate=2e-5 \
--early_stopping_rounds=5 \
--validate_every_n_examples=5000

# Extraction takes ~15mins on Colab GPU (K80)
$ python3 extract_paraphrases.py \
--experiment_dir="extract-argmax" \
--pretrained_name_or_path="../models/SNLI_NLI/snli-roberta-base-maxlen42-2e-5" \
--model_type="roberta" \
--max_seq_len=42 \
--batch_size=1024 \
--l2r_strategy="ground_truth" \
--r2l_strategy="argmax"
```


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
