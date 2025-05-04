# CS 6320.001 Project Report   
### Shawn Pan  

### 1. Goal:
This project trains an OpenAI mode to perform grammar correction on sentences or paragraphs.

### 2. Preparation:
The data resource comes from Kaggle:   

Dataset: C4_200M Synthetic Dataset for Grammatical Error Correction
https://www.kaggle.com/datasets/felixstahlberg/the-c4-200m-dataset-for-gec

It contains a series of examples. Each example consists an original sentence with grammar issue, and a corrected sentence.

My goal is to run the fine-tuned model on the original sentences, then compare:

(1) The output sentence with original input, to see how much it improves   
(2) The output sentence with corrected sentence from data resource, to see how good it performs

### 3. Process
#### Note: For privacy concern, the OpenAI account related are stored in `config.ini` and not provided when uploaded.

(1) Run `generate_database.py` to randomly pull a total number of 500 examples from kaggle. It will be splitted into 80% (400) of training data, and 20% (100) of testing data, saved into two json files, both match the required format by OpenAI API. 

The files can be validated by `dataset_validation.py` to see if they indeed match OpenAI's requirements.

(2) To run a fine-tuning, use `train_grammar_model.py`. When the job is done, update fine-tuned model id in `config.ini`.

(3) Run `test_grammar_model.py` and select mode 1 for analyzing generated data, or mode 2 for correcting manually input sentences.

(4) For mode 1, the result is saved into `summarization.csv`. It can be further analyzed by `plot_performance.py` to plot graphs showing the model's performance.
