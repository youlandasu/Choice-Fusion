## Choice Fusion as Knowledge for Zero-Shot Dialogue State Tracking

This is the PyTorch implementation of the following paper accepted by [ICASSP 2023](https://2023.ieeeicassp.org/):
[Ruolin Su](https://github.com/youlandasu), Jingfeng Yang, Ting-Wei Wu, Biing-Hwang Juang. **Choice Fusion as Knowledge for Zero-Shot Dialogue State Tracking.**

![Overview](https://github.com/youlandasu/Choice-Fusion/choice-fusion.png)

### Install Dependency
```
conda creat -n py38 python=3.8
conda activate py38
pip install -r requirements.txt
```
### Download and Create the Question-Answering Dataset
Download [RACE dataset](http://www.cs.cmu.edu/~glai1/data/race/) and put it under *qa_data* folder.
Download other QA datasets:
```
./download_data.sh
```
Combine and pre-process the QA datasets:
```
python create_qa_data.py
```

### Download and Create the MultiWOZ2.1 Dataset
```
python create_data_mwoz.py
```

### Train
1. Train our model with **appreciated-choice selection**:
```
./run_qa_pretrain_t5.sh pretrain
```
2. Train our model with **choice fusion mechanism** including **appreciated-choice selection** and **context-knowledge fusion**:
```
./run_qa_pretrain_t5.sh pretrain_fusion
```
`--percentage` The percentage of combined QA data for training.
`--max_seq_length` The max length of the input tokens.
`--num_train_epochs` The num of training epochs.
`--overwrite_cache` Whether or not use the cached training dataset.
The number of `CUDA_VISIBLE_DEVICES`, `--per_device_train_batch_size` and `--gradient_accumulation_steps` Multiply to get the total batch size.
`--neg_num --neg_context_ratio` Negative sampling rate to encourage generating none values proactively. [link](https://aclanthology.org/2021.emnlp-main.622.pdf)


run_qa_pretrain_t5.sh: (1) pretrain.  --percentage --evaluation_strategy --eval_steps --save_strategy --save_steps
(2) predict. 

### Evaluation
```
./run_qa_pretrain_t5.sh predict
```
`--history_turn` Previous turns used as the dialogue context for test.
`--per_device_eval_batch_size` The batch size for test.
`--test_type` *dst* for evaluating on the test set of MultiWOZ, or *qa* for evaluating on the QA dev set.
`--overwrite_cache` Whether or not use the cached DST test dataset.
To generate DST slot-values with the trained context-knowledge fusion model, run `./run_qa_pretrain_t5.sh predict_fusion`.



