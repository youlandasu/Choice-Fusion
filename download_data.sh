# Copyright (c) Facebook, Inc. and its affiliates
# All rights reserved.

# mostly from https://github.com/mrqa/MRQA-Shared-Task-2019

set -e
# mrqa train
curl https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/SQuAD.jsonl.gz --create-dirs -o qa_data/mrqa_train/SQuAD.jsonl.gz
curl https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/NewsQA.jsonl.gz --create-dirs -o qa_data/mrqa_train/NewsQA.jsonl.gz
curl https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/TriviaQA-web.jsonl.gz --create-dirs -o qa_data/mrqa_train/TriviaQA.jsonl.gz
curl https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/SearchQA.jsonl.gz --create-dirs -o qa_data/mrqa_train/SearchQA.jsonl.gz
curl https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/HotpotQA.jsonl.gz --create-dirs -o qa_data/mrqa_train/HotpotQA.jsonl.gz
curl https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/NaturalQuestionsShort.jsonl.gz --create-dirs -o qa_data/mrqa_train/NaturalQuestions.jsonl.gz
# mrqa valid
curl https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/SQuAD.jsonl.gz --create-dirs -o qa_data/mrqa_valid/SQuAD.jsonl.gz
curl https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/NewsQA.jsonl.gz --create-dirs -o qa_data/mrqa_valid/NewsQA.jsonl.gz
curl https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/TriviaQA-web.jsonl.gz --create-dirs -o qa_data/mrqa_valid/TriviaQA.jsonl.gz
curl https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/SearchQA.jsonl.gz --create-dirs -o qa_data/mrqa_valid/SearchQA.jsonl.gz
curl https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/HotpotQA.jsonl.gz --create-dirs -o qa_data/mrqa_valid/HotpotQA.jsonl.gz
curl https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/NaturalQuestionsShort.jsonl.gz --create-dirs -o qa_data/mrqa_valid/NaturalQuestions.jsonl.gz
# squad2
curl https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json --create-dirs -o qa_data/squad2/train-v2.0.json
curl https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json --create-dirs -o qa_data/squad2/dev-v2.0.json
# dream
curl https://github.com/nlpdata/dream/raw/master/data/train.json --create-dirs -o qa_data/dream/train.json
curl https://github.com/nlpdata/dream/raw/master/data/dev.json --create-dirs -o qa_data/dream/dev.json
curl https://github.com/nlpdata/dream/raw/master/data/test.json --create-dirs -o qa_data/dream/test.json
