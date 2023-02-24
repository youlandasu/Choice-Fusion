#!/bin/bash
MODE=$1
if [ "$MODE" = "pretrain" ]; then
    echo "Pre-train T5 with QA data" 
    declare -a lrs=("3e-4")
    declare -a warms=("100")
    for index in ${!lrs[*]}; 
    do 
        lr=${lrs[$index]}
        warm=${warms[$index]}
        echo "QA Training - $lr - $warm"
        DATA_DIR=qa_data/preprocessed
        DST_DATA_DIR=data
        CACHED_DATA_DIR=cached_data_20pct
        mkdir -p $CACHED_DATA_DIR 
        MODEL_DIR=saved_models_t5_pretrained/kld_t5_20pct
        CUDA_VISIBLE_DEVICES=0,1
        python run_dst.py \
            --task_name vadst \
            --model_name_or_path t5-small \
            --mode "$MODE" \
            --do_train \
            --do_eval \
            --seed 42 --disable_tqdm False\
            --cached_data_dir $CACHED_DATA_DIR \
            --train_file ${DATA_DIR}/train.json \
            --validation_file ${DATA_DIR}/dev.json \
            --test_file ${DST_DATA_DIR}/test_dials.json \
            --ontology_file ${DST_DATA_DIR}/ontology.json \
            --description_file utils/slot_description.json \
            --neg_num 0.4 --neg_context_ratio 0.05 \
            --value_distribution True \
            --percentage 20 \
            --knowledge_fusion initDecoder \
            --word_bow_loss 0.5 \
            --evaluation_strategy epoch \
	        --save_strategy epoch \
            --load_best_model_at_end \
            --save_total_limit 10 \
            --metric_for_best_model eval_loss \
            --greater_is_better False \
            --logging_steps 999999 \
            --warmup_steps ${warm} \
            --learning_rate ${lr} \
            --num_train_epochs 6 \
            --max_seq_length 512 \
            --output_dir ${MODEL_DIR} \
            --per_device_eval_batch_size 32 \
            --per_device_train_batch_size 4 \
            --gradient_accumulation_steps 32 \
            --fp16 --overwrite_output \
            --prediction_output "" \
            --overwrite_cache  
    done
elif [ "$MODE" = "pretrain_fusion" ]; then
    echo "Pre-train T5 with QA data" 
    declare -a lrs=("3e-4")
    declare -a warms=("100")
    for index in ${!lrs[*]}; 
    do 
        lr=${lrs[$index]}
        warm=${warms[$index]}
        echo "QA Training - $lr - $warm"
        DATA_DIR=qa_data/preprocessed
        DST_DATA_DIR=data
        CACHED_DATA_DIR=cached_data_20pct
        mkdir -p $CACHED_DATA_DIR 
        MODEL_DIR=saved_models_t5_pretrained/kld_t5_20pct
        CUDA_VISIBLE_DEVICES=0,1
        python run_dst_fusion.py \
            --task_name vadst \
            --model_name_or_path t5-small \
            --mode "$MODE" \
            --do_train \
            --do_eval \
            --seed 42 --disable_tqdm False\
            --cached_data_dir $CACHED_DATA_DIR \
            --train_file ${DATA_DIR}/train.json \
            --validation_file ${DATA_DIR}/dev.json \
            --test_file ${DST_DATA_DIR}/test_dials.json \
            --ontology_file ${DST_DATA_DIR}/ontology.json \
            --description_file utils/slot_description.json \
            --neg_num 0.4 --neg_context_ratio 0.05 \
            --value_distribution True \
            --percentage 20 \
            --knowledge_fusion initDecoder \
            --word_bow_loss 0.5 \
            --evaluation_strategy epoch \
	        --save_strategy epoch \
            --load_best_model_at_end \
            --save_total_limit 10 \
            --metric_for_best_model eval_loss \
            --greater_is_better False \
            --logging_steps 999999 \
            --warmup_steps ${warm} \
            --learning_rate ${lr} \
            --num_train_epochs 6 \
            --max_seq_length 512 \
            --output_dir ${MODEL_DIR} \
            --per_device_eval_batch_size 32 \
            --per_device_train_batch_size 4 \
            --gradient_accumulation_steps 32 \
            --fp16 --overwrite_output \
            --prediction_output "" \
            --overwrite_cache  
    done
elif [ "$MODE" = "predict" ]; then
    echo "Pre-train the xlmrb with Parallel examples" 
    declare -a lrs=("1e-4")
    declare -a warms=("100")
    for index in ${!lrs[*]}; 
    do 
        lr=${lrs[$index]}
        warm=${warms[$index]}
        echo "QA Training - $lr - $warm"
        DATA_DIR=qa_data/preprocessed
        DST_DATA_DIR=data
        CACHED_DATA_DIR=cached_data_20pct
        mkdir -p $CACHED_DATA_DIR 
        MODEL_DIR=saved_models_t5_pretrained/kld_t5_20pct
        CUDA_VISIBLE_DEVICES=0
        python run_dst.py \
            --task_name vadst \
            --model_name_or_path ${MODEL_DIR} \
            --mode "$MODE" \
            --do_predict \
            --seed 42 --disable_tqdm False\
            --cached_data_dir $CACHED_DATA_DIR \
            --train_file ${DATA_DIR}/train.json \
            --validation_file ${DATA_DIR}/dev.json \
            --test_file ${DST_DATA_DIR}/test_dials.json \
            --ontology_file ${DST_DATA_DIR}/ontology.json \
            --description_file utils/slot_description.json \
            --neg_num 0.0 --neg_context_ratio 0.00 \
            --value_distribution True \
            --percentage 20 \
            --knowledge_fusion initDecoder \
            --history_turn 8 \
            --evaluation_strategy no \
            --save_total_limit 5 \
            --metric_for_best_model eval_loss \
            --greater_is_better False \
            --logging_steps 100 \
            --max_seq_length 512 \
            --output_dir ${MODEL_DIR} \
            --per_device_eval_batch_size 32 \
            --per_device_train_batch_size 4 \
            --gradient_accumulation_steps 128 \
            --fp16 --overwrite_output \
            --prediction_output results_64 \
            --test_type dst \
            --overwrite_cache  
    done
elif [ "$MODE" = "predict_fusion" ]; then
    echo "Pre-train the xlmrb with Parallel examples" 
    declare -a lrs=("1e-4")
    declare -a warms=("100")
    for index in ${!lrs[*]}; 
    do 
        lr=${lrs[$index]}
        warm=${warms[$index]}
        echo "QA Training - $lr - $warm"
        DATA_DIR=qa_data/preprocessed
        DST_DATA_DIR=data
        CACHED_DATA_DIR=cached_data_20pct
        mkdir -p $CACHED_DATA_DIR 
        MODEL_DIR=saved_models_t5_pretrained/kld_t5_20pct
        CUDA_VISIBLE_DEVICES=0
        python run_dst_fusion.py \
            --task_name vadst \
            --model_name_or_path ${MODEL_DIR} \
            --mode "$MODE" \
            --do_predict \
            --seed 42 --disable_tqdm False\
            --cached_data_dir $CACHED_DATA_DIR \
            --train_file ${DATA_DIR}/train.json \
            --validation_file ${DATA_DIR}/dev.json \
            --test_file ${DST_DATA_DIR}/test_dials.json \
            --ontology_file ${DST_DATA_DIR}/ontology.json \
            --description_file utils/slot_description.json \
            --neg_num 0.0 --neg_context_ratio 0.00 \
            --value_distribution True \
            --percentage 20 \
            --knowledge_fusion initDecoder \
            --history_turn 8 \
            --evaluation_strategy no \
            --save_total_limit 5 \
            --metric_for_best_model eval_loss \
            --greater_is_better False \
            --logging_steps 100 \
            --max_seq_length 512 \
            --output_dir ${MODEL_DIR} \
            --per_device_eval_batch_size 32 \
            --per_device_train_batch_size 4 \
            --gradient_accumulation_steps 128 \
            --fp16 --overwrite_output \
            --prediction_output results_64 \
            --test_type dst \
            --overwrite_cache  
    done
else
    echo "Wrong Mode"
fi
