export TRAIN_DIR=aws-tensorflow-benchmarking/albert/tfrecords/train/max_seq_len_512_max_predictions_per_seq_20_masked_lm_prob_15
export VAL_DIR=aws-tensorflow-benchmarking/albert/tfrecords/validation/max_seq_len_512_max_predictions_per_seq_20_masked_lm_prob_15
export LOG_DIR=logs/albert
export CHECKPOINT_DIR=checkpoints/albert

export SAGEMAKER_ROLE=arn:aws:iam::564829616587:role/SageMakerRole
export SAGEMAKER_IMAGE_NAME=564829616587.dkr.ecr.us-east-1.amazonaws.com/nieljare:py37_tf240_hf420
export SAGEMAKER_FSX_ID=fs-03c56af4a3db22887
export SAGEMAKER_FSX_MOUNT_NAME=lml2hbmv
export SAGEMAKER_SUBNET_IDS=subnet-06201e2a536b5eea1
export SAGEMAKER_SECURITY_GROUP_IDS=sg-0caa0756da3773032

python3 -m albert.launch_sagemaker \
    --source_dir=. \
    --entry_point=albert/run_pretraining.py \
    --sm_job_name=albert-pretrain \
    --instance_type=ml.p3dn.24xlarge \
    --instance_count=1 \
    --train_dir=${TRAIN_DIR} \
    --val_dir=${VAL_DIR} \
    --log_dir=${LOG_DIR} \
    --checkpoint_dir=${CHECKPOINT_DIR} \
    --load_from=scratch \
    --model_type=albert \
    --model_size=base \
    --per_gpu_batch_size=32 \
    --gradient_accumulation_steps=2 \
    --warmup_steps=3125 \
    --total_steps=125000 \
    --learning_rate=0.00176 \
    --optimizer=lamb \
    --log_frequency=10 \
    --name=myfirstjob
