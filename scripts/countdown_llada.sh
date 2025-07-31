export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true


for seed in {0..19}
do
    output_path="./outputs/countdown/llada_conf/genlen-32_T-32_blocksize-32_seed-${seed}"

    accelerate launch eval_llada.py \
        --seed $seed \
        --tasks countdown3 \
        --model llada_dist \
        --limit 256 \
        --confirm_run_unsafe_code \
        --output_path $output_path \
        --model_args model_path='GSAI-ML/LLaDA-8B-Instruct',mask_length=32,sampling_steps=32,block_size=32,sampler='llada_conf'

    cp ./*.json ./outputs/countdown/llada_conf/genlen-32_T-32_blocksize-32_seed-${seed}/
    rm ./*.json
done