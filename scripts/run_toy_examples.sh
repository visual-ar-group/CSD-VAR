
PRETRAINED_MODEL_PATH="yresearch/Switti-AR"
USE_AR="True"

# specify the path to the benchmark prompts switch to enable test full benchmark_data/benchmark_prompts.json
BENCH_PROMPT_PATH="benchmark_data/benchmark_prompts_quick.json"
CONFIG="config/config_group_scales.yaml"

TRAIN_DATA_DIR="benchmark_data/toy_examples" # specify the path to the training data
OUTPUT_EVAL_DIR="output/toy_examples" # specify the path to the output directory


for TRAIN_DATA in "$TRAIN_DATA_DIR"/*; do
    TRAIN_DATA_FILENAME=$(basename "$TRAIN_DATA")

    # Split filename using '+' as a delimiter to content inital tokens and style inital tokens
    IFS='+' read -r TRAIN_DATA_OBJECT TRAIN_DATA_STYLE <<< "$TRAIN_DATA_FILENAME"

    echo "Processing: $TRAIN_DATA_FILENAME"

    SCALE=3 # specify the end scale for the first group 
    
    # Read the name field from the YAML config file
    CONFIG_NAME=$(grep "^name:" "$CONFIG" | cut -d ":" -f2 | tr -d " ")
    echo "CONFIG_NAME: $CONFIG_NAME"

    # Second run with disabled orthogonal loss
    LOCAL_OUTPUT_DIR="exps/$TRAIN_DATA_FILENAME/local_output_${TRAIN_DATA_FILENAME}_${CONFIG_NAME}"

    # specify the path to the proj and proj_2 files
    PROJ_PATH="$TRAIN_DATA/proj_top_10.pt"
    PROJ_2_PATH="$TRAIN_DATA/proj_2_top_10.pt"


    # [Train]
    python -m torch.distributed.run --standalone --master-addr=0.0.0.0 --nproc_per_node=1 train.py \
        --pretrained_path="$PRETRAINED_MODEL_PATH" \
        --use_fsdp=True \
        --grad_accum=1 \
        --dataset_repeats=100 \
        --pn=512 \
        --depth=30 \
        --rope_size=128 \
        --rope_theta=10000 \
        --use_ar="$USE_AR" \
        --use_crop_cond=True \
        --use_swiglu_ffn=True \
        --data_path="$TRAIN_DATA" \
        --vae_ckpt="yresearch/VQVAE-Switti" \
        --max_iters=200 \
        --bs=1 \
        --eval_batch_size=16 \
        --log_iters=50 \
        --log_images_iters=50 \
        --save_iters=50 \
        --global_save_iters=50 \
        --fp16=2 \
        --alng=1e-3 \
        --tblr=1e-2 \
        --vfast=1 \
        --wp=50 \
        --twd=0.05 \
        --placeholder_token_content="<object>" \
        --initializer_token_content="$TRAIN_DATA_OBJECT" \
        --placeholder_token_style="<style>" \
        --initializer_token_style="$TRAIN_DATA_STYLE" \
        --use_captions=True \
        --num_vectors_content=4 \
        --num_vectors_style=4 \
        --major_scale_end="$SCALE" \
        --major_type_scale="style" \
        --enable_orthogonal_loss=False \
        --local_out_dir_path="$LOCAL_OUTPUT_DIR" \
        --minor_attend_weight=0.1 \
        --prompt_config_path="$CONFIG" \
        --proj_path="$PROJ_PATH" \
        --proj_2_path="$PROJ_2_PATH"

    # [Eval]
    echo "Evaluating: $LOCAL_OUTPUT_DIR"
    STEPS=(150) # specify the step ckpt for evaluation
    for STEP in "${STEPS[@]}"; do
        
        python inference.py \
            --prompts_path="$BENCH_PROMPT_PATH" \
            --step=$STEP \
            --content_path_ckpt="$LOCAL_OUTPUT_DIR" \
            --style_path_ckpt="$LOCAL_OUTPUT_DIR" \
            --content_name="$TRAIN_DATA_OBJECT" \
            --style_name="$TRAIN_DATA_STYLE" \
            --output_dir="$OUTPUT_EVAL_DIR" \
            --captions_path="$TRAIN_DATA/captions.csv" \
            --style_scale_end=$SCALE \
            --images_per_prompt=10 \
            --model_path="$PRETRAINED_MODEL_PATH" \
            --prompt_config_path="$CONFIG" \
            --proj_path="$PROJ_PATH" \
            --proj_2_path="$PROJ_2_PATH" \
            --save_grid
        
    done

done

python evaluate_with_metrics.py --result_dir "$OUTPUT_EVAL_DIR" --data_dir "$TRAIN_DATA_DIR"