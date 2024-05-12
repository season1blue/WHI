
# python3 utils/TrainInputProcess.py

#! /bin/bash

for i in 16
do
    j=$((3100/i))
    python3 ./Train.py \
    --epochs 150 \
    --save_steps $j \
    --dataset_type 2015 \
    --batch_size $i \
    --lr 1e-5 \
    --text_model_name "deberta" \
    --image_model_name "clip" \
    --output_dir /data/results \
    --output_result_file /data/result.txt \
    --log_dir log.log \
    --device_id "cuda:0" \
    --enable_log \
    --only_text_loss \
    --add_gan \
    # --add_gan_loss
    # --alpha 0 \
    # --beta 0 \
    # --add_llm \
done

