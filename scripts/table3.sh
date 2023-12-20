cd ..

### Change this path ###
dataset_root="/media/data2/Booster/train/balanced"
########################

dataset_txt="datasets/booster/train_stereo.txt"

# RESULTS TABLE 3
for model in "midas_v21" "dpt_large"
do
    ## Ft. Virtual Depth (GT) MODEL ### 
    output_dir="results/Table3/Ft. Virtual Depth (GT)/"$model
    model_weights="weights/Table 3/Ft. Virtual Depth (GT)/"$model"_final.pt"
    python run.py --model_type $model \
                  --input_path $dataset_root \
                  --dataset_txt $dataset_txt \
                  --output_path "$output_dir" \
                  --model_weights "$model_weights"
    result_path="results/Table3_ftvirutaldepthgt_"$model".txt"
    python evaluate_mono.py --gt_root $dataset_root \
                            --pred_root "$output_dir" \
                            --dataset_txt $dataset_txt \
                            --output_path $result_path

    ## Ft. Virtual Depth (Proxy) MODEL - OUR ### 
    output_dir="results/Table3/Ft. Virtual Depth (Proxy)/"$model
    model_weights="weights/Table 3/Ft. Virtual Depth (Proxy)/"$model"_final.pt"
    python run.py --model_type $model \
                  --input_path $dataset_root \
                  --dataset_txt $dataset_txt \
                  --output_path "$output_dir" \
                  --model_weights "$model_weights"
    result_path="results/Table3_ftvirtualdepthproxy_"$model".txt"
    python evaluate_mono.py --gt_root $dataset_root \
                            --pred_root "$output_dir" \
                            --dataset_txt $dataset_txt \
                            --output_path $result_path
done