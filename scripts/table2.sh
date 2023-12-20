cd ..

# RESULTS TABLE 2
for model in "midas_v21" "dpt_large"
do
    dataset_root="/media/data2/Booster/train/balanced"
    dataset_txt="datasets/booster/train_stereo.txt"

    ## BASE MODEL ###
    output_dir="results/Base/"$model
    python run.py --model_type $model \
                  --input_path $dataset_root \
                  --dataset_txt $dataset_txt \
                  --output_path $output_dir
    result_path="results/table2_base_"$model".txt"
    python evaluate_mono.py --gt_root $dataset_root \
                            --pred_root $output_dir \
                            --dataset_txt $dataset_txt \
                            --output_path $result_path

    ## FT. BASE MODEL ### 
    output_dir="results/Table2/Ft. Base/"$model
    model_weights="weights/Table 2/Ft. Base/"$model"_final.pt"
    python run.py --model_type $model \
                  --input_path $dataset_root \
                  --dataset_txt $dataset_txt \
                  --output_path "$output_dir" \
                  --model_weights "$model_weights"
    result_path="results/table2_ftbase_"$model".txt"
    python evaluate_mono.py --gt_root $dataset_root \
                            --pred_root "$output_dir" \
                            --dataset_txt $dataset_txt \
                            --output_path $result_path

    ## FT. VIRTUAL DEPTH MODEL - OUR ### 
    output_dir="results/Table2/Ft. Virtual Depth/"$model
    model_weights="weights/Table 2/Ft. Virtual Depth/"$model"_final.pt"
    python run.py --model_type $model \
                  --input_path $dataset_root \
                  --dataset_txt $dataset_txt \
                  --output_path "$output_dir" \
                  --model_weights "$model_weights"
    result_path="results/table2_ftvirtualdepth_"$model".txt"
    python evaluate_mono.py --gt_root $dataset_root \
                            --pred_root "$output_dir" \
                            --dataset_txt $dataset_txt \
                            --output_path $result_path
done