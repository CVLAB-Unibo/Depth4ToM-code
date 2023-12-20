root="path_to_dataset_root"
cd ..

model="dpt_large" # ["midas_v21", "dpt_large"]
dataset="Trans10K" # ["Trans10K", "MSD"]
splits="train test validation"
for split in $splits
do
    echo $model $dataset $split
    input_dir=$root/$dataset/$split/images # path to dataset folder with images
    mask_dir=$root/$dataset/$split/masks # path to dataset folder with segmentations, either GT or proxy
    output_dir=$root"/"$dataset/$split/$model"_proxies"/$exp # output path
    
    dataset_lower=$(echo $dataset | tr '[:upper:]' '[:lower:]')
    dataset_txt="datasets/"$dataset_lower"/"$split".txt" # inference list

    ### define output_list if you want to save the list of the generated virtual depths
    exp="base"
    output_list="datasets/"$dataset_lower"/"$split"_"$model"_"$exp".txt"
    ###
    
    if [ -f $dataset_txt ]
    then
        python run.py --model_type $model \
                    --input_path $input_dir \
                    --dataset_txt $dataset_txt \
                    --output_path $output_dir \
                    --output_list $output_list \
                    --mask_path $mask_dir \
                    --it 5 \
                    --cls2mask 255 # list of class ids in segmentation maps relative to ToM surfaces.
    fi
done