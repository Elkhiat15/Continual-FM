
models=('PanDerm' 'SwAVDerm'  'dinov2' 'imgnet_large21k')
#models=('imgnet_large21k')

for model in "${models[@]}"; do
  csv_file="${models}_Result.csv"
  for seed in 0; do
    CUDA_VISIBLE_DEVICES=1 python linear_eval.py \
      --batch_size 1000 \
      --model $model \
      --nb_classes 7 \
      --percent_data 1.0 \
      --csv_filename $csv_file \
      --output_dir "/home/share/FM_Code/FM_Eval/LP_Eval/output_dir2/ID_Res/"$model"_res/" \
      --csv_path "/home/share/Uni_Eval/HAM10000_clean/ISIC2018_splits/HAM_clean.csv" \
      --root_path "/home/share/Uni_Eval/HAM10000_clean/ISIC2018/"
    CUDA_VISIBLE_DEVICES=0 python linear_eval.py \
        --batch_size 1000 \
        --model $model \
        --nb_classes 16 \
        --percent_data 1 \
        --csv_filename $csv_file \
        --output_dir "/home/share/FM_Code/FM_Eval/LP_Eval/output_dir2/ID_Res/"$model"_res/" \
        --csv_path '/home/share/Uni_Eval/WSI_patch_data/patch.csv' \
        --root_path '/home/share/Uni_Eval/WSI_patch_data/images/'
    CUDA_VISIBLE_DEVICES=0 python linear_eval.py \
        --batch_size 1000 \
        --model $model \
        --nb_classes 2 \
        --percent_data 1.0 \
        --csv_filename $csv_file \
        --output_dir "/home/share/FM_Code/FM_Eval/LP_Eval/output_dir2/ID_Res/"$model"_res/" \
        --csv_path '/home/share/Uni_Eval/MSKCC/mskcc.csv' \
        --root_path '/home/share/Uni_Eval/MSKCC/images/'
    CUDA_VISIBLE_DEVICES=1 python linear_eval.py \
      --batch_size 1000 \
      --model $model \
      --nb_classes 9 \
      --percent_data 1.0 \
      --csv_filename $csv_file \
      --output_dir "/home/share/FM_Code/FM_Eval/LP_Eval/output_dir2/ID_Res/"$model"_res/" \
      --csv_path '/home/share/Uni_Eval/BCN20000/bcn20000.csv' \
      --root_path '/home/share/Uni_Eval/BCN20000/images/'

    CUDA_VISIBLE_DEVICES=2 python linear_eval.py \
      --batch_size 1000 \
      --model $model \
      --nb_classes 2 \
      --percent_data 1.0 \
      --csv_filename $csv_file \
      --output_dir "/home/share/FM_Code/FM_Eval/LP_Eval/output_dir2/ID_Res/"$model"_res/" \
      --csv_path '/home/share/Uni_Eval/HIBA/hiba.csv' \
      --root_path '/home/share/Uni_Eval/HIBA/images/'

    CUDA_VISIBLE_DEVICES=2 python linear_eval.py \
      --batch_size 1000 \
      --model $model \
      --nb_classes 6 \
      --percent_data 1.0 \
      --csv_filename $csv_file \
      --output_dir "/home/share/FM_Code/FM_Eval/LP_Eval/output_dir2/ID_Res/"$model"_res/" \
      --csv_path "/home/share/Uni_Eval/pad-ufes/2000.csv" \
      --root_path "/home/share/Uni_Eval/pad-ufes/images/"
    CUDA_VISIBLE_DEVICES=3 python linear_eval.py \
      --batch_size 1000 \
      --model $model \
      --nb_classes 2 \
      --percent_data 1.0 \
      --csv_filename $csv_file \
      --output_dir "/home/share/FM_Code/FM_Eval/LP_Eval/output_dir2/ID_Res/"$model"_res/" \
      --csv_path "/home/share/Uni_Eval/Derm7pt/atlas-clinical-all.csv" \
      --root_path "/home/share/Uni_Eval/Derm7pt/images/"
    CUDA_VISIBLE_DEVICES=3 python linear_eval.py \
      --batch_size 1000 \
      --model $model \
      --nb_classes 2 \
      --percent_data 1.0 \
      --csv_filename $csv_file \
      --output_dir "/home/share/FM_Code/FM_Eval/LP_Eval/output_dir2/ID_Res/"$model"_res/" \
      --csv_path '/home/share/Uni_Eval/DDI/ddi_clean.csv' \
      --root_path '/home/share/Uni_Eval/DDI/images/'

      CUDA_VISIBLE_DEVICES=1 python linear_eval.py \
        --batch_size 1000 \
        --model $model \
        --nb_classes 23 \
        --percent_data 1.0 \
        --csv_filename $csv_file \
        --output_dir "/home/share/FM_Code/FM_Eval/LP_Eval/output_dir2/task2/"$model"_res/" \
        --csv_path "/home/share/Uni_Eval/Dermnet/dermnet.csv" \
        --root_path "/home/share/Uni_Eval/Dermnet/images/"

  done
  wait
done

