for i in {0..8};
    do
    python normalize_txt.py --in_file ./data/pretrain/txt/abs_"$i".txt --out_norm_file ./data/pretrain/norm/"$i".txt
    done