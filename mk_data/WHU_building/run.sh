BASE_MASKDIR=$(pwd)/data/WHU_building/train/clean_mask

python mk_data/WHU_building/add_omissionmoise.py --add_ratio 0.0 --remove_ratio 0.1 --marginnoise_prob 0.3 --base_maskdir ${BASE_MASKDIR}
python mk_data/WHU_building/add_omissionmoise.py --add_ratio 0.05 --remove_ratio 0.2 --marginnoise_prob 0.5 --base_maskdir ${BASE_MASKDIR}
python mk_data/WHU_building/add_omissionmoise.py --add_ratio 0.1 --remove_ratio 0.3 --marginnoise_prob 0.7 --base_maskdir ${BASE_MASKDIR}