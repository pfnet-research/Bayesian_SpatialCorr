### ver001: effect of ρ for JSRT dataset

BASE_SAVEDIR=$(pwd)/results/
BASE_DATADIR=$(pwd)/data/

exp='002'
filename=$(basename "$0")
ver=$(echo "$filename" | sed -E 's/[^0-9]*([0-9]+)\.sh/\1/')

cd experiment/src
noise_type0='clean_mask'
noise_type1='label_noise_0.3_0.5'
noise_type2='label_noise_0.5_0.7'
noise_type3='label_noise_0.7_0.7'

loss='bayesian_spatialcorr' 
dataset_names=(jsrt_lung jsrt_heart jsrt_clavicle_c)
noise_types=(${noise_type2})
seeds=(0)
model=efficientnet
bs_is_lr_epoch_comb=(
  "efficientnet 1.0 256 16 200 0.001 10 0.0 0.1 -5 -2 1 1 default True True"
  "efficientnet 1.0 256 16 200 0.001 10 0.1 0.1 -5 -2 1 1 default True True"
  "efficientnet 1.0 256 16 200 0.001 10 0.3 0.1 -5 -2 1 1 default True True"
  "efficientnet 1.0 256 16 200 0.001 10 0.5 0.1 -5 -2 1 1 default True True"
  "efficientnet 1.0 256 16 200 0.001 10 0.75 0.1 -5 -2 1 1 default True True"
  "efficientnet 1.0 256 16 200 0.001 10 0.99 0.1 -5 -2 1 1 default True True"
  )

for dataset_name in "${dataset_names[@]}"; do 
  for noise_type in "${noise_types[@]}"; do 
    for seed in "${seeds[@]}"; do 
      for combination in "${bs_is_lr_epoch_comb[@]}"; do
        IFS=' ' read -r model downsample_ratio imsize batch_size epoch lr estep rho imgwise_lr m_init mu_init sigma_init gamma_init augment_type geo_aug grad_clip <<< "$combination"
        
        torchrun --nproc-per-node=1 main_ddp.py distributed=True \
                                                amp=True \
                                                loss=${loss} \
                                                data=${dataset_name} \
                                                expname=exp${exp} \
                                                vername=ver${ver} \
                                                model=${model} \
                                                data.noise_type=${noise_type} \
                                                data.augment_type=${augment_type} \
                                                data.data_size.downsample_ratio=${downsample_ratio} \
                                                data.imsize=${imsize} \
                                                data.geo_aug=${geo_aug} \
                                                train.epoch=${epoch} \
                                                train.batch_size=${batch_size} \
                                                train.max_lr=${lr} \
                                                train.seed=${seed} \
                                                val.per_epoch=5 \
                                                loss.rhosigma_init=${rho} \
                                                loss.rhogamma_init=${rho} \
                                                loss.mu_init=${mu_init} \
                                                loss.sigma_init=${sigma_init} \
                                                loss.m_init=${m_init} \
                                                loss.gamma_init=${gamma_init} \
                                                loss.estep=${estep} \
                                                loss.imgwise_lr=${imgwise_lr} \
                                                loss.imgwise_wd=0.0 \
                                                loss.rho_lr=0.0 \
                                                loss.mu_lr=0.0 \
                                                loss.sigma_lr=0.0 \
                                                loss.trainable=False \
                                                utils.base_savedir=${BASE_SAVEDIR} \
                                                utils.base_datadir=${BASE_DATADIR} \
                                                utils.base_savename='${expname}/${vername}/${data.dataset_name}/downsample_ratio${data.data_size.downsample_ratio}/${data.noise_type}/${loss.name}_rho${loss.rhosigma_init}_estep${loss.estep}_trainable${loss.trainable}_imgwiseLR${loss.imgwise_lr}_m${loss.m_init}_mu${loss.mu_init}_sigma${loss.sigma_init}_gamma${loss.gamma_init}/seed${train.seed}'

      done
    done
  done
done
