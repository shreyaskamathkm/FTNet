sbatch --job-name='FTN50' --gres=gpu:v100:2 --export=BACKBONE='resnext50_32x4d' RUN_me.sh
sbatch --job-name='FTN101' --gres=gpu:v100:2 --export=BACKBONE='resnext101_32x8d' RUN_me.sh

sbatch --job-name='RX50_20' --gres=gpu:v100:2 --export=BACKBONE='resnext50_32x4d',ALPHA='20' RUN_me_AlphaLoss.sh
sbatch --job-name='RX50_10' --gres=gpu:v100:2 --export=BACKBONE='resnext50_32x4d',ALPHA='10' RUN_me_AlphaLoss.sh
sbatch --job-name='RX50_30' --gres=gpu:v100:2 --export=BACKBONE='resnext50_32x4d',ALPHA='30' RUN_me_AlphaLoss.sh
sbatch --job-name='RX50_15' --gres=gpu:v100:2 --export=BACKBONE='resnext50_32x4d',ALPHA='15' RUN_me_AlphaLoss.sh
sbatch --job-name='RX50_5' --gres=gpu:v100:2 --export=BACKBONE='resnext50_32x4d',ALPHA='5' RUN_me_AlphaLoss.sh
