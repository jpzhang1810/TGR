echo "The adv_path: $1"
python evaluate.py --adv_path $1 --model_name vit_base_patch16_224 --batch_size 10 &
python evaluate.py --adv_path $1 --model_name deit_base_distilled_patch16_224 --batch_size 10 &
python evaluate.py --adv_path $1 --model_name levit_256 --batch_size 10 &
python evaluate.py --adv_path $1 --model_name pit_b_224 --batch_size 10 
python evaluate.py --adv_path $1 --model_name cait_s24_224 --batch_size 10 &
python evaluate.py --adv_path $1 --model_name convit_base --batch_size 10 &
python evaluate.py --adv_path $1 --model_name tnt_s_patch16_224 --batch_size 10 &
python evaluate.py --adv_path $1 --model_name visformer_small --batch_size 10
wait