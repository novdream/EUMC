PYTHON_SCRIPT_PATH="EUMC/train.py"
dataset="Cora"
prompt_sizes=5
num_prompt=300
homo_boost_thrd=0.5
hidden=64
id=0
trojan_epochs=600
val_feq=50
evaluate_mode=1by1
selection_method=cluster_degree
defense_mode="none"
homo_loss_weight=10.0
norm_weight=0.0001
dis_weight=0.0
layer=2
seed=106
train_lr=0.005
new="Y"
index=0
num_attach=100
fit_attach_num=100
test_thr=0.1
prune_thr=0.1


python $PYTHON_SCRIPT_PATH --dataset $dataset --prompt_size $prompt_size  --num_prompts $num_prompt \
    --homo_boost_thrd $homo_boost_thrd --hidden $hidden --num_attach $num_attach \
    --norm_weight $norm_weight --layer $layer --trojan_epochs $trojan_epochs\
    --device_id $id --evaluate_mode $evaluate_mode --homo_loss_weight $homo_loss_weight\
    --dis_weight $dis_weight --selection_method $selection_method --seed $seed \
    --prune_thr $prune_thr --defense_mode $defense_mode --index $index --fit_attach_num $fit_attach_num\
    --new $new --train_lr $train_lr --test_thr $test_thr --val_feq $val_feq 
