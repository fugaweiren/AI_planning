conda activate project

python eval_env.py --env simple --model_type random
python eval_env.py --env simple --model_type MAPPO --load_model_path <weight_pth>;
python eval_env.py --env simple --model_type MAPPO --use_kg --kg_set 0 --load_model_path <weight_pth>;
python eval_env.py --env simple --model_type MAPPO --use_kg --kg_set 1 --load_model_path <weight_pth>;
python eval_env.py --env simple --model_type MAPPO --use_kg --kg_set 2 --load_model_path <weight_pth>;
python eval_env.py --env simple --model_type MAPPO --use_kg --kg_set 3 --load_model_path <weight_pth>;


python eval_env.py --env simple --model_type COMA;
python eval_env.py --env simple --model_type COMA --use_kg --kg_set 0 --load_model_path <weights_directory>;
python eval_env.py --env simple --model_type COMA --use_kg --kg_set 1 --load_model_path <weights_directory>;
python eval_env.py --env simple --model_type COMA --use_kg --kg_set 2 --load_model_path <weights_directory>;
python eval_env.py --env simple --model_type COMA --use_kg --kg_set 3 --load_model_path <weights_directory>;

python eval_env.py --env simple --model_type expert --use_kg --kg_set 0;
python eval_env.py --env simple --model_type expert --use_kg --kg_set 1;
python eval_env.py --env simple --model_type expert --use_kg --kg_set 2;
python eval_env.py --env simple --model_type expert --use_kg --kg_set 3;

python visualize_stats.py --env simple;

python eval_env.py --env lava --model_type random;
python eval_env.py --env lava --model_type MAPPO --load_model_path <weight_pth>;
python eval_env.py --env lava --model_type MAPPO --use_kg --load_model_path <weight_pth>;
python eval_env.py --env lava --model_type COMA --load_model_path <weights_directory>;
python eval_env.py --env lava --model_type COMA --use_kg --load_model_path <weights_directory>;
python eval_env.py --env lava --model_type expert --use_kg;

python visualize_stats.py --env lava;

python eval_env.py --env key --model_type random --load_model_path <weight_pth>;
python eval_env.py --env key --model_type MAPPO --load_model_path <weight_pth>;
python eval_env.py --env key --model_type MAPPO --use_kg --load_model_path <weight_pth>;
python eval_env.py --env key --model_type COMA --load_model_path <weights_directory>;
python eval_env.py --env key --model_type COMA --use_kg --load_model_path <weights_directory>;
python eval_env.py --env key --model_type expert --use_kg;

python visualize_stats.py --env key; 

# python eval_env.py --env lava2 --model_type random;
# python eval_env.py --env lava2 --model_type MAPPO --load_model_path
# python eval_env.py --env lava2 --model_type MAPPO --use_kg --load_model_path
# python eval_env.py --env lava2 --model_type COMA --load_model_path
# python eval_env.py --env lava2 --model_type COMA --use_kg --load_model_path
# python eval_env.py --env lava2 --model_type expert --use_kg;
# python visualize_stats.py --env lava2; 