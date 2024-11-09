call conda activate maagent

call python eval_env.py --env simple --model_type random
call python eval_env.py --env simple --model_type MAPPO --load_model_path <weight_pth>
call python eval_env.py --env simple --model_type MAPPO --use_kg --kg_set 0 --load_model_path <weight_pth>
call python eval_env.py --env simple --model_type MAPPO --use_kg --kg_set 1 --load_model_path <weight_pth>
call python eval_env.py --env simple --model_type MAPPO --use_kg --kg_set 2 --load_model_path <weight_pth>
call python eval_env.py --env simple --model_type MAPPO --use_kg --kg_set 3 --load_model_path <weight_pth>

@REM call python eval_env.py --env simple --model_type COMA
@REM call python eval_env.py --env simple --model_type COMA --use_kg --kg_set 0 --load_model_path <weights_directory>
@REM call python eval_env.py --env simple --model_type COMA --use_kg --kg_set 1 --load_model_path <weights_directory>
@REM call python eval_env.py --env simple --model_type COMA --use_kg --kg_set 2 --load_model_path <weights_directory>
@REM call python eval_env.py --env simple --model_type COMA --use_kg --kg_set 3 --load_model_path <weights_directory>

call python eval_env.py --env simple --model_type expert --use_kg --kg_set 0
call python eval_env.py --env simple --model_type expert --use_kg --kg_set 1
call python eval_env.py --env simple --model_type expert --use_kg --kg_set 2
call python eval_env.py --env simple --model_type expert --use_kg --kg_set 3

call python visualize_stats.py --env simple

call python eval_env.py --env lava --model_type random
call python eval_env.py --env lava --model_type MAPPO --load_model_path
call python eval_env.py --env lava --model_type MAPPO --use_kg --load_model_path
@REM call python eval_env.py --env lava --model_type COMA --load_model_path <weights_directory>
@REM call python eval_env.py --env lava --model_type COMA --use_kg --load_model_path <weights_directory>
call python eval_env.py --env lava --model_type expert --use_kg

call python visualize_stats.py --env lava

call python eval_env.py --env key --model_type random
call python eval_env.py --env key --model_type MAPPO --load_model_path 
call python eval_env.py --env key --model_type MAPPO --use_kg --load_model_path
@REM call python eval_env.py --env key --model_type COMA --load_model_path <weights_directory>
@REM call python eval_env.py --env key --model_type COMA --use_kg --load_model_path <weights_directory>
call python eval_env.py --env key --model_type expert --use_kg

call python visualize_stats.py --env key

@REM call python eval_env.py --env lava2 --model_type random --load_model_path
@REM call python eval_env.py --env lava2 --model_type MAPPO --load_model_path
@REM call python eval_env.py --env lava2 --model_type MAPPO --use_kg --load_model_path
@REM call python eval_env.py --env lava2 --model_type COMA --load_model_path
@REM call python eval_env.py --env lava2 --model_type COMA --use_kg --load_model_path
@REM call python eval_env.py --env lava2 --model_type expert --use_kg
@REM call python visualize_stats.py --env lava2