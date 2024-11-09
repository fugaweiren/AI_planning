conda activate multiagent

python eval_env.py --env simple --model_type random
# python eval_env.py --env simple --model_type MAPPO --load_model_path <weight_pth>;
# python eval_env.py --env simple --model_type MAPPO --use_kg --kg_set 0 --load_model_path <weight_pth>;
# python eval_env.py --env simple --model_type MAPPO --use_kg --kg_set 1 --load_model_path <weight_pth>;
python eval_env.py --env simple --model_type MAPPO --use_kg --kg_set 2 --load_model_path '/home/senyu/AI_planning/gym-multigrid-master/results/mappo/simple/steps_5000000_ngames_39062_USE_KG_conflicting rules/mappo_agent_model.pth';
python eval_env.py --env simple --model_type MAPPO --use_kg --kg_set 3 --load_model_path '/home/senyu/AI_planning/gym-multigrid-master/results/mappo/simple/steps_5000000_ngames_39062_USE_KG_irrelevant rules/mappo_agent_model.pth';


# python eval_env.py --env simple --model_type COMA --load_model_path ;
# python eval_env.py --env simple --model_type COMA --use_kg --kg_set 0 --load_model_path '/Users/celine/Desktop/3A/CS5446/Project_Git/AI_planning/gym-multigrid-master/results/coma/simple/steps_1500000_ngames_32363_USE_KG_ball only/model_weights_ep30k';
# python eval_env.py --env simple --model_type COMA --use_kg --kg_set 1 --load_model_path '/Users/celine/Desktop/3A/CS5446/Project_Git/AI_planning/gym-multigrid-masterg';
# python eval_env.py --env simple --model_type COMA --use_kg --kg_set 2 --load_model_path '/Users/celine/Desktop/3A/CS5446/Project_Git/AI_planning/gym-multigrid-master/results/coma/simple/steps_1500000_ngames_33875_USE_KG_conflicting rules/model_weights_save_dir';
# python eval_env.py --env simple --model_type COMA --use_kg --kg_set 3 --load_model_path '/Users/celine/Desktop/3A/CS5446/Project_Git/AI_planning/gym-multigrid-master/results/coma/simple/steps_1500000_ngames_29379_USE_KG_irrelevant rules/model_weights_save_dir';

python eval_env.py --env simple --model_type expert --use_kg --kg_set 0;
python eval_env.py --env simple --model_type expert --use_kg --kg_set 1;
python eval_env.py --env simple --model_type expert --use_kg --kg_set 2;
python eval_env.py --env simple --model_type expert --use_kg --kg_set 3;

#python visualize_stats.py --env simple;

python eval_env.py --env lava --model_type random;
python eval_env.py --env lava --model_type MAPPO --load_model_path '/home/senyu/AI_planning/gym-multigrid-master/results/mappo/lava/steps_5000000_ngames_39062/mappo_agent_model.pth';
python eval_env.py --env lava --model_type MAPPO --use_kg --load_model_path '/home/senyu/AI_planning/gym-multigrid-master/results/mappo/lava/steps_5000000_ngames_39062_USE_KG_lava + ball/mappo_agent_model.pth';

# python eval_env.py --env lava --model_type COMA --load_model_path '/Users/celine/Desktop/3A/CS5446/Project_Git/AI_planning/gym-multigrid-master/results/coma/lava/steps_1500000_ngames_13999/model_weights_save_dir';
# python eval_env.py --env lava --model_type COMA --use_kg --load_model_path '/Users/celine/Desktop/3A/CS5446/Project_Git/AI_planning/gym-multigrid-master/results/coma/lava/steps_1500000_ngames_32366_USE_KG_lava + ball/model_weights_save_dir';

python eval_env.py --env lava --model_type expert --use_kg;

# python visualize_stats.py --env lava;

python eval_env.py --env key --model_type random;
python eval_env.py --env key --model_type MAPPO --load_model_path '/home/senyu/AI_planning/gym-multigrid-master/results/mappo/key/steps_5000000_ngames_39062/mappo_agent_model.pth';
python eval_env.py --env key --model_type MAPPO --use_kg --load_model_path '/home/senyu/AI_planning/gym-multigrid-master/results/mappo/key/steps_5000000_ngames_39062_USE_KG_ball + key/mappo_agent_model.pth';

# python eval_env.py --env key --model_type COMA --load_model_path '/Users/celine/Desktop/3A/CS5446/Project_Git/AI_planning/gym-multigrid-master/results/coma/key/steps_1500000_ngames_11974/model_weights_save_dir';
# python eval_env.py --env key --model_type COMA --use_kg --load_model_path '/Users/celine/Desktop/3A/CS5446/Project_Git/AI_planning/gym-multigrid-master/results/coma/key/steps_1500000_ngames_12797_USE_KG_ball + key/model_weights_save_dir';

python eval_env.py --env key --model_type expert --use_kg;

# python visualize_stats.py --env key; 

# python eval_env.py --env lava2 --model_type random;
 #python eval_env.py --env lava2 --model_type MAPPO --load_model_path
 #python eval_env.py --env lava2 --model_type MAPPO --use_kg --load_model_path
 #python eval_env.py --env lava2 --model_type COMA --load_model_path
 #python eval_env.py --env lava2 --model_type COMA --use_kg --load_model_path
 #python eval_env.py --env lava2 --model_type expert --use_kg;
 #python visualize_stats.py --env lava2; 