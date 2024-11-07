conda activate project
# python main_mappo.py --env simple --use_kg --kg_set 0 --steps 5000000
# python main_mappo.py --env simple --use_kg --kg_set 1 --steps 5000000
python main_mappo.py --env simple --use_kg --kg_set 2 --steps 5000000
python main_mappo.py --env simple --use_kg --kg_set 3 --steps 5000000

python main_mappo.py --env lava --steps 5000000
python main_mappo.py --env lava --use_kg --steps 5000000

python main_mappo.py --env key --steps 5000000
python main_mappo.py --env key --use_kg --steps 5000000