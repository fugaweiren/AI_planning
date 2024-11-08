call conda activate maagent

@REM python main_mappo.py --env simple --use_kg --kg_set 0 --steps 4000000
@REM python main_mappo.py --env simple --use_kg --kg_set 1 --steps 4000000
call python main_mappo.py --env simple --use_kg --kg_set 2 --steps 4000000
call python main_mappo.py --env simple --use_kg --kg_set 3 --steps 4000000

call python main_mappo.py --env lava --steps 4000000
call python main_mappo.py --env lava --use_kg --steps 4000000


call python main_mappo.py --env key --steps 4000000
call python main_mappo.py --env key --use_kg --steps 4000000
