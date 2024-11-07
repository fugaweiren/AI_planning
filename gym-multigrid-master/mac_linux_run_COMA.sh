conda activate project
# python COMA.py --env simple --use_kg --kg_set 0 --steps 1500000
# python COMA.py --env simple --use_kg --kg_set 1 --steps 1500000
python COMA.py --env simple --use_kg --kg_set 2 --steps 1500000
python COMA.py --env simple --use_kg --kg_set 3 --steps 1500000

python COMA.py --env lava --steps 1500000
python COMA.py --env lava --use_kg --steps 1500000

python COMA.py --env key --steps 1500000
python COMA.py --env key --use_kg --steps 1500000