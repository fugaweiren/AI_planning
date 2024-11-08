ENV_CLASS = {
    "simple": 'gym_multigrid.envs:CollectGame4HEnv10x10N2',
    "lava" : 'gym_multigrid.envs:CollectGame4HEnv10x10N2Lava',
    "lava_complicated" : 'gym_multigrid.envs:CollectGame4HEnv20x20N2Lava',
    "key" : 'gym_multigrid.envs:KeyCollectGame4HEnv10x10N2'
}

ENV_RULE_SETS = {
    "simple": ["ball only",                 #0
               "ball with search strats",   #1
               "conflicting rules",         #2
               "irrelevant rules"],         #3
    "lava" : ["lava + ball"],
    "key" : ["ball + key", "others"]
}