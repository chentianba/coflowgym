INNER_TRACES = [
    {
        "id": 1,
        "path": "scripts/FB2010-1Hr-150-0.txt",
        "desc": "Facebook公开数据集"
    },
    {
        "id": 2,
        "path": "scripts/light_tail.txt",
        "desc": "自定义数据集（LightTail）"
    },
    {
        "id": 3,
        "path": "scripts/valid_1.txt",
        "desc": "自定义数据集"
    },
]

ALGOS = ["SCF", "SEBF", "Aalo", "M-DRL", "CS-GAIL"]

MDRL_MODELS = [
    {
        "model": "model_290_20200604T201123.ckpt",
        "dir": "models/mdrl-facebook/",
        "datasource": 1,
    },
    {
        "model": "model_230_20200607T122043.ckpt",
        "dir": "models/mdrl-facebook/",
        "datasource": 1,
    },
    {
        "model": "model_140_20200618T030642.ckpt",
        "dir": "models/mdrl-lighttail/",
        "datasource": 2,
    },
    {
        "model": "model_190_20200618T032511.ckpt",
        "dir": "models/mdrl-lighttail/",
        "datasource": 2,
    },
    {
        "model": "model_250_20200618T033747.ckpt",
        "dir": "models/mdrl-lighttail/",
        "datasource": 2,
    },
]

GAIL_MODELS = [
    {
        "model": "",
        "dir": "models/gail-valid_1",
        "datasource": 3,
    },
]
