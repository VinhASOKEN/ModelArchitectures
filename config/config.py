import os
from yaml import load

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

path_save_cfg = "/data/disk2/vinhnguyen/ModelArchitectures/config/config.yaml"

def get_config():
    cfg = None
    if os.path.exists(path_save_cfg):
        with open(path_save_cfg) as f:
            cfg = load(f, Loader)
    return cfg