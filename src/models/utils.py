import os
import pickle

def torch_save(model, save_path):
    if os.path.dirname(save_path) != '':
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'wb') as f:
        pickle.dump(model, f)


def torch_load(save_path, device=None):
    with open(save_path, 'rb') as f:
        model = pickle.load(f)

    if device is not None:
        model = model.to(device)

    return model