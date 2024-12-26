model_dict = dict()

def register_model(model_name):
    def decorator(func):
        model_dict[model_name] = func

    return decorator

def create_model(model_name, num_classes):
    if model_name in model_dict:
        model = model_dict[model_name]
        return model(num_classes = num_classes)
    
    else:
        raise ValueError(f"Model {model_name} not found in register !")