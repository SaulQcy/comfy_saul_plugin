import importlib

def get_exp_by_name(exp_name):
    exp = exp_name.replace("-", "_")  # convert string like "yolox-s" to "yolox_s"
    module_name = ".".join(["exp", exp])
    exp_object = importlib.import_module(module_name).Exp()
    return exp_object