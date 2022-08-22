from .piev2 import PieV2Module

MODELS = {
    "piev2": PieV2Module,
}


def get_model(model_type):
    return MODELS[model_type]


def add_model_specific_args(parent_parser, model_type):
    return get_model(model_type).add_model_specific_args(parent_parser)
