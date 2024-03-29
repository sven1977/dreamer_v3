
_ALLOWED_DIMS = [
    # Debug sizes.
    "nano", "micro",
    # Regular sizes.
    "XXS", "XS", "S", "M", "L", "XL"]


def get_cnn_multiplier(model_dimension, override=None):
    if override is not None:
        return override

    assert model_dimension in _ALLOWED_DIMS
    cnn_multipliers = {
        "nano": 2, "micro": 4,
        "XXS": 16, "XS": 24, "S": 32, "M": 48, "L": 64, "XL": 96
    }
    return cnn_multipliers[model_dimension]


def get_dense_hidden_units(model_dimension, override=None):
    if override is not None:
        return override

    assert model_dimension in _ALLOWED_DIMS
    dense_units = {
        "nano": 16, "micro": 32,
        "XXS": 128, "XS": 256, "S": 512, "M": 640, "L": 768, "XL": 1024
    }
    return dense_units[model_dimension]


def get_gru_units(model_dimension, override=None):
    if override is not None:
        return override

    assert model_dimension in _ALLOWED_DIMS
    gru_units = {
        "nano": 16, "micro": 32,
        "XXS": 128, "XS": 256, "S": 512, "M": 1024, "L": 2048, "XL": 4096
    }
    return gru_units[model_dimension]


def get_num_z_categoricals(model_dimension, override=None):
    if override is not None:
        return override

    assert model_dimension in _ALLOWED_DIMS
    gru_units = {
        "nano": 4, "micro": 8,
        "XXS": 32, "XS": 32, "S": 32, "M": 32, "L": 32, "XL": 32
    }
    return gru_units[model_dimension]


def get_num_z_classes(model_dimension, override=None):
    if override is not None:
        return override

    assert model_dimension in _ALLOWED_DIMS
    gru_units = {
        "nano": 4, "micro": 8,
        "XXS": 32, "XS": 32, "S": 32, "M": 32, "L": 32, "XL": 32
    }
    return gru_units[model_dimension]


def get_num_curiosity_nets(model_dimension, override=None):
    if override is not None:
        return override

    assert model_dimension in _ALLOWED_DIMS
    num_curiosity_nets = {
        "nano": 8, "micro": 8,
        "XXS": 8, "XS": 8, "S": 8, "M": 8, "L": 8, "XL": 8
    }
    return num_curiosity_nets[model_dimension]


def get_num_dense_layers(model_dimension, override=None):
    if override is not None:
        return override

    assert model_dimension in _ALLOWED_DIMS
    num_dense_layers = {
        "nano": 1, "micro": 1,
        "XXS": 1, "XS": 1, "S": 2, "M": 3, "L": 4, "XL": 5
    }
    return num_dense_layers[model_dimension]
