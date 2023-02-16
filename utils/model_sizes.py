

def get_cnn_multiplier(model_dimension, override):
    if override is not None:
        return override

    # Determine size of this model.
    assert model_dimension in ["micro", "XS", "S", "M", "L", "XL"]
    cnn_multipliers = {
        "micro": 8, "XS": 24, "S": 32, "M": 48, "L": 64, "XL": 96
    }
    return cnn_multipliers[model_dimension]


def get_dense_hidden_units(model_dimension, override):
    if override is not None:
        return override

    # Determine size of this model.
    assert model_dimension in ["micro", "XS", "S", "M", "L", "XL"]
    dense_units = {
        "micro": 16, "XS": 256, "S": 512, "M": 640, "L": 768, "XL": 1024
    }
    return dense_units[model_dimension]


def get_gru_units(model_dimension, override):
    if override is not None:
        return override

    # Determine size of this model.
    assert model_dimension in ["micro", "XS", "S", "M", "L", "XL"]
    gru_units = {
        "micro": 32, "XS": 256, "S": 512, "M": 1024, "L": 2048, "XL": 4096
    }
    return gru_units[model_dimension]


def get_num_dense_layers(model_dimension, override):
    if override is not None:
        return override

    # Determine size of this model.
    assert model_dimension in ["micro", "XS", "S", "M", "L", "XL"]
    num_dense_layers = {
        "micro": 1, "XS": 1, "S": 2, "M": 3, "L": 4, "XL": 5
    }
    return num_dense_layers[model_dimension]
