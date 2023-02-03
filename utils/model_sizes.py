

def get_cnn_multiplier(model_dimension, default):
    # Determine size of this model.
    assert model_dimension in [None, "XS", "S", "M", "L", "XL"]
    cnn_multipliers = {
        "XS": 24, "S": 32, "M": 48, "L": 64, "XL": 96
    }
    cnn_multiplier = (
        cnn_multipliers[model_dimension]
        if model_dimension is not None else default
    )
    assert cnn_multiplier is not None
    return cnn_multiplier


def get_dense_hidden_units(model_dimension, default):
    # Determine size of this model.
    assert model_dimension in [None, "XS", "S", "M", "L", "XL"]
    dense_units = {
        "XS": 256, "S": 512, "M": 640, "L": 768, "XL": 1024
    }
    dense_units = (
        dense_units[model_dimension]
        if model_dimension is not None else default
    )
    assert dense_units is not None
    return dense_units


def get_gru_units(model_dimension, default):
    # Determine size of this model.
    assert model_dimension in [None, "XS", "S", "M", "L", "XL"]
    gru_units = {
        "XS": 256, "S": 512, "M": 1024, "L": 2048, "XL": 4096
    }
    gru_units = (
        gru_units[model_dimension]
        if model_dimension is not None else default
    )
    assert gru_units is not None
    return gru_units


def get_num_dense_layers(model_dimension, default):
    # Determine size of this model.
    assert model_dimension in [None, "XS", "S", "M", "L", "XL"]
    num_dense_layers = {
        "XS": 1, "S": 2, "M": 3, "L": 4, "XL": 5
    }
    num_dense_layers = (
        num_dense_layers[model_dimension]
        if model_dimension is not None else default
    )
    assert num_dense_layers is not None
    return num_dense_layers
