from keras.layers import Average
from keras.models import Model


def ensemble(models, model_input):
    outputs = [model.outputs[0] for model in models]
    y = Average()(outputs)

    model = Model(inputs=[model_input], outputs=[y], name='ensemble')

    return model
