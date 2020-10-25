import os
if "TF_CPP_MIN_LOG_LEVEL" not in os.environ:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from aristote.tensorflow_helper.predictor_helper import TensorflowPredictor


if __name__ == '__main__':
    name, model_load, cleaning_func = "2020_10_25_01_48_49_sentiment", True, None
    text = "I am very happy"
    predictor = TensorflowPredictor(name, model_load, cleaning_func)
    prediction = predictor.predict(text=text)
    print(prediction)
