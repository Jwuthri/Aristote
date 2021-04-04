import os

if "TF_CPP_MIN_LOG_LEVEL" not in os.environ:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from aristote.tensorflow_helper.predictor_helper import TensorflowPredictor


if __name__ == '__main__':
    name, model_load = "2020_11_29_20_50_35_generation", True
    text = "I am very happy"
    predictor = TensorflowPredictor(name, model_load)
    prediction = predictor.generate_text(text, completion_threshold=0.0, prediction_threshold=0.0, max_predictions=5)
    print(prediction)
