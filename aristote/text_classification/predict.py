from aristote.tensorflow_helper.predictor_helper import TensorflowPredictor


if __name__ == '__main__':
    name, model_load, cleaning_func = "2020_10_25_01_12_01_sentiment", True, None
    text = "I am very happy"
    predictor = TensorflowPredictor(name, model_load, cleaning_func)
    prediction = predictor.predict(text=text)
    print(prediction)
