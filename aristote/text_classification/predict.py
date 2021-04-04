import os
if "TF_CPP_MIN_LOG_LEVEL" not in os.environ:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from aristote.tensorflow_helper.predictor_helper import TensorflowPredictor


if __name__ == '__main__':
    name, model_load, cleaning_func = "2020_11_30_17_41_50_sentiment", True, None
    text = [
        "🤩🤩🤩  stoked, you can never have too many thread wallets",
        "🤩🤩🤩, you can never have too many thread wallets",
        "stoked, you can never have too many thread wallets",
        "where is my order?",
        "I don't hate it",
        "I don't like it",
        "It's good",
        "Like it",
        "fuck this shit",
        "You are asshole"
    ]
    predictor = TensorflowPredictor(name, model_load, cleaning_func)

    for t in text:
        prediction = predictor.predict(text=t)
        print(prediction)
