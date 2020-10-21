import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

from aristote.tensorflow_helper.saver_helper import TensorflowLoaderSaver
from aristote.settings import MODEL_PATH
from aristote.utils import timer, predict_format


class Tfidf(TensorflowLoaderSaver):

	def __init__(
			self, bigrams=False, unigrams=True, analyzer='word', base_path=MODEL_PATH, name="tf_idf", model_load=False):
		super().__init__(name, model_load, base_path=base_path)
		self.unigrams = 1 if unigrams else 2
		self.bigrams = 2 if bigrams else 1
		self.analyzer = analyzer
		self.tfidf = self.get_model()

	def get_model(self):
		return TfidfVectorizer(analyzer=self.analyzer, ngram_range=(self.unigrams, self.bigrams))

	@timer
	def fit_model(self, documents):
		assert type(documents) in [list, np.ndarray]
		self.tfidf.fit(documents)

	@timer
	@predict_format
	def transform(self, text):
		return self.tfidf.transform(text).toarray()

	@property
	def get_features_name(self):
		return self.tfidf.get_feature_names()
