from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

from tensorflow.keras.preprocessing.text import Tokenizer

from aristote.utils import predict_format


class TextTokenizer(object):

    def __init__(self, filters='', num_words=100_000, lower=True, char_level=False, oov_token="[UNK]"):
        self.filters = filters
        self.num_words = num_words
        self.lower = lower
        self.char_level = char_level
        self.oov_token = oov_token
        self.detokenizer = TreebankWordDetokenizer().detokenize
        self.tokenizer = Tokenizer(
            filters=filters, num_words=num_words, lower=lower, char_level=char_level, oov_token=oov_token)

    @staticmethod
    def sequence_to_sentences(sequence):
        return sent_tokenize(sequence)

    def sentences_to_sequences(self, sentences):
        return self.detokenizer(sentences)

    @staticmethod
    def sentence_to_words(sentence):
        return word_tokenize(sentence)

    def words_to_sentence(self, words):
        return self.detokenizer(words)

    @predict_format
    def fit_tokenizer(self, text):
        self.tokenizer = self.tokenizer.fit_on_texts(text)

    @predict_format
    def text_to_token(self, text):
        self.tokenizer.texts_to_sequences(text)

    def token_to_text(self, tokens):
        words = [self.tokenizer.word_index.get(token, self.oov_token) for token in tokens]

        return self.words_to_sentence(words)
