import streamlit as st

from aristote.preprocessing.tokenization import SequenceSplitter
from aristote.preprocessing.normalization import TextNormalization


class Normalization(object):

    def __init__(self):
        self.sequencer = SequenceSplitter()
        self.normalizer = TextNormalization()

    def main(self):
        possibles_pipeline = [
            "text_correction", "text_demojis", "text_demoticons", "text_decontraction", "text_stemming"
        ]
        text = st.text_area("Enter the text to normalize here:")
        pipeline = st.multiselect("Select one or more transformations", possibles_pipeline)
        if text:
            sentences = self.sequencer.tokenize(text)
            lst_sentences = []
            for sentence in sentences:
                for transformation in pipeline:
                    fct = getattr(self.normalizer, transformation)
                    sentence = fct(sentence)
                lst_sentences.append(sentence)
            new_text = SequenceSplitter().detokenize(lst_sentences)
            st.markdown(new_text)
