![Aristote](docs/ARTISTOTE.png)

![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)
-----------------

aristote is a suite of open source Python module for Natural Language. All the modules are easy to use, also you can tune them easily. 

It can help you in differents ways:

* Preprocessing
    * Normalization
    * Tokenization

* Featuring
    * Clustering
    * Dimensional Reduction
    * Feature Extraction

* Modeling
    * Text Classification
    * Text Extraction
    * Text Similarity
    * Text Question Answer

##### Dependencies
```bash
- Python (>= 3.6)
- Tensorflow (>= 2.0.1)
- Nltk (>= 3.4.5)
- Scikit-learn (>= 0.22.2)
```

##### User installation
With pypi:
```pip install aristote```

Custom installation:
```bash
git clone git@github.com:Jwuthri/aristote.git
cd aristote
pip install -e .
```

Using docker:
```bash
docker-compose build
docker-compose run
```

Processing
----------
###### Normalization
```
* Word Stemming
* Word Lemming
* SpellCheck correction
* Remove Emoji
* Remove Emoticons
* Remove text contraction
```

```python
from aristote.preprocessing.normalization import TextNormalization

tn = TextNormalization()

# text correction
text = 'Let me tell you somthing you alrady know.'
cleaned_text = tn.text_correction(text)
cleaned_text
>> 'Let me tell you something you already know.'

# Remove emoji/emot
text = 'Let me tell you something you already know 👍'
demojize_text = tn.text_demojis(text, how_replace="")
demojize_text
>> 'Let me tell you something you already know'

text = 'Let me tell you something you already know :)'
demoticons_text = tn.text_demoticons(text, how_replace="")
demoticons_text
>> 'Let me tell you something you already know '

# Decontract words
text = "I'd like to know yall guys"
decontraction_text = tn.text_decontraction(text)
decontraction_text
>> 'I would like to know you all guys'

# Stemming word
word = "shipping"
stemmed_word = TextNormalization().word_stemming(word)
stemmed_word
>> 'ship'

# example of sentence cleaning:
text = "I'd like to tell you somthing you alrady know."
decontraction_text = tn.text_decontraction(text)
cleaned_text = tn.text_correction(decontraction_text)
cleaned_text
>> 'I would like to tell you something you already know.'
```

###### Tokenization
```
* Sentence Tokenization
* Sentence DeTokenization
* Sequence Tokenization
* Sequence DeTokenization
```

```python
from aristote.preprocessing.tokenization import SentenceSplitter, SequenceSplitter

# Sentence Tokenization
sentence = "Let me tell you something you already know."
tokens = SentenceSplitter().tokenize(sentence)
tokens
>> ['Let', 'me', 'tell', 'you', 'something', 'you', 'already', 'know', '.']

# Sentence DeTokenization
sentence = SentenceSplitter().detokenize(tokens)
sentence
>> 'Let me tell you something you already know.'

# Sequence Tokenization
sequence = "Let me tell you something you already know. The world ain’t all sunshine and rainbows."
sentences = SequenceSplitter().tokenize(sequence)
sentences
>> ['Let me tell you something you already know.', 'The world ain’t all sunshine and rainbows.']

# Sequence DeTokenization
sequence = SequenceSplitter().detokenize(sentences)
sequence
>> 'Let me tell you something you already know. The world ain’t all sunshine and rainbows.'
```

Featuring
---------
###### Clustering
```
* HDBSCAN
* Kmeans
* AHC (Agglomerative-Hierarchical-Clustering)
* AE (Auto-Encoder)
* VAE (Variationnal-Auto-Encoder)
```
###### Dimensional Reduction
```
* LDA (Linear-Discriminant-Analysis)
* QDA (Quadratic-Discriminant-Analysis)
* PCA (Principal-Component-Analysis)
* SVD (Singular-Value-Decomposition)
* SPCA (Scaled-Principal-Component-Analysis)
* UMAP (Uniform-Manifold-Approximation-Projection)
* AE (Auto-Encoder)
* VAE (Variationnal-Auto-Encoder)
```
###### Feature Extraction
```
* TfIdf (Term-Frequency–Inverse-Document-Frequency)
* Embedding (MultiLanguage)
```

Modeling
---------
###### Text Classification

```
* Find the most relevant label for a given text
```

| index | binary   | multi        | single   | feature                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
|-------:|:---------|:-------------|:---------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|  1 | positive | ['positive'] | positive | About some articles you contribute significantly  I know you have contributed to the articles Armenian Genocide and Confiscated Armenian properties in Turkey, and your contribution makes non-Armenians know better about the situation of Armenians in the Ottoman Empire and Republic of Turkey. However, both articles have problems. The former does not include papers and reviews published in International Journal of Armenian Genocide Studies, ... |
|  2 | negative | ['negative'] | negative |  I'm afraid that you need to help yourself - by following the advice/instruction in FisherQueens unblock decline message above.  That is the only way that you have a chance of getting your block lifted.  talk "                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
|  3 | positive | ['positive'] | positive | Hrmm, gotcha.. I thought it was okay since most of the other leaders in the civilization games have it in their popular culture sections as well... Napoleon, Elizabeth I, Wu Zetian, Boudica, Dido, Nebuchadnezzar II, Harun al-Rashid, George Washington, Alexander the Great, Oda Nobunaga, Askia, Augustus Caesar, Genghis Khan, Gustavus Adolphus, Hiawatha, Kamehameha, Ramkhamhaeng and Sejong all have it listed. — -   "                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
|  4 | negative | ['insult', 'obscene', 'toxic', 'negative'] | insult | COME DUCT-TAKE YOU AND RAPE YOU TILL YOU DIE FUCKHEAD |
|  5 | positive | ['neutral']  | neutral  | Each alum agrees to  how much information can be released to other alums and to the general public.  Whether you think it is stalking is irrelevant.  I believe you are throwing allegations of stalking because you are a petulant little boy who is stamping his feet because he didn't get his way.  You need metaphorically pantsed, your glasses thumbed, and your milk spilled.  Now go pull some wings off of flies, you weirdo.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |


```python
import os

import pandas as pd

from aristote.settings import DATASET_PATH
from aristote.dataset.pandas import remove_rows_contains_null
from aristote.text_classification.trainer import Trainer

dataset_path = os.path.join(DATASET_PATH, "sentiment.csv")
dataset = pd.read_csv(dataset_path)
# possible label_type: ['binary-class', 'multi-label', 'multi-class']

# Binary classifier
x, y, label_type, epochs = "feature", "binary", "binary-class", 2
dataset = dataset[dataset[x].notnull()]
dataset = remove_rows_contains_null(dataset, y)
architecture = [('DENSE', 256), ("DROPOUT", 0.2), ('DENSE', 128)]
trainer = Trainer(dataset, x, y, label_type, architecture, epochs=epochs, use_comet=True)
trainer.train()
# results = ["Ok": "positive", "I don't like this": "negative", "I like it": "positive", "Fuck you": "negative"]

# Single label classifier
x, y, label_type, epochs = "feature", "single", "multi-class", 2
dataset = remove_rows_contains_null(dataset, x)
dataset = remove_rows_contains_null(dataset, y)
architecture = [('CNN', 256), ("DROPOUT", 0.2), ('DENSE', 128)]
trainer = Trainer(dataset, x, y, label_type, architecture, epochs=epochs, use_comet=True)
trainer.train()
# results = ["Ok": "neutral", "I don't like this": "negative", "I like it": "positive", "Fuck you": "insult"]

# Multi label classifier
x, y, label_type, epochs = "feature", "multi", "multi-label", 2
dataset = remove_rows_contains_null(dataset, x)
dataset = remove_rows_contains_null(dataset, y)
architecture = [('LSTM', 256), ("DROPOUT", 0.2), ('DENSE', 128)]
trainer = Trainer(dataset, x, y, label_type, architecture, epochs=epochs, use_comet=True)
trainer.train()
# results = ["Ok": "neutral", "I don't like this": "negative", "I like it": "positive", "Fuck you": ("negative", "toxic", "insult")]
```
###### Text Generation

```
* Predict the next "n" words for a given input text, also able to complet a word.
    (ex: I am very => I am very happy about this)    
    (ex: I am v => I am very happy about this)
```

```python
import os
import pandas as pd

from aristote.settings import DATASET_PATH
from aristote.dataset.pandas import remove_rows_contains_null
from aristote.text_generation.trainer import Trainer

dataset_path = os.path.join(DATASET_PATH, "sentiment.csv")
dataset = pd.read_csv(dataset_path)
dataset = remove_rows_contains_null(dataset, "feature")
architecture = [('RNN', 512), ('DENSE', 1024)]
input_shape, embedding_size, epochs, number_labels_max = 64, 128, 2, 5000

data = dataset['feature'].values
trainer = Trainer(
    architecture, number_labels_max, data, input_shape=input_shape, embedding_size=embedding_size, epochs=epochs
)
trainer.train()
# results = #TODO
```

###### Text Extraction
```
* Find the most relevants sentences in a text
```

###### Entities Recognition
```
* Find specific information from a text like:
    addresses
    email
    phone number ... 
```

###### Text Similarity
```
* Cosine similarity between 2 documents
* For a given document find the most similar document from a list of documents
```
###### Text Question Answer
```
* For a given context, find the best answer for a question
```
