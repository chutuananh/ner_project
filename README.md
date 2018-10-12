Pretrained word embeddings:
- GloVe embeddings for English: download at https://drive.google.com/file/d/1uYtTEp-WpFGLCQ2MyuLiE4RO5vQZ35ej/view?usp=sharing
- Word2Vec embeddings for Dutch and English: download at http://vectors.nlpl.eu/repository/ with id=39 for Dutch and id=40 for English.
- Other Word2Vec embeddings (including French): also download at http://vectors.nlpl.eu/repository/

Necessary packages:
- NumPy
- TensorFlow
- Keras
- Keras-contrib: Installation based on instructions at https://github.com/keras-team/keras-contrib

Run following command to run code file: for example BLSTM_CRF_model.py:
 python BLSTM_CRF_model.py -d english -c yes
