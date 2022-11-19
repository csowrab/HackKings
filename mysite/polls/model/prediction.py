import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class SentimentalClassifier():

    def __init__(self):
        vocab_size = 10000
        max_length = 100
        embedding_dim = 64
        trunc_type = 'post'
        padding_type = 'post'
        oov_tok = '<OOV>'
        training_size = 4000

        review_data = pd.read_csv('data/NewIMDBReviews.csv')

        text = list(review_data['text'])
        labels = list(review_data['sentiment'])

        training_sentences = text[0:training_size]
        training_labels = labels[0:training_size]

        testing_sentences = text[training_size:]
        testing_labels = labels[training_size:]

        tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
        tokenizer.fit_on_texts(training_sentences)

        word_index = tokenizer.word_index

        training_sequences = tokenizer.texts_to_sequences(training_sentences)
        training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

        testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
        testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

        training_padded = np.array(training_padded)
        training_labels = np.array(training_labels)
        testing_padded = np.array(testing_padded)
        testing_labels = np.array(testing_labels)

        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
            tf.keras.layers.Conv1D(64, 5, activation='relu'),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        num_epochs = 10
        history = model.fit(training_padded, training_labels, epochs=num_epochs, validation_data=(testing_padded, testing_labels), verbose=2)


        # input_text = st.text_input("Please type a message!", "I love this app")
        new_sentence = list("I like the new pm")
        sequences = tokenizer.texts_to_sequences(new_sentence)
        padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

        self.model = model

        model.save('./SA.h5')

        self.tokenizer = tokenizer
        pickle.dump(self.tokenizer, open('t.sav','wb'))


        if model.predict(padded)[0][0] >= 0.5:
            print("The message was reviewed as positive!")
        else:
            print("The message was reviewed as negative!")

    def predict(self, sentence):
        max_length = 100
        trunc_type = 'post'
        padding_type = 'post'

        sequences = self.tokenizer.texts_to_sequences(sentence)
        padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

        return self.model.predict(padded)[0][0]
