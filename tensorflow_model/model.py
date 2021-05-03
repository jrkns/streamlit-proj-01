import os
import pickle
import random
from collections import defaultdict

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences


class LyricsGeneratorModel:
    def __init__(self):
        self._model = self._init_model()
        self._model.load_weights(
            f'{os.path.dirname(os.path.abspath(__file__))}/model_weight.h5')
        with open(f'{os.path.dirname(os.path.abspath(__file__))}/model_meta.pickle', 'rb') as f:
            _, self._word_to_idx, self._idx_to_word = pickle.load(f)

        self._artist_map = defaultdict(int)
        self._artist_map['bnk48'] = 1
#         self._artist_map['potato'] = 2
#         self._artist_map['carabao'] = 3
#         self._artist_map['sekloso'] = 4
#         self._artist_map['grasshopper'] = 5

    def _init_model(self):
        inpt = Input(shape=(5, ))
        x = Embedding(11963, 16, input_length=5)(inpt)
        x = Dropout(0.1)(x)
        x = LSTM(512)(x)
        x = Dropout(0.1)(x)
        opt_0 = Dense(11963, activation='softmax', name='output_0')(x)
        opt_1 = Dense(11963, activation='softmax', name='output_1')(x)

        model = Model(inputs=inpt, outputs=[opt_0, opt_1])

        model.compile(optimizer=Adam(
            lr=0.01), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def _change_word_to_index(self, word):
        if word not in self._word_to_idx:
            return self._word_to_idx['<unk>']
        return self._word_to_idx[word]

    def _texts_to_sequences(self, text, word_to_index):
        text = text.strip().split(' ')
        token_list = [self._change_word_to_index(x) for x in text]
        return token_list

    def _sample_output(self, probs, temperature=1.0):
        probs = [p**(1/temperature) for p in probs]
        sum_probs = sum(probs)
        probs = [p/sum_probs for p in probs]
        r = random.uniform(0, 1)
        acp = 0.
        for i, p in enumerate(probs):
            acp += p
            if acp >= r:
                return i
        return len(probs)-1

    def _temperature_sampling_decode(self, seed_text, max_gen_length, input_len, temperature, opt_head=0):
        current_text = seed_text
        probs = []
        for _ in range(max_gen_length):
            current_idx = pad_sequences([self._texts_to_sequences(
                current_text, self._word_to_idx)], maxlen=input_len, padding='pre')
            next_token_idx_probs = self._model.predict(current_idx)[
                opt_head][0]
            next_token_idx = self._sample_output(
                next_token_idx_probs, temperature)
            output_word = self._idx_to_word[next_token_idx]
            probs.append(next_token_idx_probs[next_token_idx])
            current_text += ' ' + output_word
        return current_text, probs

    def _greedy_decode(self, seed_text, max_gen_length, input_len, opt_head=0):
        current_text = seed_text
        probs = []
        for _ in range(max_gen_length):
            current_idx = pad_sequences([self._texts_to_sequences(
                current_text, self._word_to_idx)], maxlen=input_len, padding='pre')
            next_token_idx = self._model.predict(current_idx)[opt_head]
            output_word = self._idx_to_word[np.argmax(
                next_token_idx, axis=1)[0]]
            probs.append(np.max(next_token_idx, axis=1)[0])
            current_text += ' ' + output_word
        return current_text, probs

    def _post_process(self, inpt):
        return [e for e in ''.join([e.replace('<next>', '\n') for e in inpt.split()]).split('\n') if e]

    def _greedy_predict(self, seed_text, max_gen_length=100, post_process=True, artist='default'):
        predictions = self._greedy_decode(
            str(seed_text), max_gen_length, 5, self._artist_map[artist])[0]
        return self._post_process(predictions) if post_process else predictions

    def predict(self, seed_text, max_gen_length=50, temp=0.875, post_process=True, artist='default'):
        predictions = self._temperature_sampling_decode(
            str(seed_text), max_gen_length, 5, temp, self._artist_map[artist])[0]
        return self._post_process(predictions) if post_process else predictions
