import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from keras import layers, models
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import pandas as pd

batch_size = 128
epochs = 100
latent_dim = 50
num_samples = 30000 #학습 데이터 개수
data_path = "seq2seq_keras/data/Corpus10/eng2kor.txt"

input_texts = []
target_texts = []

with open(data_path, 'r', encoding='utf-8') as f:
    lines = f.read().split("\n")

sorted(lines, key=len)

for idx, line in enumerate(lines):
    if len(line.split("\t")) < 2 :
        continue
    input_text, target_text = line.split("\t")[:2]
    input_texts.append(input_text)
    target_texts.append(target_text)

lines = pd.DataFrame({'eng':input_texts, 'kor':target_texts})
lines = lines[: min(num_samples, len(lines) - 1)]

lines.shape

#Data Cleanup
lines.eng=lines.eng.apply(lambda x: x.lower())
lines.kor=lines.kor.apply(lambda x: x.lower())

# Take the length as 50
import re
lines.eng=lines.eng.apply(lambda x: re.sub("'", '', x)).apply(lambda x: re.sub(",", ' COMMA', x))
lines.kor=lines.kor.apply(lambda x: re.sub("'", '', x)).apply(lambda x: re.sub(",", ' COMMA', x))

import string
exclude = set(string.punctuation)
lines.eng=lines.eng.apply(lambda x: ''.join(ch for ch in x if ch not in exclude))
lines.kor=lines.kor.apply(lambda x: ''.join(ch for ch in x if ch not in exclude))

remove_digits = str.maketrans('', '', string.digits)
lines.eng=lines.eng.apply(lambda x: x.translate(remove_digits))
lines.kor=lines.kor.apply(lambda x: x.translate(remove_digits))

#Generate synthetic data
lines.kor = lines.kor.apply(lambda x: 'START_ ' + x + ' _END')

# Create vocabulary of words
all_eng_words = set()
for eng in lines.eng:
    for word in eng.split():
        if word not in all_eng_words:
            all_eng_words.add(word)

all_kor_words = set()
for kor in lines.kor:
    for word in kor.split():
        if word not in all_kor_words:
            all_kor_words.add(word)

print(len(all_eng_words), len(all_kor_words))

max_decoder_seq_length = np.max([len(l.split()) for l in lines.kor])
max_encoder_seq_length = np.max([len(l.split()) for l in lines.eng])

input_words = sorted(list(all_eng_words))
target_words = sorted(list(all_kor_words))
num_encoder_tokens = len(all_eng_words)
num_decoder_tokens = len(all_kor_words)

input_token_index = dict(
    [(word, i) for i, word in enumerate(input_words)])
target_token_index = dict(
    [(word, i) for i, word in enumerate(target_words)])



encoder_input_data = np.zeros((len(lines.eng), max_encoder_seq_length),dtype='float32')
decoder_input_data = np.zeros((len(lines.kor), max_decoder_seq_length),dtype='float32')
decoder_target_data = np.zeros((len(lines.kor), max_decoder_seq_length, num_decoder_tokens), dtype="float16")

for i, (input_text, target_text) in enumerate(zip(lines.eng, lines.kor)):
    for t, word in enumerate(input_text.split()):
        encoder_input_data[i, t] = input_token_index[word]
    for t, word in enumerate(target_text.split()):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t] = target_token_index[word]
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[word]] = 1.

encoder_inputs = layers.Input(shape=(None,))
en_x = layers.Embedding(num_encoder_tokens, latent_dim)(encoder_inputs)
encoder = layers.GRU(latent_dim, return_state=True)
_, state_h = encoder(en_x)

decoder_inputs = layers.Input(shape=(None,))
dex = layers.Embedding(num_decoder_tokens, latent_dim)
final_dex = dex(decoder_inputs)

decoder_gru = layers.GRU(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _ = decoder_gru(final_dex, initial_state=state_h)

decoder_dense = layers.Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)
model = models.Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
model.summary()

import os
BASE_DIR = os.path.abspath(".")
WORK_DIR = "seq2seq_keras"
os.makedirs(os.path.join(BASE_DIR, WORK_DIR), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, WORK_DIR, "model"), exist_ok=True)
callbacks = [
    ModelCheckpoint(filepath=os.path.join(BASE_DIR, WORK_DIR, "model", "seq2seq2_gru_word_level_model.h5"), monitor='loss', save_best_only=True)
]

model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=callbacks)

encoder_model = models.Model(encoder_inputs, state_h)
encoder_model.summary()


#Create sampling model

decoder_state_input_h = layers.Input(shape=(latent_dim,))

final_dex2 = dex(decoder_inputs)

decoder_outputs2, state_h2 = decoder_gru(final_dex2, initial_state=decoder_state_input_h)
decoder_outputs2 = decoder_dense(decoder_outputs2)
decoder_model = models.Model(
    [decoder_inputs, decoder_state_input_h],
    [decoder_outputs2, state_h2])

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())

#Function to generate sequences
def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0] = target_token_index['START_']

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h = decoder_model.predict([target_seq, states_value])

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += ' '+sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '_END' or len(decoded_sentence) > 52):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = h

    return decoded_sentence

for seq_index in range(10):
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', lines.eng[seq_index: seq_index + 1])
    print('Decoded sentence:', decoded_sentence)