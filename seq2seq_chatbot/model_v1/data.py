import re, os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from konlpy.tag import Okt
from configs import DEFINES

FILTERS = "([~.,!?\"':;)(])"
PAD = "<PADDING>"
STD = "<START>"
END = "<END>"
UNK = "<UNKNOWN>"

PAD_INDEX = 0
STD_INDEX = 1
END_INDEX = 2
UNK_INDEX = 3

MARKER = [PAD, STD, END, UNK]
CHANGE_FILTER = re.compile(FILTERS)

#데이터 로드 함수
def load_data():
    data_df = pd.read_csv(DEFINES.data_path, header=0)
    question, answer = list(data_df['Q']), list(data_df['A'])
    train_input, eval_input, train_label, eval_label = train_test_split(question, answer, test_size=0.33, random_state=42)
    return train_input, train_label, eval_input, eval_label

#형태소 분석 및 재결합 함수
def prepro_like_morphlized(data) :
    morph_analyzer = Okt()
    result_data = list()
    for seq in tqdm(data) :
        morphlized_seq = " ".join(morph_analyzer.morphs(seq.replace(' ', '')))
        result_data.append(morphlized_seq)
    return result_data

#인코더 입력 데이터 생성함수
def enc_processing(value, dictionary) :
    seqs_input_idx = []
    seqs_len = []
    if DEFINES.tokenize_as_morph :
        value = prepro_like_morphlized(value)

    for seq in value :
        seq = re.sub(CHANGE_FILTER, "", seq)
        seq_idx = []
        for word in seq.split() :
            if dictionary.get(word) is not None :
                seq_idx.extend([dictionary[word]])
            else :
                seq_idx.extend([dictionary[UNK]])

        if len(seq_idx) > DEFINES.max_sequence_length:
            seq_idx = seq_idx[:DEFINES.max_sequence_length]

        seqs_len.append(len(seq_idx))
        seq_idx += (DEFINES.max_sequence_length - len(seq_idx)) * [dictionary[PAD]]
        seqs_input_idx.append(seq_idx)

    return np.asarray(seqs_input_idx), seqs_len

#디코더 입력 데이터 생성함수
def dec_input_processing(value, dictionary) :
    seqs_output_idx = []
    seqs_len = []

    if DEFINES.tokenize_as_morph :
        value = prepro_like_morphlized(value)

    for seq in value :
        seq = re.sub(CHANGE_FILTER, "", seq)
        seq_idx = []
        seq_idx = [dictionary[STD]] + [dictionary[word] for word in seq.split()]

        if len(seq_idx) > DEFINES.max_sequence_length:
            seq_idx = seq_idx[:DEFINES.max_sequence_length]

        seqs_len.append(len(seq_idx))
        seq_idx += (DEFINES.max_sequence_length - len(seq_idx)) * [dictionary[PAD]]
        seqs_output_idx.append(seq_idx)

    return np.asarray(seqs_output_idx), seqs_len


# 디코더 타겟 데이터 생성함수
def dec_target_processing(value, dictionary):
    seqs_target_idx = []

    if DEFINES.tokenize_as_morph:
        value = prepro_like_morphlized(value)

    for seq in value:
        seq = re.sub(CHANGE_FILTER, "", seq)
        seq_idx = [dictionary[word] for word in seq.split()]

        if len(seq_idx) >= DEFINES.max_sequence_length:
            seq_idx = seq_idx[:DEFINES.max_sequence_length-1] + [dictionary[END]]
        else :
            seq_idx.append(dictionary.get(END))

        seq_idx += (DEFINES.max_sequence_length - len(seq_idx)) * [dictionary[PAD]]

        seqs_target_idx.append(seq_idx)

    return np.asarray(seqs_target_idx)

#인덱스를 문장으로 변환하는 함수
def pred2string(value, dictionary):
    sentence_string = []
    for v in value:
        # 딕셔너리에 있는 단어로 변경해서 배열에 담는다.
        sentence_string = [dictionary[index] for index in v['indexs']]

    print(sentence_string)
    answer = ""
    # 패딩값과 엔드값이 담겨 있으므로 패딩은 모두 스페이스 처리 한다.
    for word in sentence_string:
        if word not in PAD and word not in END:
            answer += word
            answer += " "

    print(answer)
    return answer

#데이터의 단어 토큰나이징 함수
def data_tokenizer(data):
    words = []
    for sentence in data :
        sentence = re.sub(CHANGE_FILTER, "", sentence)
        for word in sentence.split() :
            words.append(word)

    return words

#단어 사전 생성 함수
def load_vocabulary() :
    vocab_list = []
    if not os.path.exists(DEFINES.vocabulary_path) :
        if os.path.exists(DEFINES.data_path) :
            data_df = pd.read_csv(DEFINES.data_path, encoding='utf-8')
            Q, A = data_df.Q.tolist(), data_df.A.tolist()
            if DEFINES.tokenize_as_morph :
                Q = prepro_like_morphlized(Q)
                A = prepro_like_morphlized(A)
            data = []
            data.extend(Q + A)
            words = data_tokenizer(data)
            words = list(set(words))
            words[:0] = MARKER

        with open(DEFINES.vocabulary_path, 'w', encoding='utf-8') as vocab_file :
            for word in words:
                vocab_file.write(word + '\n')

    with open(DEFINES.vocabulary_path, 'r', encoding='utf-8') as vocab_file :
        for line in vocab_file :
            vocab_list.append(line.strip())

    word2idx = {word:idx for idx, word in enumerate(vocab_list)}
    idx2word = {val:key for key, val in word2idx.items()}

    return word2idx, idx2word, len(word2idx)

# 데이터셋 생성 함수들

# 맵함수
def rearrange(input, output, target) :
    features = {"input": input, "output": output}
    return features, target

def train_input_fn(train_input_enc, train_output_dec, train_target_dec, batch_size) :
    dataset = tf.data.Dataset.from_tensor_slices((train_input_enc, train_output_dec, train_target_dec)).\
        shuffle(len(train_input_enc)).batch(batch_size).map(rearrange).repeat().\
        make_one_shot_iterator()
    return dataset.get_next()

def eval_input_fn(eval_input_enc, eval_output_dec, eval_target_dec, batch_size) :
    dataset = tf.data.Dataset.from_tensor_slices((eval_input_enc, eval_output_dec, eval_target_dec)).\
        shuffle(len(eval_input_enc)).batch(batch_size).map(rearrange).repeat(1).\
        make_one_shot_iterator()
    return dataset.get_next()


