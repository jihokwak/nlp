import re, os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from konlpy.tag import Okt
from configs import DEFINES

PAD_MASK = 0
NON_PAD_MASK = 1

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
        seq_idx.reverse()
        seqs_input_idx.append(seq_idx)

    return np.asarray(seqs_input_idx), seqs_len

# 디코더 타겟 데이터 생성함수
def dec_target_processing(value, dictionary):
    seqs_target_idx = []
    seqs_len = []

    if DEFINES.tokenize_as_morph:
        value = prepro_like_morphlized(value)

    for seq in value:
        seq = re.sub(CHANGE_FILTER, "", seq)
        seq_idx = [dictionary[word] for word in seq.split()]

        if len(seq_idx) >= DEFINES.max_sequence_length:
            seq_idx = seq_idx[:DEFINES.max_sequence_length-1] + [dictionary[END]]
        else :
            seq_idx.append(dictionary.get(END))
        seqs_len.append([PAD_MASK if num > len(seq_idx) else NON_PAD_MASK for num in range (DEFINES.max_sequence_length)])
        seq_idx += (DEFINES.max_sequence_length - len(seq_idx)) * [dictionary[PAD]]

        seqs_target_idx.append(seq_idx)

    return np.asarray(seqs_target_idx), np.asarray(seqs_len)

#인덱스를 문장으로 변환하는 함수
def pred2string(value, dictionary):
    # 텍스트 문장을 보관할 배열을 선언한다.
    sentence_string = []
    # 인덱스 배열 하나를 꺼내서 v에 넘겨준다.
    if DEFINES.serving == True:
        for v in value['output']:
            sentence_string = [dictionary[index] for index in v]
    else:
        for v in value:
            # 딕셔너리에 있는 단어로 변경해서 배열에 담는다.
            sentence_string = [dictionary[index] for index in v['indexs']]

    print(sentence_string)
    answer = ""
    # 패딩값도 담겨 있으므로 패딩은 모두 스페이스 처리 한다.
    for word in sentence_string:
        if word not in PAD and word not in END:
            answer += word
            answer += " "
    # 결과를 출력한다.
    print(answer)
    return answer

#데이터의 단어 토큰나이징 함수
def data_tokenizer(data):
    words = []
    for sentence in data :
        sentence = re.sub(CHANGE_FILTER, "", sentence)
        for word in sentence.split() :
            words.append(word)

    return [word for word in words if word]

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
            data.extend(Q)
            data.extend(A)
            words = data_tokenizer(data)
            words = list(set(words))
            words[:0] = MARKER

        with open(DEFINES.vocabulary_path, 'w', encoding='utf-8') as vocab_file :
            for word in words:
                vocab_file.write(word + '\n')

    with open(DEFINES.vocabulary_path, 'r', encoding='utf-8') as vocab_file :
        for line in vocab_file :
            vocab_list.append(line.strip())

    char2idx = {word:idx for idx, word in enumerate(vocab_list)}
    idx2char = {val:key for key, val in char2idx.items()}

    return char2idx, idx2char, len(char2idx)

# 데이터셋 생성 함수들

# 맵함수
def rearrange(input, target) :
    features = {"input": input}
    return features, target

def train_rearrange(input,length, target):
    features = {"input": input, "length":length}
    return features, target

def train_input_fn(train_input_enc, train_target_dec_len, train_target_dec, batch_size) :
    dataset = tf.data.Dataset.from_tensor_slices((train_input_enc, train_target_dec_len, train_target_dec)).\
        shuffle(len(train_input_enc)).batch(batch_size).map(train_rearrange).repeat().\
        make_one_shot_iterator()
    return dataset.get_next()

def eval_input_fn(eval_input_enc, eval_target_dec, batch_size) :
    dataset = tf.data.Dataset.from_tensor_slices((eval_input_enc,  eval_target_dec)).\
        shuffle(len(eval_input_enc)).batch(batch_size).map(rearrange).repeat(1).\
        make_one_shot_iterator()
    return dataset.get_next()

def main(self):
    char2idx, idx2char, vocabulary_length = load_vocabulary()



if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)


