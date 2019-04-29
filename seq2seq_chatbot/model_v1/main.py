'''
시퀀스2시퀀스
시퀀스 입력을 받아 시퀀스 출력을 만드는 모델. 즉 하나의 텍스트 문장이 입력으로 들어오면,
하나의 텍스트 문장을 출력하는 구조.
이 모델의 활용분야는 '기계번역', '텍스트요약', '이미지 설명', '대화모델' 이다.
RNN 모델을 기반으로 한다.
인코더, 디코더로 나뉨.
인코더 부분에서 입력값을 받아 입력값의 정보를 담은 벡터를 만들고,
이후, 디코더에서 이 벡터를 활용해 재귀적으로 출력값을 만드는 구조.

'''

import tensorflow as tf
import numpy  as np
import os, sys
import model as ml
import data
from configs import DEFINES

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_OUT_PATH = 'data_out'

def train(self) :
    data_out_path = os.path.join(BASE_DIR, DATA_OUT_PATH)
    os.makedirs(data_out_path, exist_ok=True)
    # 데이터를 통한 사전 구성
    word2idx, idx2word, vocab_len = data.load_vocabulary()

    train_input, train_label, eval_input, eval_label = data.load_data()
    # 훈련셋 인코딩 / 디코딩 입력 / 디코딩 출력
    train_input_enc, train_input_enc_len = data.enc_processing(train_input, word2idx)
    train_input_dec, train_input_dec_len = data.dec_input_processing(train_label, word2idx)
    train_target_dec = data.dec_target_processing(train_label, word2idx)
    # 평가셋 인코딩 / 디코딩 입력 / 디코딩 출력
    eval_input_enc, eval_input_enc_len = data.enc_processing(eval_input, word2idx)
    eval_input_dec, eval_input_dec_len = data.dec_input_processing(eval_label, word2idx)
    eval_target_dec = data.dec_target_processing(eval_label, word2idx)

    os.makedirs(DEFINES.check_point_path, exist_ok=True)

    classifier = tf.estimator.Estimator(
        model_fn = ml.model,
        model_dir = DEFINES.check_point_path,
        params= {
            'hidden_size': DEFINES.hidden_size,
            'layer_size' : DEFINES.layer_size,
            'learning_rate' : DEFINES.learning_rate,
            'vocab_len' : vocab_len,
            'embedding_size' : DEFINES.embedding_size,
            'embedding' : DEFINES.embedding,
            'multilayer': DEFINES.multilayer,
        }
    )
    # 학습 실행
    classifier.train(
        input_fn=lambda:data.train_input_fn(
            train_input_enc, train_input_dec, train_target_dec, DEFINES.batch_size
        ),
        steps = DEFINES.train_steps
    )
    # 평가 실행
    eval_result = classifier.evaluate(
        input_fn=lambda:data.eval_input_fn(eval_input_enc, eval_input_dec, eval_target_dec, DEFINES.batch_size)
    )
    print('\nEVAL set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
    # 테스트셋 인코딩 / 디코딩 입력 / 디코딩 출력

def predict(text) :
    word2idx, idx2word, vocab_len = data.load_vocabulary()
    classifier = tf.estimator.Estimator(
        model_fn = ml.model,
        model_dir = DEFINES.check_point_path,
        params= {
            'hidden_size': DEFINES.hidden_size,
            'layer_size' : DEFINES.layer_size,
            'learning_rate' : DEFINES.learning_rate,
            'vocab_len' : vocab_len,
            'embedding_size' : DEFINES.embedding_size,
            'embedding' : DEFINES.embedding,
            'multilayer': DEFINES.multilayer,
        }
    )
    # 테스트셋 인코딩 / 디코딩 입력 / 디코딩 출력
    predic_input_enc, predic_input_enc_length = data.enc_processing([text], word2idx)
    predic_input_dec, predic_input_dec_length = data.dec_input_processing([""], word2idx)
    predic_target_dec = data.dec_target_processing([""], word2idx)

    # 예측 실행
    predictions = classifier.predict(
        input_fn=lambda: data.eval_input_fn(predic_input_enc, predic_input_dec, predic_target_dec, DEFINES.batch_size))

    # 예측한 값을 텍스트로 변경하는 부분이다.
    data.pred2string(predictions, idx2word)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    #tf.app.run(train)
    predict("남자친구가 잘생겼어요")

tf.logging.set_verbosity