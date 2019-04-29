import tensorflow as tf
import os
import model_v2 as ml
import data_v2 as data
from configs import DEFINES

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_OUT_PATH = 'data_out'

def serving_input_receiver_fn():
    receiver_tensor = {
        'input': tf.placeholder(dtype=tf.int32, shape=[None, DEFINES.max_sequence_length]),
        'output': tf.placeholder(dtype=tf.int32, shape=[None, DEFINES.max_sequence_length])
    }
    features = {
        key : tensor for key, tensor in receiver_tensor.items()
    }
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensor)

def main(self) :
    data_out_path = os.path.join(BASE_DIR, DATA_OUT_PATH)
    os.makedirs(data_out_path, exist_ok=True)
    # 데이터를 통한 사전 구성
    char2idx, idx2char, vocab_len = data.load_vocabulary()

    train_input, train_label, eval_input, eval_label = data.load_data()
    # 훈련셋 인코딩 / 디코딩 출력
    train_input_enc, train_input_enc_len = data.enc_processing(train_input, char2idx)
    train_target_dec, train_target_dec_len = data.dec_target_processing(train_label, char2idx)
    # 평가셋 인코딩 / 디코딩 출력
    eval_input_enc, eval_input_enc_len = data.enc_processing(eval_input, char2idx)
    eval_target_dec, _ = data.dec_target_processing(eval_label, char2idx)

    os.makedirs(DEFINES.check_point_path, exist_ok=True)
    os.makedirs(DEFINES.save_model_path, exist_ok=True)

    classifier = tf.estimator.Estimator(
        model_fn = ml.model_v2,
        model_dir = DEFINES.check_point_path,
        params= {
            'hidden_size': DEFINES.hidden_size,
            'layer_size' : DEFINES.layer_size,
            'learning_rate' : DEFINES.learning_rate,
            'teacher_forcing_rate': DEFINES.teacher_forcing_rate,
            'vocab_len' : vocab_len,
            'embedding_size' : DEFINES.embedding_size,
            'embedding' : DEFINES.embedding,
            'multilayer': DEFINES.multilayer,
            'attention' : DEFINES.attention,
            'teacher_forcing' : DEFINES.teacher_forcing,
            'loss_mask' : DEFINES.loss_mask,
            'serving' : DEFINES.serving
        }
    )
    # 학습 실행
    classifier.train(input_fn=lambda: data.train_input_fn(
        train_input_enc, train_target_dec_len, train_target_dec, DEFINES.batch_size), steps=DEFINES.train_steps)
    # 서빙 기능 유무에 따라 모델을 Save 한다.
    if DEFINES.serving == True:
        save_model_path = classifier.export_savedmodel(
            export_dir_base=DEFINES.save_model_path,
            serving_input_receiver_fn=serving_input_receiver_fn
        )
    # 평가 실행
    eval_result = classifier.evaluate(
        input_fn=lambda:data.eval_input_fn(
            eval_input_enc, eval_target_dec, DEFINES.batch_size
        )
    )
    print('\nEVAL set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
    # 테스트셋 인코딩 / 디코딩 입력 / 디코딩 출력

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)

tf.logging.set_verbosity