import tensorflow as tf
import data_v2 as data
import sys
import model_v2 as ml
import os

from configs import DEFINES

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    arg_length = len(sys.argv)
    if(arg_length < 2):
        raise Exception("Don't call us. We'll call you")

    char2idx, idx2char, vocab_len = data.load_vocabulary()

    input = ""
    for i in sys.argv[1:]:
        input += i
        input += " "

    print(input)
    predic_input_enc, predic_input_enc_len = data.enc_processing([input], char2idx)
    predic_target_dec, _ = data.dec_target_processing([""], char2idx)

    if DEFINES.serving == True:
        predictor_fn = tf.contrib.predictor.from_saved_model(
            export_dir = os.path.join(BASE_DIR, "model", "xxxx")
        )

    else :
        classifier = tf.estimator.Estimator(
            model_fn=ml.model_v2,
            model_dir=DEFINES.check_point_path,
            params={
                'hidden_size': DEFINES.hidden_size,  # 가중치 크기 설정한다.
                'layer_size': DEFINES.layer_size,  # 멀티 레이어 층 개수를 설정한다.
                'learning_rate': DEFINES.learning_rate,  # 학습율 설정한다.
                'teacher_forcing_rate': DEFINES.teacher_forcing_rate,  # 학습시 디코더 인풋 정답 지원율 설정
                'vocab_len': vocab_len,  # 딕셔너리 크기를 설정한다.
                'embedding_size': DEFINES.embedding_size,  # 임베딩 크기를 설정한다.
                'embedding': DEFINES.embedding,  # 임베딩 사용 유무를 설정한다.
                'multilayer': DEFINES.multilayer,  # 멀티 레이어 사용 유무를 설정한다.
                'attention': DEFINES.attention,  # 어텐션 지원 유무를 설정한다.
                'teacher_forcing': DEFINES.teacher_forcing,  # 학습시 디코더 인풋 정답 지원 유무 설정한다.
                'loss_mask': DEFINES.loss_mask,  # PAD에 대한 마스크를 통한 loss를 제한 한다.
                'serving': DEFINES.serving  # 모델 저장 및 serving 유무를 설정한다.
            }
        )

    if DEFINES.serving == True :
        predictions = predictor_fn({'input':predic_input_enc, 'output':predic_target_dec})
        data.pred2string(predictions, idx2char)

    else :
        predictions = classifier.predict(
            input_fn=lambda: data.eval_input_fn(predic_input_enc, predic_target_dec, DEFINES.batch_size))
        data.pred2string(predictions, idx2char)
