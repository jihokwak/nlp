import tensorflow as tf
from configs import DEFINES

def make_lstm_cell(mode, hiddenSize, index) :
    cell = tf.nn.rnn_cell.BasicLSTMCell(hiddenSize, name='lstm'+str(index), state_is_tuple=False)
    if mode == tf.estimator.ModeKeys.TRAIN :
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=DEFINES.dropout_width)
    return cell

def model_v2(features, labels, mode, params) :
    TRAIN = mode == tf.estimator.ModeKeys.TRAIN
    EVAL = mode == tf.estimator.ModeKeys.EVAL
    PREDICT = mode == tf.estimator.ModeKeys.PREDICT

    if params['embedding'] == True : #미리 정의된 임베딩 벡터 사용유무
        # xavier (Xavier Glorot와 Yoshua Bengio (2010)
        # URL : http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
        initializer = tf.contrib.layers.xavier_initializer()
        embedding_encoder = tf.get_variable(name = "embedding_encoder",
                                    shape = [params['vocabulary_length'], params['embedding_size']],
                                    dtype = tf.float32,
                                    initializer=initializer,
                                    trainable=True)

    else :
        # tf.eye를 통해서 사전의 크기 만큼의 단위행렬 구조를 만든다.
        embedding_encoder = tf.eye(num_rows= params['vocabulary_length'], dtype=tf.float32)
        embedding_encoder = tf.get_variable(name = "embedding_encoder",
                                    initializer=embedding_encoder,
                                    trainable=False)

    embedding_encoder_batch = tf.nn.embedding_lookup(params= embedding_encoder, ids = features['input'])

    if params['embedding'] == True:
        initializer = tf.contrib.layers.xavier_initializer()
        embedding_decoder = tf.get_variable(name="embedding_decoder",  # 이름
                                            shape=[params['vocabulary_length'], params['embedding_size']],  # 모양
                                            dtype=tf.float32,  # 타입
                                            initializer=initializer,  # 초기화 값
                                            trainable=True)  # 학습 유무

    else:
        embedding_decoder = tf.eye(num_rows=params['vocabulary_length'], dtype=tf.float32)
        embedding_decoder = tf.get_variable(name='embedding_decoder',  # 이름
                                            initializer=embedding_decoder,  # 초기화 값
                                            trainable=False)  # 학습 유무


    with tf.variable_scope('encoder_scope', reuse=tf.AUTO_REUSE):
        if params['multilayer'] == True : # 값이 True이면 멀티레이어,  False 이면 단일레이어
            encoder_cell_list = [make_lstm_cell(mode, params['hidden_size'], i) for i in range(params['layer_size'])]
            rnn_cell = tf.contrib.rnn.MultiRNNCell(encoder_cell_list, state_is_tuple=False)
        else :
            rnn_cell = make_lstm_cell(mode, params['hidden_size'], "")
        # encoder_states 최종 상태  [batch_size, cell.state_size]
        encoder_outputs, encoder_states = tf.nn.dynamic_rnn(cell=rnn_cell,
                                                            inputs=embedding_encoder_batch,
                                                            dtype=tf.float32)

    with tf.variable_scope('decoder_scope', reuse=tf.AUTO_REUSE):
        if params['multilayer'] == True :
            decoder_cell_list = [make_lstm_cell(mode, params['hidden_size'], i) for i in range(params['layer_size'])]
            rnn_cell = tf.contrib.rnn.MultiRNNCell(decoder_cell_list, state_is_tuple=False)
        else :
            rnn_cell = make_lstm_cell(mode, params['hidden_size'], "")

        decoder_state = encoder_states # 인코딩의 마지막 값으로 초기화


        predict_tokens = list() # 하나는 토큰 인덱스는 predict_tokens 저장
        temp_logits = list() # 다른 하나는 temp_logits에 logits 저장한다.
        # 평가인 경우에는 teacher forcing이 되지 않도록 해야한다.
        # 따라서 학습이 아닌경우에 is_train을 False로 하여 teacher forcing이 되지 않도록 한다.
        output_token = tf.ones(shape=(tf.shape(encoder_outputs)[0],), dtype=tf.int32) * 1
        # 전체 문장 길이 만큼 타임 스텝을 돌도록 한다.
        for i in range(DEFINES.max_sequence_length):
    # 두 번쨰 스텝 이후에는 teacher forcing을 적용하는지 확률에 따라 결정하도록 한다.
    # teacher forcing rate은 teacher forcing을 어느정도 줄 것인지를 조절한다.
            if TRAIN:
                if i > 0:
                    input_token_emb = tf.cond(
                        tf.logical_and(
                            True,
                            tf.random_uniform(shape=(), maxval=1) <= params['teacher_forcing_rate']# 률에 따른 labels값 지원 유무
                        ),
                        lambda: tf.nn.embedding_lookup(embedding_encoder, labels[:, i-1]), #레이블 정답을 넣어주고 있음.
                        lambda: tf.nn.embedding_lookup(embedding_encoder, output_token) # 모델이 정답이라고 생각 하는
                    )
                else :
                    input_token_emb = tf.nn.embedding_lookup(embedding_encoder, output_token) # 모델이 정답이라고 생각 하는 값
            else :
                input_token_emb = tf.nn.embedding_lookup(embedding_encoder, output_token)
            #어텐션 적용 부분
            if params['attention'] == True:
                W1 = tf.keras.layers.Dense(params['hidden_size'])
                W2 = tf.keras.layers.Dense(params['hidden_size'])
                V = tf.keras.layers.Dense(1)
                # (?, 256) -> (?, 128)
                hidden_with_time_axis = W2(decoder_state)
                # (?, 128) -> (?, 1, 128)
                hidden_with_time_axis = tf.expand_dims(hidden_with_time_axis, axis=1)
                # (?, 1, 128) -> (?, 25, 128)
                hidden_with_time_axis = tf.manip.tile(hidden_with_time_axis, [1, DEFINES.max_sequence_length, 1])
                # (?, 25, 1)
                score = V(tf.nn.tanh(W1(encoder_outputs) + hidden_with_time_axis))
                # score = V(tf.nn.tanh(W1(encoderOutputs) + tf.manip.tile(tf.expand_dims(W2(decoder_state), axis=1), [1, DEFINES.maxSequenceLength, 1])))
                # (?, 25, 1)
                attention_weights = tf.nn.softmax(score, axis=-1)
                # (?, 25, 128)
                context_vector = attention_weights * encoder_outputs
                # (?, 25, 128) -> (?, 128)
                context_vector = tf.reduce_sum(context_vector, axis=1)
                # (?, 256)
                input_token_emb = tf.concat([context_vector, input_token_emb], axis=-1)

            # RNNCell을 호출하여 RNN 스텝 연산을 진행하도록 한다.
            input_token_emb = tf.keras.layers.Dropout(0.5)(input_token_emb)
            decoder_outputs, decoder_state = rnn_cell(input_token_emb, decoder_state)
            decoder_outputs = tf.keras.layers.Dropout(0.5)(decoder_outputs)
            # feedforward를 거쳐 output에 대한 logit 값을 구한다.
            output_logits = tf.layers.dense(decoder_outputs, params['vocabulary_length'], activation=None)

            # softmax를 통해 단어에 대한 예측 probability를 구한다.
            output_probs = tf.nn.softmax(output_logits)
            output_token = tf.argmax(output_probs, axis=-1)

            # 한 스텝에 나온 토큰과 logit 결과를 저장해둔다.
            predict_tokens.append(output_token)
            temp_logits.append(output_logits)

        #저장했던 토큰과 logit 리스트를 stack을 통해 매트릭스로 만들어 준다.
        #만들게 되면 차원이 [시퀀스 x 배치 x 단어 feature 수] 이렇게 되는데
        #이를 transpose하여 [배치 x 시퀀스 x 단어 feature 수] 로 맞춰준다.
        predict = tf.transpose(tf.stack(predict_tokens, axis=0), [1,0])
        logits = tf.transpose(tf.stack(temp_logits, axis=0), [1,0,2])

        print(predict.shape)
        print(logits.shape)

    if PREDICT :
        if params['serving'] == True :
            export_outputs = { # 서빙 결과값을 준다.
                'indexs' : tf.estimator.export.PredictOutput(predict)
            }

        predictions = {
            'indexs':predict, # 시퀀스 마다 예측한 값
            'logits':logits # 마지막 결과 값
        }
        # 에스티메이터에서 리턴하는 값은 tf.estimator.EstimatorSpec 객체임.

        if params['serving'] == True:
            return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)
        return tf.estimator.EstimatorSpec(mode, predictions)
    # logits과 같은 차원을 만들어 마지막 결과 값과 정답 값을 비교하여 에러를 구한다.
    labels_ = tf.one_hot(labels, params['vocabulary_length'])

    if TRAIN and params['loss_mask'] == True :
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels_))
        masks = features['length']

        loss = loss * tf.cast(masks, tf.float32)
        loss = tf.reduce_mean(loss)
    else :
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels_))
    # 라벨과 결과가 일치하는지 빈도 계산을 통해 정확도를 측정. accuracy를 전체 값으로 나눠 확률로 출력
    accuracy = tf.metrics.accuracy(labels=labels, predictions=predict, name='acc0p')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if EVAL :
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    assert TRAIN
    optimizer  = tf.train.AdamOptimizer(learning_rate=DEFINES.learning_rate)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
