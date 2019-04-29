import tensorflow as tf
from configs import DEFINES

def make_lstm_cell(mode, hiddenSize, index) :
    cell = tf.nn.rnn_cell.BasicLSTMCell(hiddenSize, name='lstm'+str(index))
    if mode == tf.estimator.ModeKeys.TRAIN :
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=DEFINES.dropout_width)
    return cell

def model(features, labels, mode, params) :
    TRAIN = mode == tf.estimator.ModeKeys.TRAIN
    EVAL = mode == tf.estimator.ModeKeys.EVAL
    PREDICT = mode == tf.estimator.ModeKeys.PREDICT

    if params['embedding'] == True : #미리 정의된 임베딩 벡터 사용유무
        # xavier (Xavier Glorot와 Yoshua Bengio (2010)
        # URL : http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
        initializer = tf.contrib.layers.xavier_initializer()
        embedding = tf.get_variable(name = "embedding",
                                            shape = [params['vocab_len'], params['embedding_size']],
                                            dtype = tf.float32,
                                            initializer=initializer,
                                            trainable=True)

    else :
        # tf.eye를 통해서 사전의 크기 만큼의 단위행렬 구조를 만든다.
        embedding = tf.eye(num_rows= params['vocab_len'], dtype=tf.float32)
        embedding = tf.get_variable(name = "embedding",
                                            initializer=embedding,
                                            trainable=False)

    embedding_encoder = tf.nn.embedding_lookup(params= embedding, ids = features['input'])
    embedding_decoder = tf.nn.embedding_lookup(params= embedding, ids = features['output'])

    with tf.variable_scope('encoder_scope', reuse=tf.AUTO_REUSE):
        if params['multilayer'] == True : # 값이 True이면 멀티레이어,  False 이면 단일레이어
            encoder_cell_list = [make_lstm_cell(mode, params['hidden_size'], i) for i in range(params['layer_size'])]
            rnn_cell = tf.contrib.rnn.MultiRNNCell(encoder_cell_list)
        else :
            rnn_cell = make_lstm_cell(mode, params['hidden_size'], "")
        # encoder_states 최종 상태  [batch_size, cell.state_size]
        encoder_outputs, encoder_states = tf.nn.dynamic_rnn(cell=rnn_cell,
                                                            inputs=embedding_encoder,
                                                            dtype=tf.float32)

    with tf.variable_scope('decoder_scope', reuse=tf.AUTO_REUSE):
        if params['multilayer'] == True :
            decoder_cell_list = [make_lstm_cell(mode, params['hidden_size'], i) for i in range(params['layer_size'])]
            rnn_cell = tf.contrib.rnn.MultiRNNCell(decoder_cell_list)
        else :
            rnn_cell = make_lstm_cell(mode, params['hidden_size'], "")

        decoder_initial_state = encoder_states # 인코딩의 마지막 값으로 초기화
        decoder_outputs, decoder_states = tf.nn.dynamic_rnn(cell=rnn_cell,
                                                            inputs=embedding_decoder,
                                                            initial_state=decoder_initial_state,
                                                            dtype=tf.float32)
    # logits는 마지막 히든레이어를 통과한 결과값
    logits = tf.layers.dense(decoder_outputs, params['vocab_len'], activation=None)
    # argmax를 통해서 최대 값 추출
    predict = tf.argmax(logits, 2)

    if PREDICT :
        predictions = {'indexs':predict}
        return tf.estimator.EstimatorSpec(mode, predictions)
    # logits과 같은 차원을 만들어 마지막 결과 값과 정답 값을 비교하여 에러를 구한다.
    labels_ = tf.one_hot(labels, params['vocab_len'])
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels_))
    # 라벨과 결과가 일치하는지 빈도 계산을 통해 정확도를 측정. accuracy를 전체 값으로 나눠 확률로 출력
    accuracy = tf.metrics.accuracy(labels=labels, predictions=predict, name='accOp')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if EVAL :
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    assert TRAIN
    optimizer  = tf.train.AdamOptimizer(learning_rate=DEFINES.learning_rate)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
