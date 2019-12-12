import json
import numpy as np
import modeling
import tokenization
import tensorflow as tf
from sklearn.model_selection import train_test_split

is_training = True    # 是否训练BERT参数
max_seq_length = 512  # 语句最大长度
epoch = 50            # 微调迭代轮数
batch_size = 5        # 微调batch_size
learning_rate = 1e-5  # 微调学习率
data_file = "train_data.json"  # 训练集地址
ckpt_file = "chinese_L-12_H-768_A-12/bert_model.ckpt"  # 模型地址
vocab_file = "chinese_L-12_H-768_A-12/vocab.txt"  # 字典地址
bert_config = modeling.BertConfig.from_json_file("chinese_L-12_H-768_A-12/bert_config.json")  # 配置地址
tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file)
input_ids = tf.placeholder(shape=[None, max_seq_length], dtype=tf.int32, name="input_ids")
input_mask = tf.placeholder(shape=[None, max_seq_length], dtype=tf.int32, name="input_mask")
segment_ids = tf.placeholder(shape=[None, max_seq_length], dtype=tf.int32, name="segment_ids")
input_labels = tf.placeholder(shape=batch_size, dtype=tf.int32, name="input_labels")


# 获取数据
def loadData():
    dic = {"国际件": 0, "业务咨询": 1, "查单": 2, "下单": 3}
    combined = []
    y = []
    num_classes = 4
    f = open(data_file, encoding='utf-8')
    data = json.load(f)
    for single in data:
        dialogs = single["dialog"]
        temp = ""
        for dialog in dialogs:
            temp += dialog["text"]
        combined.append(temp)
        type = single["tasks"]["intent"]
        y.append(dic[type])
    x_train, x_test, y_train, y_test = train_test_split(combined[:500], y[:500], test_size=0.25)
    return x_train, x_test, y_train, y_test


# 将序列对截断为最大长度
def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


# BERT输入标准化
def convert_single_example(max_seq_length, tokenizer, text_a, text_b=None):
    tokens_a = tokenizer.tokenize(text_a)
    tokens_b = None
    if text_b:
        tokens_b = tokenizer.tokenize(text_b)  # 中文分字
    if tokens_b:
        # 如果有第二个句子，那么两个句子的总长度要小于 max_seq_length - 3
        # 因为要为句子补上[CLS], [SEP], [SEP]
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # 如果只有一个句子，只用在前后加上[CLS], [SEP] 所以句子长度要小于 max_seq_length - 2
        if len(tokens_a) > max_seq_length - 2:
          tokens_a = tokens_a[0:(max_seq_length - 2)]

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)
    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    while len(input_ids) < max_seq_length:
        input_ids.append(0)  # input_ids：标记化文本的数字id列表
        input_mask.append(0)  # input_mask：对于真实标记将设置为1，对于填充标记将设置为0
        segment_ids.append(0)  # segment_ids：句子1赋值为0，句子2赋值为1
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    return input_ids, input_mask, segment_ids # 对应的就是创建bert模型时候的input_ids,input_mask,segment_ids 参数


# 获取BERT向量
def response_request(text):
    _input_ids = []
    _input_mask = []
    _segment_ids = []
    for t in text:
        if len(t) > 1:
            _input_ids_p, _input_mask_p, _segment_ids_p = convert_single_example(max_seq_length, tokenizer, t[0], t[1])
        else:
            _input_ids_p, _input_mask_p, _segment_ids_p = convert_single_example(max_seq_length, tokenizer, t)
        _input_ids.append(_input_ids_p)
        _input_mask.append(_input_mask_p)
        _segment_ids.append(_segment_ids_p)
    return _input_ids, _input_mask, _segment_ids


model = modeling.BertModel(
    config=bert_config,
    is_training=is_training,
    input_ids=input_ids,
    input_mask=input_mask,
    token_type_ids=segment_ids,
    use_one_hot_embeddings=False)

# 简单的使用了句子向量，获取字符向量用model.get_sequence_output()
output_layer = model.get_pooled_output()

# 接了一个简单的全连接层
layer = tf.layers.dense(output_layer, 256)
logits = tf.layers.dense(layer, 4)
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=input_labels, name="soft_loss")
loss = tf.reduce_mean(loss, name="loss")
predict = tf.argmax(tf.nn.softmax(logits), axis=1, name="predictions")
acc = tf.reduce_mean(tf.cast(tf.equal(input_labels, tf.cast(predict, dtype=tf.int32)), "float"), name="accuracy")
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# 获取模型中所有的训练参数
tvars = tf.trainable_variables()

# 加载BERT模型
assignment_map, initialized_variable_names = modeling.get_assignment_map_from_checkpoint(tvars, ckpt_file)
tf.train.init_from_checkpoint(ckpt_file, assignment_map)

if __name__ == "__main__":
    x_train, x_test, y_train, y_test = loadData()
    _input_ids, _input_mask, _segment_ids = response_request(x_train)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(epoch):
            shuffIndex = np.random.permutation(np.arange(len(x_train)))[:batch_size]
            batch_labels = np.array(y_train)[shuffIndex]
            batch_input_ids = np.array(_input_ids)[shuffIndex]
            batch_input_mask = np.array(_input_mask)[shuffIndex]
            batch_segment_ids = np.array(_segment_ids)[shuffIndex]
            l, a, _ = sess.run([loss, acc, train_op], feed_dict={
                input_ids: batch_input_ids, input_mask: batch_input_mask,
                segment_ids: batch_segment_ids,
                input_labels: batch_labels
            })
            print("Accracy:{}, Loss:{}".format(a, l))