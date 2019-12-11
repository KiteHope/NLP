import json
import numpy as np
import modeling
import tokenization
import tensorflow as tf


max_seq_length = 512
batch_size = 2
ckpt_file = "chinese_L-12_H-768_A-12/bert_model.ckpt"  # 模型地址
vocab_file = "chinese_L-12_H-768_A-12/vocab.txt"  # 字典地址
bert_config = modeling.BertConfig.from_json_file("chinese_L-12_H-768_A-12/bert_config.json")  # 配置地址
tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file)
input_ids = tf.placeholder(shape=[batch_size, max_seq_length], dtype=tf.int32, name="input_ids")
input_mask = tf.placeholder(shape=[batch_size, max_seq_length], dtype=tf.int32, name="input_mask")
segment_ids = tf.placeholder(shape=[batch_size, max_seq_length], dtype=tf.int32, name="segment_ids")


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

    with tf.Session() as sess:
        tf.global_variables_initializer()
        model = modeling.BertModel(
            config=bert_config,
            is_training=False,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=False)
        saver = tf.train.Saver()
        saver.restore(sess, ckpt_file)
        # output_layer = model.get_sequence_output()
        # output_layer = model.get_pooled_output()
        # 获取的是字符级向量，并且包含首尾标记[CLS]和[SEP]向量，长度为768个神经元
        result = sess.run(model.get_pooled_output(), feed_dict={input_ids: _input_ids, input_mask: _input_mask, segment_ids: _segment_ids})
    return result


# 计算向量余弦相似性
def cosVector(x,y):
    if(len(x)!=len(y)):
        print('error input,x and y is not in the same space')
        return;
    result1=0.0;
    result2=0.0;
    result3=0.0;
    for i in range(len(x)):
        result1+=x[i]*y[i]   #sum(X*Y)
        result2+=x[i]**2     #sum(X*X)
        result3+=y[i]**2     #sum(Y*Y)
    return str(result1/((result2*result3)**0.5))


result = response_request(["今天天气挺好的", "今儿天很晴朗"])
print(cosVector(result[0], result[1]))
