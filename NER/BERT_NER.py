# coding:utf-8

# -------------------------------------------------------------------------------
# @Author        chenfeiyu01
# @Name:         BERT_NER.py
# @Project       BertUsage
# @Product       PyCharm
# @DateTime:     2019-07-22 11:30
# @Contact       chenfeiyu01@baidu.com
# @Version       1.0
# @Description:  
# -------------------------------------------------------------------------------
from bert import run_classifier, tokenization, modeling, optimization
import tensorflow as tf
import os, argparse
import tf_metrics

parser = argparse.ArgumentParser(description="Bert NER Demo")
parser.add_argument('--max_length', type=int, default=128, help='序列最大长度')
parser.add_argument('--batch_size', type=int, default=32, help='序列最大长度')
parser.add_argument('--learning_rate', type=float, default=2e-5, help='学习率')
parser.add_argument('--num_train_epochs', type=int, default=3, help='epochs')
parser.add_argument('--warmup_proportion', type=float, default=0.1, help='warmup_proportion')
parser.add_argument('--save_checkpoints_steps', type=int, default=500, help='save_checkpoints_steps')
parser.add_argument('--save_summary_steps', type=int, default=100, help='save_summary_steps')
parser.add_argument('--data_path', type=str, default="data", help='数据路径')
parser.add_argument('--model_path', type=str, default="model", help='模型保存路径')
parser.add_argument('--bert_path', type=str, default="/Users/chenfeiyu01/Downloads/uncased_L-12_H-768_A-12",
                    help='bert模型位置')


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    features = list()
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))
        feature = convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer)
        features.append(feature)
    return features


class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids


def input_fn_builder(features, seq_length, is_training, drop_remainder):
    all_input_ids = list()
    all_input_mask = list()
    all_segment_ids = list()
    all_label_ids = list()

    for feature in features:
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_segment_ids.append(feature.segment_ids)
        all_label_ids.append(feature.label_ids)

    def input_fn(params):
        batch_size = params["batch_size"]
        num_examples = len(features)
        d = tf.data.Dataset.from_tensor_slices(
            {"input_ids": tf.constant(all_input_ids, shape=[num_examples, seq_length], dtype=tf.int32),
             "input_mask": tf.constant(all_input_mask, shape=[num_examples, seq_length], dtype=tf.int32),
             "segment_ids": tf.constant(all_segment_ids, shape=[num_examples, seq_length], dtype=tf.int32),
             "label_ids": tf.constant(all_label_ids, shape=[num_examples, seq_length], dtype=tf.int32),
             })
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)
        d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
        return d

    return input_fn


def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer):
    textlist = example.text_a.split(' ')
    labellist = example.label.split(' ')
    tokens = list()
    labels = list()
    label_map = dict()
    for (i, label) in enumerate(label_list):
        label_map[label] = i
    for i, word in enumerate(textlist):
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        label_1 = labellist[i]
        for m in range(len(token)):
            if m == 0:
                labels.append(label_1)
            else:
                labels.append("X")
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]
        labels = labels[0:(max_seq_length - 2)]
    ntokens = list()
    segment_ids = list()
    label_ids = list()
    ntokens.append("[CLS]")
    segment_ids.append(0)
    label_ids.append(label_map["[CLS]"])
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(label_map[labels[i]])
    ntokens.append("[SEP]")
    segment_ids.append(0)
    label_ids.append(label_map["[SEP]"])
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    input_mask = [1] * len(input_ids)
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        label_ids.append(0)
        ntokens.append("**NULL**")
    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
    feature = InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, label_ids=label_ids)
    return feature


class DataProcessor(object):
    def get_train_examples(self, data_dir):
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        raise NotImplementedError()

    def get_labels(self):
        raise NotImplementedError()

    @classmethod
    def _read_data(cls, input_file):
        with open(input_file) as f:
            lines = list()
            words = list()
            labels = list()
            for line in f:
                contends = line.strip()
                word = line.strip().split(' ')[0]
                label = line.strip().split(' ')[-1]
                if contends.startswith("-DOCSTART-"):
                    words.append('')
                    continue
                if len(contends) == 0:
                    l = ' '.join([label for label in labels if len(label) > 0])
                    w = ' '.join([word for word in words if len(word) > 0])
                    lines.append([l, w])
                    words = []
                    labels = []
                    continue
                words.append(word)
                labels.append(label)
            return lines


class NerProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_example(self._read_data(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_example(self._read_data(os.path.join(data_dir, "dev.txt")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_example(self._read_data(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self):
        return ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X", "[CLS]", "[SEP]"]

    def _create_example(self, lines, set_type):
        examples = list()
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = tokenization.convert_to_unicode(line[1])
            label = tokenization.convert_to_unicode(line[0])
            examples.append(run_classifier.InputExample(guid=guid, text_a=text, label=label))
        return examples


def create_model(bert_config,
                 is_training,
                 input_ids,
                 input_mask,
                 segment_ids,
                 labels,
                 num_labels,
                 use_one_hot_embeddings):
    model = modeling.BertModel(config=bert_config,
                               is_training=is_training,
                               input_ids=input_ids,
                               input_mask=input_mask,
                               token_type_ids=segment_ids,
                               use_one_hot_embeddings=use_one_hot_embeddings)
    output_layer = model.get_sequence_output()
    hidden_size = output_layer.shape[-1].value
    output_weights = tf.get_variable("output_weights",
                                     [num_labels, hidden_size],
                                     initializer=tf.truncated_normal_initializer(stddev=0.02))
    output_bias = tf.get_variable("output_bias", [num_labels], initializer=tf.zeros_initializer())
    with tf.variable_scope("loss"):
        if is_training:
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        probabilities = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)
        return (loss, per_example_loss, logits, probabilities)


def model_fn_builder(bert_config,
                     num_labels,
                     learning_rate,
                     num_train_steps,
                     num_warmup_steps,
                     use_one_hot_embeddings):
    def model_fn(features, labels, mode, params):
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        (total_loss, per_example_loss, logits, probabilities) = create_model(bert_config,
                                                                             is_training,
                                                                             input_ids,
                                                                             input_mask,
                                                                             segment_ids,
                                                                             label_ids,
                                                                             num_labels,
                                                                             use_one_hot_embeddings)
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(total_loss,
                                                     learning_rate,
                                                     num_train_steps,
                                                     num_warmup_steps,
                                                     False)
            return tf.estimator.EstimatorSpec(mode=mode, loss=total_loss, train_op=train_op)
        elif mode == tf.estimator.ModeKeys.EVAL:
            def metric_fn(label_ids, predicted_labels):
                precision = tf_metrics.precision(label_ids, predicted_labels, 11, [2, 3, 4, 5, 6, 7], average="macro")
                recall = tf_metrics.recall(label_ids, predicted_labels, 11, [2, 3, 4, 5, 6, 7], average="macro")
                f1 = tf_metrics.f1(label_ids, predicted_labels, 11, [2, 3, 4, 5, 6, 7], average="macro")
                return {
                    "eval_precision": precision,
                    "eval_recall": recall,
                    "eval_f": f1,
                }

            # predicted_labels = tf.squeeze(tf.argmax(probabilities, axis=-1, output_type=tf.int32))
            predicted_labels = tf.argmax(probabilities, axis=-1, output_type=tf.int32)
            eval_metrics = metric_fn(label_ids, predicted_labels)
            return tf.estimator.EstimatorSpec(mode=mode, loss=total_loss, eval_metric_ops=eval_metrics)
        else:
            predicted_labels = tf.squeeze(tf.argmax(probabilities, axis=-1, output_type=tf.int32))
            predictions = {'predicted_labels': predicted_labels, 'labels': labels}
            return tf.estimator.EstimatorSpec(mode=mode, predictions={"predictions": predictions})

    return model_fn


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS = parser.parse_args()
    nerProcessor = NerProcessor()
    tokenizer = tokenization.FullTokenizer(vocab_file=os.path.join(FLAGS.bert_path, 'vocab.txt'),
                                           do_lower_case=True)
    label_list = nerProcessor.get_labels()
    train_examples = nerProcessor.get_train_examples(FLAGS.data_path)
    num_train_steps = int(len(train_examples) / FLAGS.batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
    bert_config = modeling.BertConfig.from_json_file(os.path.join(FLAGS.bert_path, 'bert_config.json'))
    model_fn = model_fn_builder(bert_config=bert_config,
                                num_labels=len(label_list),
                                learning_rate=FLAGS.learning_rate,
                                num_train_steps=num_train_steps,
                                num_warmup_steps=num_warmup_steps,
                                use_one_hot_embeddings=False)
    run_config = tf.estimator.RunConfig(model_dir=FLAGS.model_path,
                                        save_summary_steps=FLAGS.save_summary_steps,
                                        save_checkpoints_steps=FLAGS.save_checkpoints_steps)
    estimator = tf.estimator.Estimator(model_fn=model_fn,
                                       config=run_config,
                                       params={"batch_size": FLAGS.batch_size})
    train_features = convert_examples_to_features(train_examples,
                                                  label_list,
                                                  FLAGS.max_length,
                                                  tokenizer)
    train_input_fn = input_fn_builder(features=train_features,
                                      seq_length=FLAGS.max_length,
                                      is_training=True,
                                      drop_remainder=False)
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
    eval_examples = nerProcessor.get_dev_examples(FLAGS.data)
    eval_steps = int(len(eval_examples) // FLAGS.batch_size)
    dev_features = run_classifier.convert_examples_to_features(eval_examples,
                                                               label_list,
                                                               FLAGS.max_length,
                                                               tokenizer)
    dev_input_fn = run_classifier.input_fn_builder(features=dev_features,
                                                   seq_length=FLAGS.max_length,
                                                   is_training=True,
                                                   drop_remainder=False)
    estimator.evaluate(dev_input_fn, eval_steps)
    predict_examples = nerProcessor.get_test_examples(FLAGS.data)
    predict_steps = int(len(eval_examples) // FLAGS.batch_size)
    predict_features = run_classifier.convert_examples_to_features(predict_examples,
                                                                   label_list,
                                                                   FLAGS.max_length,
                                                                   tokenizer)
    predict_input_fn = run_classifier.input_fn_builder(features=predict_features,
                                                       seq_length=FLAGS.max_length,
                                                       is_training=True,
                                                       drop_remainder=False)
    result = estimator.predict(predict_input_fn)
