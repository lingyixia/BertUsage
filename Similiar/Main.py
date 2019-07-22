# coding:utf-8

# -------------------------------------------------------------------------------
# @Author        chenfeiyu01
# @Name:         Main.py
# @Project       BertUsage
# @Product       PyCharm
# @DateTime:     2019-07-21 18:47
# @Contact       chenfeiyu01@baidu.com
# @Version       1.0
# @Description:  
# -------------------------------------------------------------------------------
from bert import run_classifier, tokenization, modeling, optimization
import tensorflow as tf
import os, argparse

parser = argparse.ArgumentParser(description="Bert相似度Demo")
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


class MyTextSimiliar(run_classifier.DataProcessor):
    def get_labels(self):
        return ["0", "1"]

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[0])
            text_b = tokenization.convert_to_unicode(line[1])
            label = tokenization.convert_to_unicode(line[2])
            examples.append(run_classifier.InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
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
    output_layer = model.get_pooled_output()
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
                accuracy = tf.metrics.accuracy(label_ids, predicted_labels)
                f1_score = tf.contrib.metrics.f1_score(label_ids, predicted_labels)
                auc = tf.metrics.auc(label_ids, predicted_labels)
                recall = tf.metrics.recall(label_ids, predicted_labels)
                precision = tf.metrics.precision(label_ids, predicted_labels)
                true_pos = tf.metrics.true_positives(label_ids, predicted_labels)
                true_neg = tf.metrics.true_negatives(label_ids, predicted_labels)
                false_pos = tf.metrics.false_positives(label_ids, predicted_labels)
                false_neg = tf.metrics.false_negatives(label_ids, predicted_labels)
                return {"eval_accuracy": accuracy,
                        "f1_score": f1_score,
                        "auc": auc,
                        "precision": precision,
                        "recall": recall,
                        "true_positives": true_pos,
                        "true_negatives": true_neg,
                        "false_positives": false_pos,
                        "false_negatives": false_neg}

            predicted_labels = tf.squeeze(tf.argmax(probabilities, axis=-1, output_type=tf.int32))
            eval_metrics = metric_fn(label_ids, predicted_labels)
            return tf.estimator.EstimatorSpec(mode=mode, loss=total_loss, eval_metric_ops=eval_metrics)
        else:
            predicted_labels = tf.squeeze(tf.argmax(probabilities, axis=-1, output_type=tf.int32))
            predictions = {'probabilities': probabilities, 'labels': predicted_labels}
            return tf.estimator.EstimatorSpec(mode=mode, predictions={"predictions": predictions})

    return model_fn


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS = parser.parse_args()
    myTextClassification = MyTextSimiliar()
    tokenizer = tokenization.FullTokenizer(vocab_file=os.path.join(FLAGS.bert_path, 'vocab.txt'),
                                           do_lower_case=True)
    label_list = myTextClassification.get_labels()
    train_examples = myTextClassification.get_train_examples(FLAGS.data_path)
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
    train_features = run_classifier.convert_examples_to_features(train_examples,
                                                                 label_list,
                                                                 FLAGS.max_length,
                                                                 tokenizer)
    train_input_fn = run_classifier.input_fn_builder(features=train_features,
                                                     seq_length=FLAGS.max_length,
                                                     is_training=True,
                                                     drop_remainder=False)
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
    eval_examples = myTextClassification.get_dev_examples(FLAGS.data)
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
    predict_examples = myTextClassification.get_test_examples(FLAGS.data)
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
