import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--task', default='yelp', choices=['yelp', 'yahoo'])
parser.add_argument('--mode', default='train', choices=['train', 'eval', 'test', 'visual'])
parser.add_argument('--checkpoint-frequency', type=int, default=100)
parser.add_argument('--eval-frequency', type=int, default=10000)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument("--device", default="/cpu:0")
parser.add_argument("--max-grad-norm", type=float, default=5.0)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--batch-index", type=int, default=0)
args = parser.parse_args()

import importlib
import os
import pickle
import random
import time
import math
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import spacy
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from tqdm import tqdm

import ujson
from data_util import batch

task_name = args.task

task = importlib.import_module(task_name)   # default: task := yelp

checkpoint_dir = os.path.join(task.train_dir, 'checkpoint')
tflog_dir = os.path.join(task.train_dir, 'tflog')
checkpoint_name = task_name + '-model'
checkpoint_dir = os.path.join(task.train_dir, 'checkpoints')
checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

# @TODO: move calculation into `task file`
trainset = task.read_trainset(epochs=1)
class_weights = pd.Series(Counter([l for _, l in trainset]))
class_weights = 1 / (class_weights / class_weights.mean())
class_weights = class_weights.to_dict()

vocab = task.read_vocab()
labels = task.read_labels()

classes = max(labels.values()) + 1      # labels: 0~4, classes: 0~5
vocab_size = task.get_vocab_size()
# vocab_size = len(vocab)

labels_rev = {int(v): k for k, v in labels.items()}
vocab_rev = {int(v): k for k, v in vocab.items()}


def HAN_model_fnc(session, restore_only=False):
    """Hierarhical Attention Network"""
    import tensorflow as tf
    try:
        from tensorflow.contrib.rnn import GRUCell, MultiRNNCell, DropoutWrapper
    except ImportError:
        MultiRNNCell = tf.nn.rnn_cell.MultiRNNCell
        GRUCell = tf.nn.rnn_cell.GRUCell
    from HAN_model import HANClassifierModel

    is_training = tf.placeholder(dtype=tf.bool, name='is_training')

    word_cell = GRUCell(50)
    sentence_cell = GRUCell(50)
    #cell = MultiRNNCell([cell] * 5)     # multi-layer RNN

    model = HANClassifierModel(
        vocab_size=vocab_size,
        embedding_size=200,
        classes=classes,
        word_cell=word_cell,
        sentence_cell=sentence_cell,
        device=args.device,
        learning_rate=args.lr,
        max_grad_norm=args.max_grad_norm,
        dropout_keep_proba=0.5,
        is_training=is_training,
    )

    saver = tf.train.Saver(tf.global_variables())
    checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
    if checkpoint:
        print("Reading model parameters from %s" % checkpoint.model_checkpoint_path)
        saver.restore(session, checkpoint.model_checkpoint_path)
    elif restore_only:
        raise FileNotFoundError("Cannot restore model")
    else:
        print("Created model with fresh parameters")
        session.run(tf.global_variables_initializer())
    # tf.get_default_graph().finalize()
    return model, saver


model_fn = HAN_model_fnc


def decode(ex):
    print('text: ' + '\n'.join([' '.join([vocab_rev.get(wid, '<?>') for wid in sent]) for sent in ex[0]]))
    print('label: ', labels_rev[ex[1]])

def decode(tokens, label):
    print('text: ' + '\n'.join([' '.join([vocab_rev.get(wid, '<?>') for wid in sent]) for sent in tokens]))
    print('label: ', labels_rev[label])


print('data loaded')


def batch_iterator(dataset, batch_size, max_epochs):
    for i in range(max_epochs):
        xb = []
        yb = []
        for ex in dataset:
            x, y = ex
            xb.append(x)
            yb.append(y)
            if len(xb) == batch_size:
                yield xb, yb
                xb, yb = [], []


def ev(session, model, dataset):
    predictions = []
    labels = []
    examples = []
    for x, y in tqdm(batch_iterator(dataset, args.batch_size, 1)):
        examples.extend(x)
        labels.extend(y)
        predictions.extend(session.run(model.prediction, model.get_feed_data(x, is_training=False)))

    df = pd.DataFrame({'predictions': predictions, 'labels': labels, 'examples': examples})
    return df


def evaluate(dataset):
    tf.reset_default_graph()
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        model, _ = model_fn(sess, restore_only=True)
        df = ev(sess, model, dataset)
    print((df['predictions'] == df['labels']).mean())
    # import IPython
    # IPython.embed()


def visualize(dataset, index=0):
    tf.reset_default_graph()
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        model, _ = model_fn(sess, restore_only=True)
        for i, (x, y) in enumerate(batch_iterator(dataset, args.batch_size, 1)):
            if(i == index):
                predictions, words_attentions, sentences_attentions = sess.run(
                    [model.prediction, model.words_attentions, model.sentences_attentions],
                    feed_dict=model.get_feed_data(x, is_training=False)
                )

                visualDiv = ""
                for idx in range(args.batch_size):
                    decode(tokens=x[idx], label=y[idx])
                    print('predict: ', labels_rev[predictions[idx]])

                    for sent_idx, sentence_tokens in enumerate(x[idx]):
                        # sent_color = int(math.pow(sentences_attentions[idx][sent_idx] * 35, 2) + 100)
                        sent_color = str(int(sentences_attentions[idx][sent_idx] * 155 * 1.5 + 100))
                        visualDiv += "<div style='background-color: rgb(%s, %s, %s)' title='sentence attention = %s'>\n" \
                                     % (sent_color,
                                        sent_color,
                                        sent_color,
                                        str(sentences_attentions[idx][sent_idx])
                                        )
                        for word_idx, word_token in enumerate(sentence_tokens):
                            word = vocab_rev.get(word_token, '<?>')
                            print(word + '('+str(words_attentions[idx][sent_idx][word_idx])+') ', end='')
                            # print(word + '[' + str(word_token) + ']', end='')
                            visualDiv += "\t<span style='color: rgb(%s, 0, 0)' title='word attention = %s'>%s</span>\n" \
                                         % (str(int(words_attentions[idx][sent_idx][word_idx] * 4000)),
                                            str(words_attentions[idx][sent_idx][word_idx]),
                                            word)
                        print('\n==> sentence['+str(sent_idx)+'](' + str(sentences_attentions[idx][sent_idx]) + ')')
                        visualDiv += "</div>\n"
                    # print('words attentions: ', words_attentions[idx])
                    # print('sentences attentions: ', sentences_attentions[idx])
                    print("==========================================")
                    visualDiv += "<div><i>label: %s</i></div>\n" % labels_rev[y[idx]]
                    visualDiv += "<div>predict: %s</div>\n" % labels_rev[predictions[idx]]
                    visualDiv += "<hr />\n"

                with open('visualize-%s.html' % args.task, mode='w', encoding='utf-8') as page:
                    htmlStr = """
                        <!DOCTYPE html>
                        <html lang="en">
                        <head>
                            <meta charset="UTF-8">
                            <title>Visualize</title>
                        </head>
                        <body>
                            %s
                        </body>
                        </html>
                        """ % visualDiv
                    page.write(htmlStr)
                break

def train():
    tf.reset_default_graph()

    config = tf.ConfigProto(allow_soft_placement=True)

    with tf.Session(config=config) as s:
        model, saver = model_fn(s)
        summary_writer = tf.summary.FileWriter(tflog_dir, graph=tf.get_default_graph())

        for i, (x, y) in enumerate(batch_iterator(task.read_trainset(epochs=1), batch_size=args.batch_size, max_epochs=300)):
            fd = model.get_feed_data(x, y, class_weights=class_weights)     # {self.inputs: x_m,self.sentence_lengths: doc_sizes,self.word_lengths: sent_sizes}

            # import IPython
            # IPython.embed()

            t0 = time.clock()
            step, summaries, loss, accuracy, learning_rate, _ = s.run([
                model.global_step,
                model.summary_op,
                model.loss,
                model.accuracy,
                model.learning_rate,
                model.train_op,
            ], feed_dict=fd)
            td = time.clock() - t0

            summary_writer.add_summary(summaries, global_step=step)

            if step % 1 == 0:
                print('step %s, loss=%s, accuracy=%s, t=%s, inputs=%s, lr=%s' % (
                step, loss, accuracy, round(td, 2), fd[model.inputs].shape, learning_rate))
            if step != 0 and step % args.checkpoint_frequency == 0:
                print('checkpoint & graph meta')
                saver.save(s, checkpoint_path, global_step=step)
                print('checkpoint done')
            if step != 0 and step % args.eval_frequency == 0:
                print('evaluation at step %s' % i)
                dev_df = ev(s, model, task.read_devset(epochs=1))
                print('dev accuracy: %.2f' % (dev_df['predictions'] == dev_df['labels']).mean())


def main():
    if args.mode == 'train':
        train()
    elif args.mode == 'eval':
        evaluate(task.read_devset(epochs=1))
    elif args.mode == 'test':
        evaluate(task.read_testset(epochs=1))
    elif args.mode == 'visual':
        visualize(task.read_trainset(epochs=1), index=args.batch_index)


if __name__ == '__main__':
    main()
