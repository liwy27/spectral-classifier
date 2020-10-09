# -*- coding: utf-8 -*-
import tensorflow as tf
import logging
import sys
import pandas as pd


NUMERIC_COLUMNS = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
FEATURE_COLUMNS = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']
LABEL_NAME = 'Species'

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("train_paths", './data/iris_training.csv', "The train data ")
tf.app.flags.DEFINE_string("eval_paths", './data/iris_test.csv', "The eval data")
tf.app.flags.DEFINE_string("model_path", './model', "The output directory where the model checkpoints will be written.")
tf.app.flags.DEFINE_integer('batch_size', 6, 'Instance count in a batch.')
tf.app.flags.DEFINE_integer('train_num_epochs', 100, 'Iteration count over training data.')
tf.app.flags.DEFINE_integer("eval_num_epochs", 1, "Total batch size for eval.")
tf.app.flags.DEFINE_float("learning_rate", 0.0005, "The initial learning rate for Adam.")


feature_columns = []
for item in NUMERIC_COLUMNS:
    fc = tf.feature_column.numeric_column(item, dtype=tf.float32)
    feature_columns.append(fc)


def input_fn_builder(file_pattern, num_epochs):
    def input_fn():
        return tf.data.experimental.make_csv_dataset(
            file_pattern=file_pattern,
            select_columns=FEATURE_COLUMNS,
            label_name=LABEL_NAME,
            batch_size=FLAGS.batch_size,
            num_epochs=num_epochs,
            num_rows_for_inference=1,
            shuffle=True)
    return input_fn


def serving_input_receiver_fn():

    features = {}
    for item in NUMERIC_COLUMNS:
        features[item] = tf.compat.v1.placeholder(tf.float32, shape=[None], name=item)
    return tf.estimator.export.ServingInputReceiver(features, features)


def main(_):
    run_config = tf.estimator.RunConfig(keep_checkpoint_max=1, save_checkpoints_steps=10, log_step_count_steps=1)

    classifier_estimator = tf.estimator.DNNClassifier(
        hidden_units=[10, 10],
        feature_columns=feature_columns,
        model_dir=FLAGS.model_path,
        n_classes=3,
        optimizer='Adam',
        dropout=0.1,
        config=run_config
    )

    train_input_fn = input_fn_builder(FLAGS.train_paths, FLAGS.train_num_epochs)
    eval_input_fn = input_fn_builder(FLAGS.eval_paths, FLAGS.eval_num_epochs)

    # tf.estimator.train_and_evaluate(
    #     estimator=classifier_estimator,
    #     train_spec=tf.estimator.TrainSpec(input_fn=train_input_fn),
    #     eval_spec=tf.estimator.EvalSpec(input_fn=eval_input_fn)
    # )

    classifier_estimator.train(
        input_fn=train_input_fn,
        steps=5)

    if classifier_estimator.config.is_chief:
        classifier_estimator.export_saved_model(
            export_dir_base='./model/exported',
            serving_input_receiver_fn=serving_input_receiver_fn
        )


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)-15s [%(filename)s:%(lineno)d] %(levelname)s %(message)s',
        stream=sys.stdout
    )
    tf.app.run()
