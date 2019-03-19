# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A demo script showing how to use the uisrnn package on toy data."""

import numpy as np

import uisrnn


SAVED_MODEL_NAME = 'saved_model.uisrnn'


def diarization_experiment(model_args, training_args, inference_args):
  """Experiment pipeline.

  Load data --> train model --> test model --> output result

  Args:
    model_args: model configurations
    training_args: training configurations
    inference_args: inference configurations
  """
  predicted_cluster_ids = []
  test_record = []
  
  # train_data = np.load('./data/toy_training_data.npz')
  # test_data = np.load('./data/toy_testing_data.npz')
  # train_sequence = train_data['train_sequence']
  # train_cluster_id = train_data['train_cluster_id']
  # test_sequences = test_data['test_sequences'].tolist()
  # test_cluster_ids = test_data['test_cluster_ids'].tolist()
  train_sequence = np.load('data/train_sequence.npy').astype(np.float64)
  train_cluster_id = np.load('data/train_cluster_id.npy')
  test_sequences = np.load('data/test_sequence.npy').astype(np.float64)
  test_cluster_ids = np.load('data/test_cluster_id.npy')

  chunk_size = test_sequences.shape[0] // 86
  left_over = test_sequences.shape[0] % chunk_size
  new_len = test_sequences.shape[0] - left_over

  test_sequences = np.split(test_sequences[:new_len], chunk_size)
  test_cluster_ids = np.split(test_cluster_ids[:new_len], chunk_size)

  model = uisrnn.UISRNN(model_args)

  # training
  model.fit(train_sequence, train_cluster_id, training_args)
  model.save(SAVED_MODEL_NAME)
  # we can also skip training by calling：
  # model.load(SAVED_MODEL_NAME)
  

  # testing
  for (test_sequence, test_cluster_id) in zip(test_sequences, test_cluster_ids):
    test_cluster_id = test_cluster_id.tolist()
    predicted_cluster_id = model.predict(test_sequence, inference_args)
    predicted_cluster_ids.append(predicted_cluster_id)
    accuracy = uisrnn.compute_sequence_match_accuracy(
        test_cluster_id, predicted_cluster_id)
    test_record.append((accuracy, len(test_cluster_id)))
    print('Ground truth labels:')
    print(test_cluster_id)
    print('Predicted labels:')
    print(predicted_cluster_id)
    print('-' * 80)

  output_string = uisrnn.output_result(model_args, training_args, test_record)

  print('Finished diarization experiment')
  print(output_string)


def main():
  """The main function."""
  model_args, training_args, inference_args = uisrnn.parse_arguments()
  diarization_experiment(model_args, training_args, inference_args)


if __name__ == '__main__':
  main()
