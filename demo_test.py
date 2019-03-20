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
  orig_train_sequences = np.load('data/train_sequence.npy').astype(np.float64)
  orig_train_cluster_ids = np.array(np.load('data/train_cluster_id.npy'))
  orig_test_sequences = np.load('data/test_sequence.npy').astype(np.float64)
  orig_test_cluster_ids = np.array(np.load('data/test_cluster_id.npy'))

  print(orig_test_sequences.shape)
  print(orig_test_cluster_ids.shape)

  orig_test_sequences = orig_test_sequences[:orig_test_sequences.shape[0]//100]
  orig_test_cluster_ids = orig_test_cluster_ids[:orig_test_sequences.shape[0]//100]

  print(orig_test_sequences.shape)
  print(orig_test_cluster_ids.shape)

  test_chunk_size = orig_test_sequences.shape[0] // 86
  test_left_over = orig_test_sequences.shape[0] % test_chunk_size
  test_new_len = orig_test_sequences.shape[0] - test_left_over

  test_sequences = np.split(orig_test_sequences[:test_new_len], test_chunk_size)
  test_cluster_ids = np.split(orig_test_cluster_ids[:test_new_len], test_chunk_size)

  print(test_sequences.shape)
  print(test_cluster_ids.shape)

  model = uisrnn.UISRNN(model_args)

  # train_sequences = np.array(train_sequences)
  # train_cluster_ids = np.array(train_cluster_ids)

  # d = vars(training_args)
  # # training
  # for i in range(train_sequences.shape[0]):
  #   train_sequence = train_sequences[i]
  #   train_cluster_id = train_cluster_ids[i]
  #   train_cluster_id = train_cluster_id.tolist()
  #   d['learning_rate'] = 1e-3
  #   model.fit(train_sequence, train_cluster_id, training_args)

  # # Take care of leftovers
  # train_sequence = orig_train_sequences[train_new_len:]
  # train_cluster_id = orig_train_cluster_id[train_new_len:]
  # d['learning_rate'] = 1e-3
  # model.fit(train_sequence, train_cluster_id, training_args)
  # model.save(SAVED_MODEL_NAME)

  # we can also skip training by calling：
  model.load(SAVED_MODEL_NAME)
  

  # testing
  # Take care of leftover
  # test_sequence = orig_test_sequences[test_new_len:]
  # test_cluster_id = orig_test_cluster_ids[test_new_len:].tolist()
  # predicted_cluster_id = model.predict(test_sequence, inference_args)
  # predicted_cluster_ids.append(predicted_cluster_id)
  # accuracy = uisrnn.compute_sequence_match_accuracy(
  #     test_cluster_id, predicted_cluster_id)
  # test_record.append((accuracy, len(test_cluster_id)))
  # print('Ground truth labels:')
  # print(test_cluster_id)
  # print('Predicted labels:')
  # print(predicted_cluster_id)
  # print('-' * 80)

  # Then the rest
  for (test_sequence, test_cluster_id) in zip(test_sequences, test_cluster_ids):
    print(test_sequence.shape)
    print(test_cluster_id)
    assert 1 == 2
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
