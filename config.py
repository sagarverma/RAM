class Config(object):
  win_size = 16
  bandwidth = win_size**2
  batch_size = 32
  eval_batch_size = 50
  loc_std = 0.22
  original_size = 256
  num_channels = 3
  depth = 3
  sensor_size = win_size**2 * depth
  minRadius = 16
  hg_size = hl_size = 256
  g_size = 512
  cell_output_size = 512
  loc_dim = 2
  cell_size = 512
  cell_out_size = cell_size
  num_glimpses = 12
  num_classes = 11
  max_grad_norm = 5.
  train_images_size = 57372
  test_images_size = 16189
  train_data_path = '../../datasets/street2shop/street2shop_train.tfrecords'
  test_data_path = '../../datasets/street2shop/street2shop_test.tfrecords'

  step = 100000
  lr_start = 1e-3
  lr_min = 1e-4

  # Monte Carlo sampling
  M = 10
