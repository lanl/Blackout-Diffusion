import tensorflow as tf
import tensorflow_datasets as tfds
import datasets

def generateData(config,dataset):

    dataset_builder = tfds.builder(dataset,data_dir='dataBuffer')
    train_split_name = 'train'
    eval_split_name = 'test'
    dataset_builder.download_and_prepare()

    def resize_op(img):
        img = tf.image.convert_image_dtype(img, tf.float32)
        return tf.image.resize(img, [config.data.image_size, config.data.image_size], antialias=True)

    def create_dataset(dataset_builder, split):
        dataset_options = tf.data.Options()
        dataset_options.experimental_optimization.map_parallelization = True
        dataset_options.experimental_threading.private_threadpool_size = 48
        dataset_options.experimental_threading.max_intra_op_parallelism = 1
        read_config = tfds.ReadConfig(options=dataset_options)
        if isinstance(dataset_builder, tfds.core.DatasetBuilder):
            dataset_builder.download_and_prepare()
            ds = dataset_builder.as_dataset(
            split=split, shuffle_files=True, read_config=read_config)
        else:
            ds = dataset_builder.with_options(dataset_options)

        ds = ds.repeat(count=None)
        ds = ds.shuffle(shuffle_buffer_size)
        ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.batch(config.training.batch_size, drop_remainder=True)

        return ds.prefetch(prefetch_size)


    evaluation = False
    uniform_dequantization = False
    shuffle_buffer_size = 10000
    prefetch_size = tf.data.experimental.AUTOTUNE
    num_epochs = None

    def preprocess_fn(d):

          """Basic preprocessing function scales data to [0, 1) and randomly flips."""
          img = resize_op(d['image'])
          if config.data.random_flip and not evaluation:
            img = tf.image.random_flip_left_right(img)
          if uniform_dequantization:
            img = (tf.random.uniform(img.shape, dtype=tf.float32) + img * 255.) / 256.

          return dict(image=img, label=d.get('label', None))


    prefetch_size = tf.data.experimental.AUTOTUNE
    
    train_ds = create_dataset(dataset_builder, train_split_name)
    eval_ds = create_dataset(dataset_builder, eval_split_name)
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)
    
    return train_ds, eval_ds, scaler
