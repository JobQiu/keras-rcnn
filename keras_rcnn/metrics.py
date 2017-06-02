import tensorflow

def mean_average_precision(y_true, y_pred):
    return tensorflow.reduce_mean(tensorflow.metrics.sparse_average_precision_at_k(tensorflow.cast(y_true, tensorflow.int64), y_pred, 1)[0])