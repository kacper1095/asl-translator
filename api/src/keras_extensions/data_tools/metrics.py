import keras.backend as K


def fmeasure(y_true, y_pred):
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    rec = recall(y_true, y_pred)
    prec = precision(y_true, y_pred)
    return 2 * prec * rec / (prec + rec + K.epsilon())


def precision(y_true, y_pred):
    true_p = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    pred_p = K.sum(K.round(K.clip(y_pred, 0, 1)))
    prec = true_p / (pred_p + K.epsilon())
    return prec


def recall(y_true, y_pred):
    true_p = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_p = K.sum(K.round(K.clip(y_true, 0, 1)))
    rec = true_p / (possible_p + K.epsilon())
    return rec
