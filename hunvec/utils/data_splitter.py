from random import shuffle


def shuffled_indices(data_len, ratios):
    indices = range(data_len)
    shuffle(indices)
    training = int(ratios[0] * data_len)
    test = int(ratios[1] * data_len)
    valid = int(ratios[2] * data_len)
    training_indices = indices[:training]
    valid_indices = indices[training:training + valid]
    test_indices = indices[training+valid:training+valid+test]
    return training_indices, valid_indices, test_indices


def datasplit(data, indices, ratios=[.7, .15, .15]):
    training_indices, valid_indices, test_indices = indices
    tr_data = [data[i] for i in training_indices]
    tst_data = [data[i] for i in test_indices]
    v_data = [data[i] for i in valid_indices]
    return tr_data, tst_data, v_data
