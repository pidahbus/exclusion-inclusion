import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import itertools


# Function: Enablers and Disablers
def find_variable_importance(X, y_true, model, max_len, vocab_with_index, gram_limit, predict_batch_size):
    # Enablers and Disablers
    en_dis_dict = dict()

    # Reshape X
    if len(X.shape) < 2:
        X = X.reshape(1, X.shape[0])

    y_true = np.array(y_true)

    # All words
    pred_with_all_words = model.predict(pad_sequences(X, maxlen=max_len), batch_size=predict_batch_size)
    pred_with_all_words = pred_with_all_words.reshape(pred_with_all_words.shape[0], )
    # logger.info(f"Prediction including all the words: {pred_with_all_words}")
    err_with_all_words = np.abs(y_true - pred_with_all_words)[0]

    # Unigrams
    X_unigram = np.array(X.tolist() * X.shape[1])
    np.fill_diagonal(X_unigram, 0.)
    pred_unigram = model.predict(pad_sequences(X_unigram, maxlen=max_len), batch_size=predict_batch_size)
    # logger.info("Predictions for 1-gram has been finished")
    pred_unigram = pred_unigram.reshape(X.shape[1], )
    err_unigram = np.abs(y_true - pred_unigram)

    # Error DF
    err_df = pd.DataFrame({"1": err_unigram, "0": err_with_all_words})
    en_dis_df = pd.DataFrame({"1": np.array(err_df.loc[:, "1"] > err_df.loc[:, "0"]) * 1})

    # Reverse Dict
    idx_to_dict = {}
    for key, value in vocab_with_index.items():
        idx_to_dict[value] = key

    idx_to_dict[0] = "<SPACE>"

    # Update enablers and disablers
    en_dis_dict[idx_to_dict[X[0, err_df.shape[0] - 1]]] = (err_df.loc[err_df.shape[
                                                                          0] - 1, "1"] - err_with_all_words) * 100 / err_with_all_words

    # Update error DF and enablers and disablers information
    err_df = err_df[:-1]
    en_dis_df = en_dis_df[:-1]

    # Bi-grams
    X_bigram = np.array(X.tolist() * err_df.shape[0])
    np.fill_diagonal(X_bigram, 0.)
    bi_rng = np.arange(err_df.shape[0])
    X_bigram[bi_rng, bi_rng + 1] = 0.

    pred_bigram = model.predict(pad_sequences(X_bigram, maxlen=max_len), batch_size=predict_batch_size)
    # logger.info("Predictions for 2-gram has been finished")
    pred_bigram = pred_bigram.reshape(err_df.shape[0], )
    err_bigram = np.abs(y_true - pred_bigram)

    # Update error DF and enablers and disablers information
    err_df.loc[err_df.index, "2"] = err_bigram
    en_dis_df.loc[en_dis_df.index, "2"] = np.array((err_df.loc[:, "2"] > err_df.loc[:, "1"]) * 1)

    # Update enablers and disablers
    en_dis_list = en_dis_df[en_dis_df.loc[:, "1"] != en_dis_df.loc[:, "2"]].index.tolist()
    if err_df.index[-1] not in en_dis_list:
        en_dis_dict[idx_to_dict[X[0, err_df.index[-1]]] + " " + idx_to_dict[X[0, err_df.index[-1] + 1]]] = (err_df.loc[
                                                                                                                err_df.index[
                                                                                                                    -1], "2"] - err_with_all_words) * 100 / err_with_all_words
        err_df = err_df[:-1]
        en_dis_df = en_dis_df[:-1]

    for idx in en_dis_list:
        en_dis_dict[idx_to_dict[X[0, idx]]] = (err_df.loc[idx, "2"] - err_with_all_words) * 100 / err_with_all_words

    # Update error DF and enablers and disablers information
    err_df = err_df.drop(en_dis_list)
    en_dis_df = en_dis_df.drop(en_dis_list)

    # While Loop
    gram = 3
    while (err_df.shape[0] > 0) and (gram <= gram_limit):

        # Try the next gram
        X_ngram = np.array(X.tolist() * (X.shape[1] * (gram - 1)))
        np.fill_diagonal(X_ngram, 0.)
        for i in range(1, gram):
            rng = np.arange(err_df.shape[0])
            X_ngram[rng, rng + i] = 0.

        X_ngram = np.take(X_ngram, indices=err_df.index.tolist(), axis=0)

        pred_ngram = model.predict(pad_sequences(X_ngram, maxlen=max_len), batch_size=predict_batch_size)
        # logger.info(f"Predictions for {gram}-gram has been finished")
        pred_ngram = pred_ngram.reshape(err_df.shape[0], )
        err_ngram = np.abs(y_true - pred_ngram)

        # Update error DF and enablers and disablers information
        err_df.loc[err_df.index, str(gram)] = err_ngram
        en_dis_df.loc[en_dis_df.index, str(gram)] = np.array(
            (err_df.loc[:, str(gram)] > err_df.loc[:, str(gram - 1)]) * 1)

        # Update enablers and disablers
        en_dis_list = en_dis_df[en_dis_df.loc[:, str(gram - 1)] != en_dis_df.loc[:, str(gram)]].index.tolist()
        if err_df.index[-1] not in en_dis_list:
            key = ""
            for i in range(gram):
                key = key + " " + idx_to_dict[X[0, err_df.index[-1] + i]]
            en_dis_dict[key] = (err_df.loc[err_df.index[-1], str(gram)] - err_with_all_words) * 100 / err_with_all_words
            err_df = err_df[:-1]
            en_dis_df = en_dis_df[:-1]

        for idx in en_dis_list:
            key = ""
            for i in range(gram - 1):
                key = key + " " + idx_to_dict[X[0, idx + i]]
            en_dis_dict[key] = (err_df.loc[idx, str(gram - 1)] - err_with_all_words) * 100 / err_with_all_words

        # Update error DF and enablers and disablers information
        err_df = err_df.drop(en_dis_list)
        en_dis_df = en_dis_df.drop(en_dis_list)

        # Update gram
        gram += 1

    return en_dis_dict


# In[8]:


# Function: Enablers and Disablers
def find_variable_effect(X, y, model, mode, labels, max_len, vocab_with_index, gram_limit, sequence_type,
                         predict_batch_size):
    # Enablers and Disablers
    en_dis_dict = dict()

    # Reshape X
    if len(X.shape) < 2:
        X = X.reshape(1, X.shape[0])

    # All words
    pred_with_all_words = model.predict(pad_sequences(X, maxlen=max_len), batch_size=predict_batch_size)
    # pred_with_all_words = model.predict(X)
    # logger.info(f"Prediction including all the words: {pred_with_all_words}")

    if mode == "classification":
        pred_with_all_words = pred_with_all_words[0, y]

    elif mode == "regression":
        pred_with_all_words = pred_with_all_words[0][0]

    # Unigrams
    X_unigram = np.array(X.tolist() * X.shape[1])
    np.fill_diagonal(X_unigram, 0.)
    pred_unigram = model.predict(pad_sequences(X_unigram, maxlen=max_len), batch_size=predict_batch_size)
    # pred_unigram = model.predict(X_unigram)
    # logger.info("Predictions for 1-gram has been finished")

    if mode == "regression":
        pred_unigram = pred_unigram.reshape(X.shape[1], )


    elif mode == "classification":
        pred_unigram = pred_unigram[:, y]

    # Error DF
    err_df = pd.DataFrame({"1": pred_unigram, "0": pred_with_all_words})
    en_dis_df = pd.DataFrame({"1": np.array(err_df.loc[:, "1"] < err_df.loc[:, "0"]) * 1})

    # Reverse Dict
    idx_to_dict = {}
    for key, value in vocab_with_index.items():
        idx_to_dict[value] = key

    idx_to_dict[0] = "<SPACE>"

    # Update enablers and disablers
    en_dis_dict[idx_to_dict[X[0, err_df.shape[0] - 1]]] = (pred_with_all_words - err_df.loc[
        err_df.shape[0] - 1, "1"]) * 100 / pred_with_all_words

    # Update error DF and enablers and disablers information
    err_df = err_df[:-1]
    en_dis_df = en_dis_df[:-1]

    # Bi-grams
    X_bigram = np.array(X.tolist() * err_df.shape[0])
    np.fill_diagonal(X_bigram, 0.)
    bi_rng = np.arange(err_df.shape[0])
    X_bigram[bi_rng, bi_rng + 1] = 0.

    pred_bigram = model.predict(pad_sequences(X_bigram, maxlen=max_len), batch_size=predict_batch_size)
    # pred_bigram = model.predict(X_bigram)
    # logger.info("Predictions for 2-gram has been finished")

    if mode == "regression":
        pred_bigram = pred_bigram.reshape(err_df.shape[0], )

    elif mode == "classification":
        pred_bigram = pred_bigram[:, y]

    # Update error DF and enablers and disablers information
    err_df.loc[err_df.index, "2"] = pred_bigram
    en_dis_df.loc[en_dis_df.index, "2"] = np.array((err_df.loc[:, "2"] < err_df.loc[:, "1"]) * 1)

    # Update enablers and disablers
    if sequence_type == "long":
        en_dis_list = en_dis_df[en_dis_df.loc[:, "1"] != en_dis_df.loc[:, "2"]].index.tolist()

    elif sequence_type == "short":
        en_dis_list = en_dis_df.index.tolist()[:-1]

    if (err_df.index[-1] not in en_dis_list) or (sequence_type == "short"):
        en_dis_dict[idx_to_dict[X[0, err_df.index[-1]]] + " " + idx_to_dict[X[0, err_df.index[-1] + 1]]] = (
                                                                                                                       pred_with_all_words -
                                                                                                                       err_df.loc[
                                                                                                                           err_df.index[
                                                                                                                               -1], "2"]) * 100 / pred_with_all_words
        err_df = err_df[:-1]
        en_dis_df = en_dis_df[:-1]

    for idx in en_dis_list:
        en_dis_dict[idx_to_dict[X[0, idx]]] = (pred_with_all_words - err_df.loc[idx, "2"]) * 100 / pred_with_all_words

    if sequence_type == "long":
        # Update error DF and enablers and disablers information
        err_df = err_df.drop(en_dis_list)
        en_dis_df = en_dis_df.drop(en_dis_list)

    # While Loop
    gram = 3
    while (err_df.shape[0] > 0) and (gram <= gram_limit):

        # Try the next gram
        X_ngram = np.array(X.tolist() * (X.shape[1] * (gram - 1)))
        np.fill_diagonal(X_ngram, 0.)
        for i in range(1, gram):
            rng = np.arange(err_df.shape[0])
            X_ngram[rng, rng + i] = 0.

        X_ngram = np.take(X_ngram, indices=err_df.index.tolist(), axis=0)

        pred_ngram = model.predict(pad_sequences(X_ngram, maxlen=max_len), batch_size=predict_batch_size)
        # pred_ngram = model.predict(pad_sequences(X_ngram, maxlen=max_len))

        # logger.info(f"Predictions for {gram}-gram has been finished")

        if mode == "regression":
            pred_ngram = pred_ngram.reshape(err_df.shape[0], )


        elif mode == "classification":
            pred_ngram = pred_ngram[:, y]

        # Update error DF and enablers and disablers information
        err_df.loc[err_df.index, str(gram)] = pred_ngram
        en_dis_df.loc[en_dis_df.index, str(gram)] = np.array(
            (err_df.loc[:, str(gram)] > err_df.loc[:, str(gram - 1)]) * 1)

        # Update enablers and disablers
        if sequence_type == "long":
            en_dis_list = en_dis_df[en_dis_df.loc[:, str(gram - 1)] != en_dis_df.loc[:, str(gram)]].index.tolist()

        elif sequence_type == "short":
            en_dis_list = en_dis_df.index.tolist()[:-1]

        if (err_df.index[-1] not in en_dis_list) or (sequence_type == "short"):
            key = ""
            for i in range(gram):
                key = key + " " + idx_to_dict[X[0, err_df.index[-1] + i]]
            en_dis_dict[key] = (pred_with_all_words - err_df.loc[
                err_df.index[-1], str(gram)]) * 100 / pred_with_all_words
            err_df = err_df[:-1]
            en_dis_df = en_dis_df[:-1]

        for idx in en_dis_list:
            key = ""
            for i in range(gram - 1):
                key = key + " " + idx_to_dict[X[0, idx + i]]
            en_dis_dict[key] = (pred_with_all_words - err_df.loc[idx, str(gram - 1)]) * 100 / pred_with_all_words

        # Update error DF and enablers and disablers information
        if sequence_type == "long":
            err_df = err_df.drop(en_dis_list)
            en_dis_df = en_dis_df.drop(en_dis_list)

        # Update gram
        gram += 1

    return en_dis_dict


# In[9]:


# Remove <SPACE> and replace actual words
def complete_words(enabler_or_disabler, response, vocab_with_index, replace_word="<SPACE>"):
    # Reverse Dict
    idx_to_dict = {}
    for key, value in vocab_with_index.items():
        idx_to_dict[value] = key

    idx_to_dict[0] = "<SPACE>"

    # Find the length of the phrase
    enabler_or_disabler = enabler_or_disabler.strip().split()
    phrase_len = len(enabler_or_disabler)

    # Find the index of the replace_word
    repl_idx = [i for i, word in enumerate(enabler_or_disabler) if word == replace_word]

    # Find the unmasked words and indices
    unmasked_idx = [idx for idx in list(range(phrase_len)) if idx not in repl_idx]
    unmasked_words_idx = [vocab_with_index[word] for word in enabler_or_disabler if word not in [replace_word]]

    # Find the occurences of all the unmasked words
    unmasked_occurences = []
    for word_idx in unmasked_words_idx:
        unmasked_occurences.append([i for i, idx in enumerate(response) if idx == word_idx])

    # Find all the combinations
    unmasked_comb = np.array(list(itertools.product(*unmasked_occurences)))
    unmasked_comb_next = unmasked_comb - unmasked_comb[:, 0].reshape(unmasked_comb.shape[0], 1) + min(unmasked_idx)
    desired_idx = np.where(np.sum(unmasked_comb_next == np.array(unmasked_idx), axis=1) == len(unmasked_idx))[0]
    desired_word_idx = unmasked_comb[desired_idx, :].tolist()[0]
    start_word_idx = min(desired_word_idx) - min(unmasked_idx)
    end_word_idx = start_word_idx + len(enabler_or_disabler) - 1

    # Extract the final phrase
    complete_phrase = " ".join([idx_to_dict[idx] for idx in response[start_word_idx:end_word_idx + 1]])

    return complete_phrase