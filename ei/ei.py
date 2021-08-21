import pandas as pd
import re
import numpy as np
from loguru import logger
from ei.utils import find_variable_importance, find_variable_effect, complete_words

class ExclusionInclusion:
    """
    model: keras model instance
    mode: "regression" or "classification"
    labels: All the label index of the classification in list e.g. [0, 1, 2, 3, 4]. This is only needed for
            classification.
    max_len: sequence length of the model i.e. the maxlen argument value while padding
    vocab_with_index: vocabulary with index value dictionary i.e. tokenizer.word_index
    verbose: 0 means no verbose, 1 means verbose
    """
    def __init__(self, model, mode, max_len, vocab_with_index, labels=None, verbose=1):
        self.model = model
        self.mode = mode
        self.max_len = max_len
        self.vocab_with_index = vocab_with_index
        self.labels = labels
        self.verbose = verbose


    #Function: Calculate enablers and disablers
    def find_enablers_and_disablers(self, X_without_padding, y_true, importance_gram_limit=5, effect_gram_limit=5,
                                    sequence_type="short", complete_phrase=True, predict_batch_size=32):

        """
        Inputs:::

        X_without_padding: The input word index numpy array sequence without padding, e.g. np.array([11, 2, 4])

        y_true : If mode is "regression" then y_true will be the true score, if model is "classification" then y_true \
        will be the class index for which the effect of phrases to be calculated.

        importance_gram_limit: The maximum phrase length while calculating importance of phrases. To get all the \
        possible combination set gram_limit to max_len argument value. This is required only for regression.

        effect_gram_limit: The maximum phrase length while calculating effect of phrases. To get all the possible \
        combination set gram_limit to max_len argument value.

        sequence_type: if "short" calculates all possible combination. if "long" breaks iteration where sign changes

        complete_phrase: if True replaces the 0 with the actual words in the response. if False, remove phrases \
        containing 0.

        Output:::
        returns a dictionary with the effect values of the important phrases

        """
        if len(X_without_padding.shape) == 2:
            X_without_padding = X_without_padding[:, :self.max_len]
        else:
            X_without_padding = X_without_padding[:self.max_len]

        response_actual = X_without_padding.copy()

        if self.mode == "regression":
            if self.verbose:
                logger.info(f"True Output: {y_true}")
                logger.info(f"Finding important variables!")

            important_feat = find_variable_importance(X_without_padding, y_true, self.model, self.max_len,
                                                      self.vocab_with_index, importance_gram_limit, predict_batch_size)

            important_feat = pd.DataFrame({"words": list(important_feat.keys()),
                                           "importance": list(important_feat.values())})
            important_feat = important_feat.loc[important_feat.importance > 0, "words"].tolist()

            important_words_list = []
            for word in important_feat:
                important_words_list += word.split()

            #convert word to index
            important_words_idx = [self.vocab_with_index[word] for word in important_words_list]

            #Fill unimportant words idx to 0
            # response_actual = X_without_padding.copy()
            response = np.array([idx if idx in important_words_idx else 0 for idx in X_without_padding])

            #set y to y_true
            y = y_true

        elif self.mode == "classification":
            response = X_without_padding
            complete_phrase = False
            y = y_true

        else:
            return "Mode should be either Regression or Classification."

        if self.verbose:
            logger.info(f"Finding enablers and disablers!")
        enablers_disablers = find_variable_effect(response, y, self.model, self.mode, self.labels, self.max_len,
                                                  self.vocab_with_index, effect_gram_limit, sequence_type,
                                                  predict_batch_size)

        #Post processing of enablers and disablers
        enablers_disablers_new = dict()

        if complete_phrase == True:
            if self.verbose:
                logger.info("Completing phrases")
            for key, value in enablers_disablers.items():

                if len(re.sub(pattern="<SPACE>", repl="", string=key).strip()) != 0:

                    if "<SPACE>" in key:
                        complete_key = complete_words(enabler_or_disabler=key, response=response_actual,
                                                      vocab_with_index=self.vocab_with_index)
                        enablers_disablers_new[complete_key] = value

                    else:
                        enablers_disablers_new[key.strip()] = value

        elif complete_phrase == False:
            for key, value in enablers_disablers.items():
                if "<SPACE>" not in key:
                    enablers_disablers_new[key] = value



        #Return Result
        return enablers_disablers_new










