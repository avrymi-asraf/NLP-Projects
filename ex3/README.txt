def get_w2v_average(sent, word_to_vec, embedding_dim): # RETURNS TENSOR INSTEAD OF NP
    """
    This method gets a sentence and returns the average word embedding of the words consisting
    the sentence.
    :param sent: the sentence object
    :param word_to_vec: a dictionary mapping words to their vector embeddings
    :param embedding_dim: the dimension of the word embedding vectors
    :return The average embedding vector as Tensor.
    """



def train_epoch( # ADDED DEVICE AND RETURNS TRAIN RESULTS
        model, data_iterator, optimizer, criterion, device="cpu"
) -> Tuple[float, float]:
    """
    This method operates one epoch (pass over the whole train set) of training of the given model,
    and returns the accuracy and loss for this epoch
    :param device: cpu or cuda
    :param model: the model we're currently training
    :param data_iterator: an iterator, iterating over the training data for the model.
    :param optimizer: the optimizer object for the training process.
    :param criterion: the criterion object for the training process.
    :return: tuple(epoch_accuracy, epoch_loss)
    """

def evaluate(model, data_iterator, criterion, device="cpu"): # ADDED DEVICE
    """
    evaluate the model performance on the given data
    :param device: cpu or cuda
    :param model: one of our models.
    :param data_iterator: torch data iterator for the relevant subset
    :param criterion: the loss criterion used for evaluation
    :return: tuple of (average loss over all examples, average accuracy over all examples)
    """


def get_predictions_for_data(model, data_iter, device="cpu"): # ADDED DEVICE AND RETURNS NP ARRAY
    """

    This function should iterate over all batches of examples from data_iter and return all of the models
    predictions as a numpy ndarray or torch tensor (or list if you prefer). the prediction should be in the
    same order of the examples returned by data_iter.
    :param device: cpu or cuda
    :param model: one of the models you implemented in the exercise
    :param data_iter: torch iterator as given by the DataManager
    :return: 1d np array of predictions
    """


def train_model( # ADDED CRITERION AND CPU. RETURNS DATAFRAME OBJECT
        model,
        data_manager: DataManager,
        n_epochs,
        lr,
        criterion,
        weight_decay=0.01,
        device="cpu",
) -> pd.DataFrame:
    """
    Runs the full training procedure for the given model. The optimization should be done using the Adam
    optimizer with all parameters but learning rate and weight decay set to default.
    :param device: cpu or cuda
    :param criterion: Loss function
    :param model: module of one of the models implemented in the exercise
    :param data_manager: the DataManager object
    :param n_epochs: number of times to go over the whole training set
    :param lr: learning rate to be used for optimization
    :param weight_decay: parameter for l2 regularization
    :return DataFrame of the results
    """

def train_log_linear_with_one_hot(device="cpu", use_sub_phrases=True) -> pd.DataFrame: # ADDED DEVICE AND FLAG
    """
    Here comes your code for training and evaluation of the log linear model with one hot representation.
    :param device: cpu or cuda
    :param use_sub_phrases: use_sub_phrases flag
    """

def train_log_linear_with_w2v(device="cpu") -> pd.DataFrame: # ADDED DEVICE
    """
    Here comes your code for training and evaluation of the log linear model with word embeddings
    representation.
    :param device: cpu or cuda
    """

def train_lstm_with_w2v(device="cpu"):  # ADDED DEVICE
    """
    Here comes your code for training and evaluation of the LSTM model.
    :param device: cpu or cuda
    """



