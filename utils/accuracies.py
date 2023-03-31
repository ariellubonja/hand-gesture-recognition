import torch
import torch.nn.functional as F
from numpy.random import choice


def accuracy(y, y_hat):
    """Calculate the simple accuracy given two numpy vectors, each with int values
    corresponding to each class.

    Args:
        y (np.ndarray): actual value
        y_hat (np.ndarray): predicted value

    Returns:
        np.float64: accuracy
    """
    ### TODO Implement accuracy function
    
    
    return torch.sum(y == y_hat) / len(y)


def approx_train_acc_and_loss(model, train_data, train_labels):
    """Given a model, training data and its associated labels, calculate the simple accuracy when the 
    model is applied to the training dataset.
    This function is meant to be run during training to evaluate model training accuracy during training.

    Args:
        model (pytorch model): model class object.
        train_data (np.ndarray): training data
        train_labels (np.ndarray): training labels

    Returns:
        np.float64: simple accuracy
    """
    idxs = choice(len(train_data), 100, replace=False)
    # x = torch.from_numpy(train_data[idxs].astype(np.float32))
    x = train_data[idxs]
    # y = torch.from_numpy(train_labels[idxs].astype(np.int))
    y = train_labels[idxs]
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    y_pred = torch.max(logits, 1)[1]
    return accuracy(train_labels[idxs], y_pred), loss.item()


def dev_acc_and_loss(model, dev_data, dev_labels):
    """Given a model, a validation dataset and its associated labels, calcualte the simple accuracy when the
    model is applied to the validation dataset.
    This function is meant to be run during training to evaluate model validation accuracy.

    Args:
        model (pytorch model): model class obj
        dev_data (np.ndarray): validation data
        dev_labels (np.ndarray): validation labels

    Returns:
        np.float64: simple validation accuracy
    """
    # x = torch.from_numpy(dev_data.astype(np.float32))
    x = dev_data
    # y = torch.from_numpy(dev_labels.astype(np.int))
    y = dev_labels
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    y_pred = torch.max(logits, 1)[1]
    return accuracy(dev_labels, y_pred), loss.item()
