import numpy as np
import pandas as pd
import _pickle as pickle
import random
import torch.nn as nn
import torch
import os
import sklearn.metrics as skm
import warnings

from sklearn.utils import assert_all_finite
from sklearn.utils import check_consistent_length
from sklearn.utils.validation import _check_sample_weight
from sklearn.utils import column_or_1d, check_array
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.extmath import stable_cumsum
from sklearn.utils.sparsefuncs import count_nonzero
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.preprocessing import label_binarize
from HORIZON.CVmortalityPred_inHF.ModelPkg.encode_sklearn import _encode, _unique

def noSepMask(tokens, token2idx):
    output_label = []
    output_token = []
    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15 and token!='SEP' and i!=0:
            # prob /= 0.15

            # 80% randomly change token to mask token
            # if prob < 0.8:
            output_token.append(token2idx["MASK"])

            # 10% randomly change token to random token
            # elif prob < 0.9:
            #     output_token.append(random.choice(list(token2idx.values())))

            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later
            output_label.append(token2idx.get(token, token2idx['UNK']))
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)
            output_token.append(token2idx.get(token, token2idx['UNK']))

    return tokens, output_token, output_label


def newrandom_mask_cardio3(tokens, token2idx):
    output_label = []
    output_token = []
    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with 15% probability
        if prob < 0:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                output_token.append(token2idx["MASK"])

            # 10% randomly change token to random token
            elif prob < 0.9:
                output_token.append(random.choice(list(token2idx.values())))

            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later
            output_label.append(token2idx.get(token, token2idx['UNK']))
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)
            output_token.append(token2idx.get(token, token2idx['UNK']))

    return tokens, output_token, output_label


def covarUnsupMaker(covar):
    inputcovar = []
    labelcovar = []
    for i,x in enumerate(covar):
        prob = random.random()
        if prob <0.4:
            inputcovar.append(0)
            labelcovar.append(covar[i])
        else:
            inputcovar.append(covar[i])
            labelcovar.append(-1)
    return np.array(inputcovar), np.array(labelcovar)


#
# def randommaskreal(tokens, token2idx):
#     output_label = []
#     output_token = []
#     for i, token in enumerate(tokens):
#         prob = random.random()
#         # mask token with 15% probability
#         if prob < 0.15:
#             prob /= 0.15
#
#             # 80% randomly change token to mask token
#             if prob < 0.8:
#                 output_token.append(token2idx["MASK"])
#
#             # 10% randomly change token to random token
#             elif prob < 0.9:
#                 output_token.append(random.choice(list(token2idx.values())))
#
#             # -> rest 10% randomly keep current token
#
#             # append current token to output (we will predict these later
#             output_label.append(token2idx.get(token, token2idx['UNK']))
#         else:
#             # no masking token (will be ignored by loss function later)
#             output_label.append(-1)
#             output_token.append(token2idx.get(token, token2idx['UNK']))
#
#     return tokens, output_token, output_label




def randommaskreal(tokens, token2idx):
    output_label = []
    output_token = []
    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                output_token.append(token2idx["MASK"])
                output_label.append(token2idx.get(token, token2idx['UNK']))

            # 10% randomly change token to random token
            elif prob < 0.9:
                output_token.append(random.choice(list(token2idx.values())))
                output_label.append(token2idx.get(token, token2idx['UNK']))

            # -> rest 10% randomly keep current token
            else:
                output_label.append(-1)

            # append current token to output (we will predict these later
                output_token.append(token2idx.get(token, token2idx['UNK']))



        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)
            output_token.append(token2idx.get(token, token2idx['UNK']))

    return tokens, output_token, output_label


def GPLoad(model, likelihood, filepath, custom = None):
    pre_bert= filepath

    pretrained_dict = torch.load(pre_bert, map_location= 'cpu')
    pretrained_dict = pretrained_dict['model']
    modeld = model.state_dict()
    # 1. filter out unnecessary keys
    if custom==None:
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in modeld }
    else:
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in modeld  and k not in custom}

    modeld.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(modeld)

    pre_bert = filepath

    pretrained_dict = torch.load(pre_bert, map_location='cpu')
    likelihood.load_state_dict(pretrained_dict["likelihood"])
    return model, likelihood
def explode(df, lst_cols, fill_value='', preserve_index=False):
    # make sure `lst_cols` is list-alike
    fullcols = list(df.columns)
    if (lst_cols is not None
            and len(lst_cols) > 0
            and not isinstance(lst_cols, (list, tuple, np.ndarray, pd.Series))):
        lst_cols = [lst_cols]
    # all columns except `lst_cols`
    idx_cols = df.columns.difference(lst_cols)
    # calculate lengths of lists
    lens = df[lst_cols[0]].str.len()
    # preserve original index values
    idx = np.repeat(df.index.values, lens)
    # create "exploded" DF
    res = (pd.DataFrame({
        col: np.repeat(df[col].values, lens)
        for col in idx_cols},
        index=idx)
           .assign(**{col: np.concatenate(df.loc[lens > 0, col].values)
                      for col in lst_cols}))
    # append those rows that have empty lists
    if (lens == 0).any():
        # at least one list in cells is empty
        res = (res.append(df.loc[lens == 0, idx_cols], sort=False)
               .fillna(fill_value))
    # revert the original index order
    res = res.sort_index()
    # reset index if requested
    if not preserve_index:
        res = res.reset_index(drop=True)
    res = res.reindex(columns=fullcols)
    return res


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def code2index(tokens, token2idx):
    output_tokens = []
    for i, token in enumerate(tokens):
        output_tokens.append(token2idx.get(token, token2idx['UNK']))
    return tokens, output_tokens


def random_mask(tokens, token2idx):
    output_label = []
    output_token = []
    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15 and i != 0:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                output_token.append(token2idx["MASK"])

            # 10% randomly change token to random token
            elif prob < 0.9:
                output_token.append(random.choice(list(token2idx.values())))

            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later
            output_label.append(token2idx.get(token, token2idx['UNK']))
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)
            output_token.append(token2idx.get(token, token2idx['UNK']))

    return tokens, output_token, output_label


def random_mask(tokens, token2idx):
    output_label = []
    output_token = []
    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15 and i != 0:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                output_token.append(token2idx["MASK"])

            # 10% randomly change token to random token
            elif prob < 0.9:
                output_token.append(random.choice(list(token2idx.values())))

            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later
            output_label.append(token2idx.get(token, token2idx['UNK']))
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)
            output_token.append(token2idx.get(token, token2idx['UNK']))

    return tokens, output_token, output_label


def random_mask_atleast(tokens, token2idx, limitnum):
    output_label = []
    output_token = []
    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15 and i >= limitnum:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                output_token.append(token2idx["MASK"])

            # 10% randomly change token to random token
            elif prob < 0.9:
                output_token.append(random.choice(list(token2idx.values())))

            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later
            output_label.append(token2idx.get(token, token2idx['UNK']))
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)
            output_token.append(token2idx.get(token, token2idx['UNK']))

    return tokens, output_token, output_label


def random_mask_selective(tokens, token2idx, idx):
    output_label = []
    output_token = []
    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15 and i != 0 and i < idx:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                output_token.append(token2idx["MASK"])

            # 10% randomly change token to random token
            elif prob < 0.9:
                output_token.append(random.choice(list(token2idx.values())))

            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later
            output_label.append(token2idx.get(token, token2idx['UNK']))
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)
            output_token.append(token2idx.get(token, token2idx['UNK']))

    return tokens, output_token, output_label


def make_weights_for_balanced_classes_CatBased(fulld, nclasses, split):
    count = fulld.DataCat.value_counts()
    print(fulld.DataCat.value_counts())
    weight_per_class = [0.] * nclasses
    #     print(weight_per_class)
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N / (float(count[i]))
    weight = [0] * int(N)
    #     print(N)
    #     print(weight_per_class)
    weight_per_class[2] = weight_per_class[2] * split
    weight_per_class[3] = weight_per_class[3] * split

    print(weight_per_class)

    for idx, val in enumerate(fulld.DataCat):
        weight[idx] = weight_per_class[int(val)]
    return weight


def make_weights_for_balanced_classes(fulld, nclasses, split):
    count = fulld.diseaseLabel.value_counts().tolist()

    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N / (float(count[i]))
    weight = [0] * int(N)
    #     print(weight_per_class)
    weight_per_class[0] = weight_per_class[0] * split
    #     print(weight_per_class)

    for idx, val in enumerate(fulld.diseaseLabel):
        weight[idx] = weight_per_class[int(val)]
    return weight


def make_weights_for_balanced_classes(fulld, nclasses, split, exp=None, out=None):
    if exp is None:
        exp = 'diseaseLabel'
    count = fulld[exp].value_counts().tolist()

    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N / (float(count[i]))
    weight = [0] * int(N)
    #     print(weight_per_class)
    weight_per_class[0] = weight_per_class[0] * split
    #     print(weight_per_class)

    for idx, val in enumerate(fulld[exp]):
        weight[idx] = weight_per_class[int(val)]
    return weight


def set_random_mask(tokens, token2idx):
    output_label = []
    output_token = []
    if len(tokens) > 5:
        randomI = random.choice(range(len(tokens)))
        for i, token in enumerate(tokens):
            prob = random.random()
            if i == randomI:
                # mask token with 15% probability
                if prob < 0.5:

                    # 80% randomly change token to mask token
                    output_token.append(token2idx["MASK"])
                    output_label.append(token2idx.get(token, token2idx['UNK']))

                    # 10% randomly change token to random token
                elif prob >= 0.5:
                    output_token.append(random.choice(list(token2idx.values())))
                    output_label.append(token2idx.get(token, token2idx['UNK']))
            else:
                # no masking token (will be ignored by loss function later)
                output_label.append(-1)
                output_token.append(token2idx.get(token, token2idx['UNK']))
        return tokens, output_token, output_label

    else:

        return newrandom_mask(tokens, token2idx)


def newrandom_mask(tokens, token2idx):
    output_label = []
    output_token = []
    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with 15% probability
        if prob < 0:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                output_token.append(token2idx["MASK"])

            # 10% randomly change token to random token
            elif prob < 0.9:
                output_token.append(random.choice(list(token2idx.values())))

            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later
            output_label.append(token2idx.get(token, token2idx['UNK']))
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)
            output_token.append(token2idx.get(token, token2idx['UNK']))

    return tokens, output_token, output_label


def random_mask_Age(tokens, token2idx):
    output_label = []
    output_token = []

    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                output_token.append(token2idx["MASK"])

            # 10% randomly change token to random token
            elif prob < 0.9:
                output_token.append(random.choice(list(token2idx.values())))

            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later
            output_label.append(token2idx.get(token, token2idx['UNK']))
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)
            output_token.append(token2idx.get(token, token2idx['UNK']))

    return tokens, output_token, output_label


def norandom_mask(tokens, token2idx):
    output_label = []
    output_token = []
    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with 15% probability
        output_token.append(token2idx.get(token, token2idx['UNK']))

    return tokens, output_token


def index_seg(tokens, symbol='SEP'):
    flag = 0
    seg = []

    for token in tokens:
        if token == symbol:
            seg.append(flag)
            if flag == 0:
                flag = 1
            else:
                flag = 0
        else:
            seg.append(flag)
    return seg


def position_idx(tokens, symbol='SEP'):
    pos = []
    flag = 0

    for token in tokens:
        if token == symbol:
            pos.append(flag)
            flag += 1
        else:
            pos.append(flag)
    return pos


def age_vocab(max_age, year=False, symbol=None):
    age2idx = {}
    idx2age = {}
    if symbol is None:
        symbol = ['PAD', 'UNK']

    for i in range(len(symbol)):
        age2idx[str(symbol[i])] = i
        idx2age[i] = str(symbol[i])

    if year:
        for i in range(max_age):
            age2idx[str(i)] = len(symbol) + i
            idx2age[len(symbol) + i] = str(i)
    else:
        for i in range(max_age * 12):
            age2idx[str(i)] = len(symbol) + i
            idx2age[len(symbol) + i] = str(i)

    return age2idx, idx2age


def seq_padding(tokens, max_len, token2idx=None, symbol=None):
    if symbol is None:
        symbol = 'PAD'

    seq = []
    token_len = len(tokens)
    for i in range(max_len):
        if token2idx is None:
            if i < token_len:
                seq.append(tokens[i])
            else:
                seq.append(symbol)
        else:
            if i < token_len:
                # 1 indicate UNK
                seq.append(token2idx.get(tokens[i], token2idx['UNK']))
            else:
                seq.append(token2idx.get(symbol))
    return seq


def seq_padding_reverse(tokens, max_len, token2idx=None, symbol=None):
    if symbol is None:
        symbol = 'PAD'

    seq = []
    token_len = len(tokens)
    tokens = tokens[::-1]
    for i in range(max_len):
        if token2idx is None:
            if i < token_len:
                seq.append(tokens[i])
            else:
                seq.append(symbol)
        else:
            if i < token_len:
                # 1 indicate UNK
                seq.append(token2idx.get(tokens[i], token2idx['UNK']))
            else:
                seq.append(token2idx.get(symbol))
    return seq[::-1]


def age_seq_padding(tokens, max_len, token2idx=None, symbol=None):
    if symbol is None:
        symbol = 'PAD'

    seq = []
    token_len = len(tokens)
    for i in range(max_len):
        if token2idx is None:
            if i < token_len:
                seq.append(tokens[i])
            else:
                seq.append(symbol)
        else:
            if i < token_len:
                # 1 indicate UNK
                seq.append(token2idx[tokens[i]])
            else:
                seq.append(token2idx[symbol])
    return seq


def BPsplitMaker(start, finish, BPSPLIT, min=0):
    # for the treatment prediction bucket
    splits = np.linspace(start, finish, BPSPLIT)
    splits = np.insert(splits, 0, min)
    splits = np.insert(splits, len(splits), float('Inf'))
    t = {}
    for i, x in enumerate(splits[:-1]):
        t[float(i)] = str(x) + "--" + str(splits[i + 1])
    t = list(t.values())
    return (splits, t)





def cal_acc(label, pred, logS=True):
    logs = nn.LogSoftmax()
    label = label.cpu().numpy()
    ind = np.where(label != -1)[0]
    truepred = pred.detach().cpu().numpy()
    truepred = truepred[ind]
    truelabel = label[ind]
    if logS == True:
        truepred = logs(torch.tensor(truepred))
    else:
        truepred = torch.tensor(truepred)
    outs = [np.argmax(pred_x) for pred_x in truepred.numpy()]
    precision = skm.precision_score(truelabel, outs, average='micro')

    return precision

def cal_acc(label, pred, logS=True):
    logs = nn.LogSoftmax()
    label = label.cpu().numpy()
    ind = np.where(label != -1)[0]
    truepred = pred.detach().cpu().numpy()
    truepred = truepred[ind]
    truelabel = label[ind]
    if logS ==True:
        truepred = logs(torch.tensor(truepred))
    else:
        truepred = torch.tensor(truepred)
    outs = [np.argmax(pred_x) for pred_x in truepred.numpy()]
    precision = skm.precision_score(truelabel, outs, average='micro')

    return precision

def partition(values, indices):
    idx = 0
    for index in indices:
        sublist = []
        idxfill = []
        while idx < len(values) and values[idx] <= index:
            # sublist.append(values[idx])
            idxfill.append(idx)

            idx += 1
        if idxfill:
            yield idxfill


def toLoad(model, filepath, custom=None):
    pre_bert = filepath

    pretrained_dict = torch.load(pre_bert, map_location='cpu')
    modeld = model.state_dict()
    # 1. filter out unnecessary keys
    if custom == None:
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in modeld}
    else:
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in modeld and k not in custom}

    modeld.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(modeld)
    return model


import sklearn


def OutcomePrecision(logits, label, sig=True):
    sig = nn.Sigmoid()
    if sig == True:
        output = sig(logits)
    else:
        output = logits
    label, output = label.cpu(), output.detach().cpu()
    tempprc = sklearn.metrics.average_precision_score(label.numpy(), output.numpy())
    return tempprc, output, label


def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad


def precision_test(logits, label, sig=True):
    sigm = nn.Sigmoid()
    if sig == True:
        output = sigm(logits)
    else:
        output = logits
    label, output = label.cpu(), output.detach().cpu()

    tempprc = sklearn.metrics.average_precision_score(label.numpy(), output.numpy())
    return tempprc, output, label


def roc_auc(logits, label, sig=True):
    sigm = nn.Sigmoid()
    if sig == True:
        output = sigm(logits)
    else:
        output = logits
    label, output = label.cpu(), output.detach().cpu()

    tempprc = sklearn.metrics.roc_auc_score(label.numpy(), output.numpy())
    return tempprc, output, label


# golobal function
def create_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)



def top_k_accuracy_score(
    y_true, y_score, *, k=2, normalize=True, sample_weight=None, labels=None
):
    """Top-k Accuracy classification score.
    This metric computes the number of times where the correct label is among
    the top `k` labels predicted (ranked by predicted scores). Note that the
    multilabel case isn't covered here.
    Read more in the :ref:`User Guide <top_k_accuracy_score>`
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True labels.
    y_score : array-like of shape (n_samples,) or (n_samples, n_classes)
        Target scores. These can be either probability estimates or
        non-thresholded decision values (as returned by
        :term:`decision_function` on some classifiers). The binary case expects
        scores with shape (n_samples,) while the multiclass case expects scores
        with shape (n_samples, n_classes). In the multiclass case, the order of
        the class scores must correspond to the order of ``labels``, if
        provided, or else to the numerical or lexicographical order of the
        labels in ``y_true``.
    k : int, default=2
        Number of most likely outcomes considered to find the correct label.
    normalize : bool, default=True
        If `True`, return the fraction of correctly classified samples.
        Otherwise, return the number of correctly classified samples.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights. If `None`, all samples are given the same weight.
    labels : array-like of shape (n_classes,), default=None
        Multiclass only. List of labels that index the classes in ``y_score``.
        If ``None``, the numerical or lexicographical order of the labels in
        ``y_true`` is used.
    Returns
    -------
    score : float
        The top-k accuracy score. The best performance is 1 with
        `normalize == True` and the number of samples with
        `normalize == False`.
    See also
    --------
    accuracy_score
    Notes
    -----
    In cases where two or more labels are assigned equal predicted scores,
    the labels with the highest indices will be chosen first. This might
    impact the result if the correct label falls after the threshold because
    of that.
    Examples
    --------

    """
    y_true = check_array(y_true, ensure_2d=False, dtype=None)
    y_true = column_or_1d(y_true)
    y_type = type_of_target(y_true)
    if y_type == "binary" and labels is not None and len(labels) > 2:
        y_type = "multiclass"
    y_score = check_array(y_score, ensure_2d=False)
    y_score = column_or_1d(y_score) if y_type == "binary" else y_score
    check_consistent_length(y_true, y_score, sample_weight)

    if y_type not in {"binary", "multiclass"}:
        raise ValueError(
            f"y type must be 'binary' or 'multiclass', got '{y_type}' instead."
        )

    y_score_n_classes = y_score.shape[1] if y_score.ndim == 2 else 2

    if labels is None:
        classes = _unique(y_true)
        n_classes = len(classes)

        if n_classes != y_score_n_classes:
            raise ValueError(
                f"Number of classes in 'y_true' ({n_classes}) not equal "
                f"to the number of classes in 'y_score' ({y_score_n_classes})."
            )
    else:
        labels = column_or_1d(labels)
        classes = _unique(labels)
        n_labels = len(labels)
        n_classes = len(classes)

        if n_classes != n_labels:
            raise ValueError("Parameter 'labels' must be unique.")

        if not np.array_equal(classes, labels):
            raise ValueError("Parameter 'labels' must be ordered.")

        if n_classes != y_score_n_classes:
            raise ValueError(
                f"Number of given labels ({n_classes}) not equal to the "
                f"number of classes in 'y_score' ({y_score_n_classes})."
            )

        if len(np.setdiff1d(y_true, classes)):
            raise ValueError("'y_true' contains labels not in parameter 'labels'.")

    if k >= n_classes:
        warnings.warn(
            f"'k' ({k}) greater than or equal to 'n_classes' ({n_classes}) "
            "will result in a perfect score and is therefore meaningless.",
            UndefinedMetricWarning,
        )

    y_true_encoded = _encode(y_true, uniques=classes)

    if y_type == "binary":
        if k == 1:
            threshold = 0.5 if y_score.min() >= 0 and y_score.max() <= 1 else 0
            y_pred = (y_score > threshold).astype(np.int64)
            hits = y_pred == y_true_encoded
        else:
            hits = np.ones_like(y_score, dtype=np.bool_)
    elif y_type == "multiclass":
        sorted_pred = np.argsort(y_score, axis=1, kind="mergesort")[:, ::-1]
        hits = (y_true_encoded == sorted_pred[:, :k].T).any(axis=0)

    if normalize:
        return np.average(hits, weights=sample_weight)
    elif sample_weight is None:
        return np.sum(hits)
    else:
        return np.dot(hits, sample_weight)