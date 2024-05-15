from __future__ import absolute_import, division, print_function

import pytorch_pretrained_bert as Bert

import os
import argparse
import random
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

from ModelPkg.CPHloss import *
from ModelPkg.Data_Deterministic import *

import os
import argparse
import random
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

import sys
import torch
import torch.nn as nn
import numpy as np
import copy
import torch.nn.functional as F
import math

import math
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint

import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint

NUM_INT_STEPS = 1000



def adam_surv(params, config=None):
    if config is None:
        config = {
            'lr': 3e-5,
            'warmup_proportion': 0.1
        }
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight', 'Eps','VAE']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in params if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optim = Bert.optimization.BertAdam(optimizer_grouped_parameters,
                                       lr=config['lr'],
                                       warmup=config['warmup_proportion'])
    return optim
def make_net(input_size, hidden_size, num_layers, output_size, dropout=0,
             batch_norm=False, act="relu", softplus=True):

    ActFn = nn.ELU
    modules = [nn.Linear(input_size, hidden_size), ActFn()]
    if batch_norm:
        modules.append(nn.BatchNorm1d(hidden_size))
    if dropout > 0:
        modules.append(nn.Dropout(p=dropout))
    if num_layers > 1:
        for _ in range(num_layers - 1):
            modules.append(nn.Linear(hidden_size, hidden_size))
            modules.append(ActFn())
            if batch_norm:
                modules.append(nn.BatchNorm1d(hidden_size))
            if dropout > 0:
                modules.append(nn.Dropout(p=dropout))
    modules.append(nn.Linear(hidden_size, output_size))
    if softplus:  # ODE models
        modules.append(nn.Softplus())
    return nn.Sequential(*modules)


class BaseSurvODEFunc(nn.Module):
    def __init__(self, config):
        super(BaseSurvODEFunc, self).__init__()
        self.nfe = 0
        self.batch_time_mode = False
        self.config = config

    def set_batch_time_mode(self, mode=True):
        self.batch_time_mode = mode
        # `odeint` requires the output of `odefunc` to have the same size as
        # `init_cond` despite the how many steps we are going to evaluate. Set
        # `self.batch_time_mode` to `False` before calling `odeint`. However,
        # when we want to call the forward function of `odefunc` directly and
        # when we would like to evaluate multiple time steps at the same time,
        # set `self.batch_time_mode` to `True` and the output will have size
        # (len(t), size(y)).

    def reset_nfe(self):
        self.nfe = 0

    def forward(self, t, y):
        raise NotImplementedError("Not implemented.")


class ContextRecMLPODEFunc(BaseSurvODEFunc):
    def __init__(self, config):
        super(ContextRecMLPODEFunc, self).__init__(config)
        self.feature_size = config.hidden_size
        self.hidden_size = config.hidden_size
        self.num_layers = 2
        self.batch_norm = config.batch_norm_ODE
        self.config = config
        self.net = make_net(input_size=self.feature_size + 2, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, output_size=1,
                            batch_norm=self.batch_norm)

    def forward(self, t, y):
        """
        Arguments:
          t: When self.batch_time_mode is False, t is a scalar indicating the
            time step to be evaluated. When self.batch_time_mode is True, t is
            a 1-D tensor with a single element [1.0].
          y: When self.batch_time_mode is False, y is a 1-D tensor with length
            2 + k, where the first dim indicates Lambda_t, the second dim
            indicates the final time step T to be evaluated, and the remaining
            k dims indicates the features. When self.batch_time_mode is True, y
            is a 2-D tensor with batch_size * (2 + k).
        """
        self.nfe += 1
        device = next(self.parameters()).device
        Lambda_t = y.index_select(-1, torch.tensor([0]).to(device)).view(-1, 1)
        T = y.index_select(-1, torch.tensor([1]).to(device)).view(-1, 1)
        x = y.index_select(-1, torch.tensor(range(2, y.size(-1))).to(device))

        # Rescaling trick
        # $\int_0^T f(s; x) ds = \int_0^1 T f(tT; x) dt$, where $t = s / T$
        inp = torch.cat(
            [Lambda_t,
             t.repeat(T.size()) * T,  # s = t * T
             x.view(-1, self.feature_size)], dim=1)
        output = self.net(inp) * T  # f(tT; x) * T
        zeros = torch.zeros_like(
            y.index_select(-1, torch.tensor(range(1, y.size(-1))).to(device))
        )

        output = torch.cat([output, zeros], dim=1)
        if self.batch_time_mode:
            return output
        else:
            return output.squeeze(0)


class NonCoxFuncModel(nn.Module):
    """NonCoxFuncModel."""

    def __init__(self, config):
        """Initializes a NonCoxFuncModel.
        Arguments:
          config: An OrderedDict of lists. The keys of the dict indicate
            the names of different parts of the model. Each value of the dict
            is a list indicating the configs of layers in the corresponding
            part. Each element of the list is a list [layer_type, arguments],
            where layer_type is a string and arguments is a dict.
          feature_size: Feature size.
          use_embed: Whether to use embedding layer after input.
        """
        super(NonCoxFuncModel, self).__init__()
        self.config = config
        self.feature_size = config.hidden_size
        self.func_type = config.func_type

        self.odefunc = ContextRecMLPODEFunc(config)
        self.model_config = config
        self.set_last_eval(False)

    def set_last_eval(self, last_eval=True):
        self.last_eval = last_eval

    def forward(self, orig_t, orig_init_cond, orig_features, fullEval):
        device = next(self.parameters()).device
        t = orig_t
        init_cond = orig_init_cond
        features = orig_features
        init_cond = torch.cat([orig_init_cond.view(-1, 1), orig_t.view(-1, 1), orig_features],
                              dim=1)
        t = torch.tensor([0., 1.]).to(device)

        outputs = {}
        self.odefunc.set_batch_time_mode(False)
        outputs["Lambda"] = odeint(
            self.odefunc, init_cond, t, rtol=1e-4, atol=1e-8)[
                            1:].squeeze()  # size: [length of t] x [batch size] x [dim of y0]
        self.odefunc.set_batch_time_mode(True)

        outputs["lambda"] = self.odefunc(t[1:], outputs["Lambda"]).squeeze()
        outputs["Lambda"] = outputs["Lambda"][:, 0]
        outputs["lambda"] = outputs["lambda"][:, 0] / orig_t

        self.odefunc.set_batch_time_mode(False)
        ones = torch.ones_like(orig_t)

        t = self.model_config.time_nums * ones
        init_cond = orig_init_cond
        features = orig_features
        init_cond = torch.cat([init_cond.view(-1, 1), t.view(-1, 1), features], dim=1)

        t_min = 1
        t_max = self.model_config.time_nums
        maxsteps = t_max
        t = torch.linspace(
            t_min, t_max, maxsteps, dtype=init_cond.dtype,
            device=device)
        t = torch.cat([torch.zeros([1]).to(device), t], dim=0)
        t = t / t_max
        outputs["hazard_seq"] = (1 - torch.exp(-odeint(self.odefunc, init_cond, t, rtol=1e-4,
                                                       atol=1e-8))[1:, :, 0]).transpose(1, 0)

        return outputs


class BertConfig(Bert.modeling.BertConfig):
    def __init__(self, config):
        super(BertConfig, self).__init__(
            vocab_size_or_config_json_file=config.get('vocab_size'),
            hidden_size=config['hidden_size'],
            num_hidden_layers=config.get('num_hidden_layers'),
            num_attention_heads=config.get('num_attention_heads'),
            intermediate_size=config.get('intermediate_size'),
            hidden_act=config.get('hidden_act'),
            hidden_dropout_prob=config.get('hidden_dropout_prob'),
            attention_probs_dropout_prob=config.get('attention_probs_dropout_prob'),
            max_position_embeddings=config.get('max_position_embedding'),
            initializer_range=config.get('initializer_range'),
        )
        self.seg_vocab_size = config.get('seg_vocab_size')
        self.age_vocab_size = config.get('age_vocab_size')
        self.region_vocab_size = config.get('region_size')
        self.gender_vocab_size = config.get('gender_size')
        self.genderHidden = config.get('genderHidden')
        self.regionHidden = config.get('regionHidden')
        self.embeddingdevice = config.get('embeddingdevice', -1)
        self.otherdevice = config.get('otherdevice', -1)
        self.concat_embeddings = config.get('concat_embeddings', -1)
        self.batch_norm_ODE = config.get('batch_norm_ODE', False)
        self.func_type = config.get('func_type', 'rec_mlp')
        self.time_nums = config.get('time_nums', False)


class BertEmbeddings(nn.Module):
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.segment_embeddings = nn.Embedding(config.seg_vocab_size, config.hidden_size)
        self.age_embeddings = nn.Embedding(config.age_vocab_size, config.hidden_size)
        self.posi_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size). \
            from_pretrained(embeddings=self._init_posi_embedding(config.max_position_embeddings, config.hidden_size))

        self.LayerNorm = Bert.modeling.BertLayerNorm(config.hidden_size, eps=1e-12)
        self.config = config
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if self.config.concat_embeddings:
            self.catmap = nn.Linear(3 * config.hidden_size, config.hidden_size)
            self.tanh = nn.Tanh()
            print('turn on concat - cehrt structure - embeddings')

    def forward(self, word_ids, age_ids=None, seg_ids=None, posi_ids=None, age=True):
        if seg_ids is None:
            seg_ids = torch.zeros_like(word_ids)
        if age_ids is None:
            age_ids = torch.zeros_like(word_ids)
        if posi_ids is None:
            posi_ids = torch.zeros_like(word_ids)

        word_embed = self.word_embeddings(word_ids)
        age_embed = self.age_embeddings(age_ids)
        posi_embeddings = self.posi_embeddings(posi_ids)
        seg_embed = self.posi_embeddings(seg_ids)
        # no year embed used here!
        if self.config.concat_embeddings:
            embeddings = self.tanh(self.catmap(torch.cat((word_embed, age_embed, posi_embeddings), dim=2))) + seg_embed

        else:
            embeddings = word_embed + seg_embed + age_embed + posi_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def _init_posi_embedding(self, max_position_embedding, hidden_size):
        def even_code(pos, idx):
            return np.sin(pos / (10000 ** (2 * idx / hidden_size)))

        def odd_code(pos, idx):
            return np.cos(pos / (10000 ** (2 * idx / hidden_size)))

        # initialize position embedding table
        lookup_table = np.zeros((max_position_embedding, hidden_size), dtype=np.float32)

        # reset table parameters with hard encoding
        # set even dimension
        for pos in range(max_position_embedding):
            for idx in np.arange(0, hidden_size, step=2):
                lookup_table[pos, idx] = even_code(pos, idx)
        # set odd dimension
        for pos in range(max_position_embedding):
            for idx in np.arange(1, hidden_size, step=2):
                lookup_table[pos, idx] = odd_code(pos, idx)

        return torch.tensor(lookup_table)


class BertModel(Bert.modeling.BertPreTrainedModel):
    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config=config)
        self.encoder = Bert.modeling.BertEncoder(config=config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, age_ids=None, seg_ids=None, posi_ids=None, attention_mask=None,
                output_all_encoded_layers=True):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if age_ids is None:
            age_ids = torch.zeros_like(input_ids)
        if seg_ids is None:
            seg_ids = torch.zeros_like(input_ids)
        if posi_ids is None:
            posi_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, age_ids, seg_ids, posi_ids)
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class SurvODEloss(torch.nn.Module):

    def __init__(self, reduction):
        super(SurvODEloss, self).__init__()
        self.reduction = reduction

    """Negative log-likelihood of the hazard parametrization model.
    See `loss.nll_logistic_hazard` for details.

    Arguments:
        reduction {string} -- How to reduce the loss.
            'none': No reduction.
            'mean': Mean of tensor.
            'sum: sum.

    Returns:
        torch.tensor -- The negative log-likelihood.
    """

    def forward(self, outputs, labels):
        return self.survodeloss(outputs, labels)

    def survodeloss(self, outputs, labels):
        def _reduction(loss, reduction):
            if reduction == 'none':
                return loss
            elif reduction == 'mean':
                return loss.mean()
            elif reduction == 'sum':
                return loss.sum()
            raise ValueError(f"`reduction` = {reduction} is not valid. Use 'none', 'mean' or 'sum'.")

        batch_loss = -labels * torch.log(
            outputs["lambda"].clamp(min=1e-8)) + outputs["Lambda"]
        return _reduction(batch_loss, self.reduction)


class BEHRT_SODENXcal(Bert.modeling.BertPreTrainedModel):
    def __init__(self, config, num_labels):
        super(BEHRT_SODENXcal, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = NonCoxFuncModel(config)
        self.apply(self.init_bert_weights)
        self.binary = False
        self.device = 0

    def forward(self, input_ids, age_ids, seg_ids, posi_ids, attention_mask, label, labelfloat, time2event,
                fulleval=False):
        _, pooled_output = self.bert(input_ids, age_ids, seg_ids, posi_ids, attention_mask,
                                     output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(time2event.squeeze(-1), torch.zeros_like(time2event.squeeze(-1)).to(self.device),
                                 pooled_output, fulleval)

        loss_fct = SurvODEloss(reduction='mean')
        lossout = loss_fct(logits, labelfloat.squeeze(-1)).squeeze(0)
        outlog = logits['lambda']
        outfull = logits['hazard_seq']

        return outlog, outfull, loss_fct(logits, labelfloat.view(-1))



def get_bin_for_time(tte, bin_boundaries):
    # we exclude bin_boundaries[0] so that the below sum equals 0
    # for a datapoint in the 0th bin.

    # we exclude bin_boundaries[-1] because otherwise a time larger
    # than all bin boundaries would have a sum thats K instead of K-1

    boundaries_to_consider = bin_boundaries[1:-1].view(1, -1)
    # The bin intex tte is in
    tte_cat = (tte > boundaries_to_consider).sum(dim=-1)
    return tte_cat


def cat_bin_target(tte, label, bin_boundaries, global_params):
    is_alive = label
    tte = tte
    bin_boundaries = torch.tensor(bin_boundaries).to(global_params['device'])

    batch_sz = tte.size()[0]

    tte_cat = get_bin_for_time(tte, bin_boundaries)

    tte = tte.squeeze()

    # there are K+1 bin boundaries
    K = (bin_boundaries[:-1]).size()[0]
    lower_boundaries = bin_boundaries[:-1].view(1, -1)
    upper_boundaries = bin_boundaries[1:].view(1, -1)
    assert lower_boundaries.size()[-1] == K

    # BATCH_SZ x K
    tte_as = lower_boundaries.repeat(batch_sz, 1)[range(batch_sz), tte_cat]
    tte_bs = upper_boundaries.repeat(batch_sz, 1)[range(batch_sz), tte_cat]

    ratio = (tte - tte_as) / (tte_bs - tte_as)
    ratio[tte > bin_boundaries[-1]] = 1.0
    ratio[tte < bin_boundaries[0]] = 0.0
    ratio = ratio.unsqueeze(-1)
    tte_cat = tte_cat.float().unsqueeze(-1)

    '''
    OVERWRITING TO COMPUTE NON CENSORED DCAL
    ON CENSORED POINTS THAT ARE IN MAX BIN
    '''
    max_bin = len(bin_boundaries) - 1.
    not_max_bin = (tte_cat < max_bin).long()
    is_alive = (is_alive * not_max_bin)
    tgt = torch.cat((tte_cat, is_alive, ratio), dim=-1)
    tgt = tgt.to(global_params['device'])
    return tgt


def d_calibration(points, is_alive, nbins=20, differentiable=False, gamma=10000.0, device='cpu'):
    # each "point" in points is a time for datapoint i mapped through the model CDF
    # each such time_i is a survival time if not censored or a time sampled
    # uniformly in (censor time, max time
    # compute empirical cdf of cdf-mapped-times
    # Move censored points with cdf values greater than 1 - 1e-4 t0 uncensored group
    new_is_alive = is_alive.detach().clone()
    new_is_alive[points > 1. - 1e-4] = 0

    points = points.to(device).view(-1, 1)
    # print(points[:200])
    # BIN DEFNITIONS
    # BIN DEFNITIONS
    # BIN DEFNITIONS
    # BIN DEFNITIONS
    # BIN DEFNITIONS
    # BIN DEFNITIONS
    bin_width = 1.0 / nbins
    bin_indices = torch.arange(nbins).view(1, -1).float().to(device)
    bin_a = bin_indices * bin_width  # + 0.02*torch.rand(size=bin_indices.shape)
    noise = 1e-6 / nbins * torch.rand(size=bin_indices.shape).to(device)
    if not differentiable:
        noise = noise * 0.
    cum_noise = torch.cumsum(noise, dim=1)
    bin_width = torch.tensor([bin_width] * nbins).to(device) + cum_noise
    bin_b = bin_a + bin_width

    bin_b_max = bin_b[:, -1]
    bin_b = bin_b / bin_b_max
    bin_a[:, 1:] = bin_b[:, :-1]
    bin_width = bin_b - bin_a

    # CENSORED POINTS
    points_cens = points[new_is_alive.long() == 1]
    upper_diff_for_soft_cens = bin_b - points_cens
    # To solve optimization issue, we change the first left bin boundary to be -1.;
    # we change the last right bin boundary to be 2.
    bin_b[:, -1] = 2.
    bin_a[:, 0] = -1.
    lower_diff_cens = points_cens - bin_a  # p - a
    upper_diff_cens = bin_b - points_cens  # b - p
    diff_product_cens = lower_diff_cens * upper_diff_cens
    # NON-CENSORED POINTS

    if differentiable:
        # sigmoid(gamma*(p-a)*(b-p))
        bin_index_ohe = torch.sigmoid(gamma * diff_product_cens)
        exact_bins_next = torch.sigmoid(-gamma * lower_diff_cens)
    else:
        # (p-a)*(b-p)
        bin_index_ohe = (lower_diff_cens >= 0).float() * (upper_diff_cens > 0).float()
        exact_bins_next = (lower_diff_cens <= 0).float()  # all bins after correct bin

    EPS = 1e-13
    right_censored_interval_size = 1 - points_cens + EPS

    # each point's distance from its bin's upper limit
    upper_diff_within_bin = (upper_diff_for_soft_cens * bin_index_ohe)

    # assigns weights to each full bin that is larger than the point
    # full_bin_assigned_weight = exact_bins*bin_width
    # 1 / right_censored_interval_size is the density of the uniform over [F(c),1]
    full_bin_assigned_weight = (exact_bins_next * bin_width.view(1, -1) / right_censored_interval_size.view(-1, 1)).sum(
        0)
    partial_bin_assigned_weight = (upper_diff_within_bin / right_censored_interval_size).sum(0)
    assert full_bin_assigned_weight.shape == partial_bin_assigned_weight.shape, (
    full_bin_assigned_weight.shape, partial_bin_assigned_weight.shape)

    # NON-CENSORED POINTS
    # NON-CENSORED POINTS
    # NON-CENSORED POINTS
    # NON-CENSORED POINTS
    # NON-CENSORED POINTS
    # NON-CENSORED POINTS
    points_uncens = points[new_is_alive.long() == 0]
    # compute p - a and b - p
    lower_diff = points_uncens - bin_a
    upper_diff = bin_b - points_uncens
    diff_product = lower_diff * upper_diff
    assert lower_diff.shape == upper_diff.shape, (lower_diff.shape, upper_diff.shape)
    assert lower_diff.shape == (points_uncens.shape[0], bin_a.shape[1])
    # NON-CENSORED POINTS

    if differentiable:
        # sigmoid(gamma*(p-a)*(b-p))
        soft_membership = torch.sigmoid(gamma * diff_product)
        fraction_in_bins = soft_membership.sum(0)
        # print('soft_membership', soft_membership)
    else:
        # (p-a)*(b-p)
        exact_membership = (lower_diff >= 0).float() * (upper_diff > 0).float()
        fraction_in_bins = exact_membership.sum(0)

    assert fraction_in_bins.shape == (nbins,), fraction_in_bins.shape

    frac_in_bins = (fraction_in_bins + full_bin_assigned_weight + partial_bin_assigned_weight) / points.shape[0]
    return torch.pow(frac_in_bins - bin_width, 2).sum()


def compute_xcal(pred_params, tgt, global_params):
    cdf = get_cdf_val(pred_params, tgt, global_params)


    # ratio not used here, no need to check for 3rd dim
    tte, is_alive = tgt[:, 0], tgt[:, 1]

    d_cal = d_calibration(points=cdf,
                          is_alive=is_alive,
                          nbins=20,
                          differentiable=True,
                          gamma=global_params['gamma'],
                          device=global_params['device'])

    return d_cal


def get_cdf_val(pred_params, tgt,global_params):
    pred = pred_params_to_dist(pred_params,global_params)

    tte, is_alive, ratio = tgt[:, 0], tgt[:, 1], tgt[:, 2]
    cdf = pred.cdf(tte, ratio)

    return cdf


def pred_params_to_dist(pred_params,global_params):
    pred = pred_params_to_cat(pred_params,global_params)
    return pred


def pred_params_to_cat(pred_params,global_params):
    pred = CatDist(pred_params, global_params)
    return pred


def get_mean_bins(pred_params, mid_points):
    mid_points = torch.Tensor(mid_points).to(DEVICE)
    probs = torch.softmax(pred_params, dim=-1)
    mean_times = torch.mm(probs, (mid_points.unsqueeze(-1).float())).flatten()
    return mean_times


class CatDist():
    def __init__(self, pred_params, global_params):
        self.pred_params = pred_params
        self.interpolate = True
        self.global_params = global_params

    def predict_time(self):
        pred_time = get_mean_bins(self.pred_params, self.global_params['mid_points'])
        return pred_time

    def log_prob(self, times):
        times = times.long()
        log_unnormalized_probs = self.pred_params[torch.arange(self.pred_params.shape[0]), times]
        normalizer = torch.logsumexp(self.pred_params, dim=-1)
        return log_unnormalized_probs - normalizer

    def cdf(self, times, ratio):
        times = times.long()
        params = self.pred_params
        batch_sz = params.size()[0]
        K = params.size()[-1]
        times = times.view(-1, 1)
        indices = torch.arange(K).view(1, -1).to(self.global_params['device'])

        '''
        Linear Interpolation for CDF
        '''
        # compute some masks

        nonzeroindex = (times != 0).long()

        mask1 = ((times - 1) == indices).float()
        # 1 up to and including correct bin, then zeros
        mask2 = (times == indices).float()
        all_probs = params
        prob_k = all_probs[range(batch_sz), times.squeeze()] - (
                    nonzeroindex.squeeze() * all_probs[range(batch_sz), (times - (1 * nonzeroindex)).squeeze()])
        cdf_km1 = (all_probs * mask1).sum(dim=-1)

        cdf_k = (all_probs * mask2).sum(dim=-1)

        assert torch.all((cdf_k - (cdf_km1 + prob_k)).abs() < 1e-4)

        if not self.interpolate:
            return cdf_k
        else:
            '''
            define cdf_i(k) = sum_{j=0}^{k} prob_i(j)

            linear_interpolation for bin k
            a_k, b_k, t_i     bin(t_i) = k   if  a_k < t_i < b_k

            without interpolation
            cdf_i(bin(t_i)) = cdf_i(k)

            with interpolation:
            ratio_ik = (t_i - a_k) / (b_k - a_k)  
            cdf_i(t_i) = cdf_i(k-1) + prob_i(k) * ratio_ik
            '''

            probs = cdf_km1 + prob_k * ratio

            if torch.any(probs > 1.0 + 1e-4):
                print('*' * 39)
                print('Warning: cdf is greater than one!')
                print('probs', probs[probs > 1. + 1e-7])
                print('ratio', ratio[probs > 1. + 1e-7])
                print('*' * 39)

            if bad(probs):
                print('probs is nan', bad(probs))

            interpolated_cdf = probs
            return interpolated_cdf


def bad(tensor):
    return torch.any(torch.isnan(tensor))


def get_bin_boundaries(train, dl_train):
    all_times = []
    all_is_alive = []

    all_times = torch.tensor(train.time2event.values)
    all_is_alive = torch.tensor(train.label.values).long()
    all_times_censored = all_times[all_is_alive == 1].cpu().numpy()
    all_times = all_times[all_is_alive == 0]
    all_times = all_times.cpu().numpy()
    print(np.unique(all_times))
    bin_boundaries = np.linspace(min(all_times) - 1, max(all_times), max(all_times) + 1)
    print("bin boundaries", bin_boundaries)
    mid_points = (bin_boundaries[1:] + bin_boundaries[:-1]) / 2.

    return bin_boundaries, mid_points




