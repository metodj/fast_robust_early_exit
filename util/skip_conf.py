import numpy as np
import torch

from transformers import AutoConfig

from typing import List


def softmax_confidence(
    logits: torch.Tensor = None,
    hidden_states: torch.Tensor = None,
    classifier: torch.nn.Linear = None,
    hidden_states_all: List = None
):
    assert logits is not None
    probs = torch.softmax(logits, dim=-1)
    top_2 = torch.topk(probs, dim=-1, k=2)[0]

    return (top_2[..., 0] - top_2[..., 1]).squeeze()


def meta_confidence(
    logits: torch.Tensor = None,
    hidden_states: torch.Tensor = None,
    classifier: torch.nn.Linear = None,
    hidden_states_all: List = None
):
    assert hidden_states is not None
    assert classifier is not None
    
    preds = classifier(hidden_states)
    probs = torch.softmax(preds, dim=-1)
    return probs[..., 1].squeeze()


def state_saturation_confidence(
    logits: torch.Tensor = None,
    hidden_states: torch.Tensor = None,
    classifier: torch.nn.Linear = None,
    hidden_states_all: List[torch.Tensor] = None):
    assert len(hidden_states_all) >= 2
    state_prev, state_curr = hidden_states_all[-2].reshape(-1), hidden_states_all[-1].reshape(-1)
    assert state_prev.shape == state_curr.shape

    # compute cosine similarity
    dot_prod = torch.dot(state_prev, state_curr)
    norm_prev = torch.norm(state_prev)
    norm_curr = torch.norm(state_curr)

    cos_sim = dot_prod / (norm_prev * norm_curr)
    assert -0.0001 <= cos_sim <= 1.0001, f'Cosine similarity must be between 0 and 1, but got {cos_sim}'
    return cos_sim



def get_confidence_class(key):

    _conf_class_map = {
        'softmax': softmax_confidence,
        'meta': meta_confidence,
        'state-saturation': state_saturation_confidence,
    }

    if key in _conf_class_map:
        return _conf_class_map[key]
    else:
        raise ValueError('Invalid confidence measure: {}'.format(key))


def get_skip_mask(
    logits: torch.Tensor = None,
    hidden_states: torch.Tensor = None,
    classifier: torch.nn.Linear = None,
    config: AutoConfig = None,
    pos_time: int = 1,
    adapt_threshold: float = None,
    return_conf=False,
    hidden_state_all: List = None
):
    assert config.exit_conf_type is not None or config.shallow2deep_conf_type is not None

    # print(len(hidden_state_all))
    # print(hidden_state_all[0].shape)
    # print(hidden_states.shape)
    # print(torch.equal(hidden_state_all[-1], hidden_states))
    # print(torch.equal(hidden_state_all[-2], hidden_states))

    if config.exit_conf_type is not None:
        key = config.exit_conf_type
        if config.exit_position_temp is not None:
            # decays the confidence threshold with decoding time stp.        
            correct_by_pos = lambda i: config.exit_conf_threshold * np.exp(
                - config.exit_position_temp * i / config.max_answer_length
            ) / 10 + 9 * config.exit_conf_threshold / 10
            threshold = correct_by_pos(pos_time)
        else:
            threshold = config.exit_conf_threshold
    elif config.shallow2deep_conf_type is not None:
        key = config.shallow2deep_conf_type
        threshold = config.shallow2deep_conf_threshold if adapt_threshold is None else adapt_threshold

    conf_measure = get_confidence_class(key=key)    
    conf = conf_measure(
        logits=logits, 
        hidden_states=hidden_states, 
        classifier=classifier,
        hidden_states_all=hidden_state_all
    )
    # print(conf)
    mask = torch.where(conf <= threshold, 0., 1.).bool()
    
    if not return_conf:
        return mask.item()  # False (0) and True (1) denote keep and exit
    else:
        return mask.item(), conf.item()