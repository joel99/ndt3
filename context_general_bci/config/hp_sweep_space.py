# For sampling with utils.generate_search

# Really the majority of experiments are about _data_
# so `dataset` (or embedding strat) is the configured item, and we sweep other nuisance params.
# Formatted according to expectations in `halton.py`

sweep_space = {
    "simpler_lr_sweep": {
        "model.lr_init": {
            'feasible_points': [4e-5, 7e-5, 1e-4]
        }
    },
    "simple_lr_sweep": {
        "model.lr_init": {
            'feasible_points': [1e-5, 3e-5, 5e-5, 1e-4]
        }
    },
    "nlb_tune_2": {
        'model.task.mask_ratio': {
            'feasible_points': [0.01, 0.05, 0.1],
        },
        'model.tune_decay': {
            'feasible_points': [0.75, 0.85, 0.95],
        },
        'model.lr_ramp_steps': {
            'feasible_points': [50, 100, 200],
        },
        'model.task.task_weights': {
            'feasible_points': [(1., 1.), (0.5, 1.), (1., 0.5), (0.25, 1.), (1., 0.25)],
        },
    },
    "nlb_tune": {
        'model.task.mask_ratio': {
            'feasible_points': [0.1, 0.2, 0.3, 0.4, 0.5],
        },
        'model.tune_decay': {
            'feasible_points': [0.3, 0.5, 0.7, 0.9],
        },
        'model.lr_ramp_steps': {
            'feasible_points': [50, 250, 500, 750, 1000],
        },
        # TODO consider other beta for optimizer
    },
    "nlb_parity": {
        "model.dropout": {
            'feasible_points': [0.05, 0.1, 0.2]
        },
        "model.weight_decay": {
            'feasible_points': [1e-3, 5e-3, 1e-2, 5e-2]
        },
        "model.task.mask_ratio": {
            'feasible_points': [0.1, 0.25, 0.5, 0.75]
        },
        "model.lr_init": {
            'min': 1e-5,
            'max': 1e-3,
            'scaling': 'log',
        },
        "model.task.freeze_backbone": {
            'feasible_points': [True, False]
        },
    },
    "base_v2": {
        "model.dropout": {
            'feasible_points': [0.1, 0.2, 0.3, 0.4]
        },
        "model.weight_decay": {
            'min': 1e-3,
            'max': 5e-2,
            'scaling': 'log',
        },
        "model.hidden_size": {
            'feasible_points': [128, 256]
        },
        "seed": {
            'feasible_points': [0, 1, 2]
        },
    },
    "reg_tight": { # this may be one strategy, or, bigger models might even be better?
        "model.dropout": {
            'feasible_points': [0.05, 0.1, 0.2, 0.3]
        },
        "model.weight_decay": {
            'min': 5e-3,
            'max': 1e-1,
            'scaling': 'log',
        }
    },
    "ft_reg": {
        "model.dropout": {
            'feasible_points': [0.1, 0.2, 0.3, 0.4]
        },
        "model.weight_decay": {
            'min': 1e-3,
            'max': 5e-2,
            'scaling': 'log',
        }
    },
    "lr": {
        "model.lr_init": {
            'min': 1e-4,
            'max': 3e-3,
            'scaling': 'log',
        },
        "model.lr_ramp_steps": {
            'feasible_points': [10, 25, 50, 100],
        },
    },
    "lr_and_dropout": {
        "model.lr_init": {
            'min': 1e-4,
            'max': 3e-3,
            'scaling': 'log',
        },
        "model.lr_ramp_steps": {
            'feasible_points': [10, 25, 50, 100],
        },
        "model.dropout": {
            'feasible_points': [0.1, 0.3, 0.5, 0.7] # in lieu of sweeping capacity
        }
    },
    'lr_v3': {
        "model.lr_init": {
            'min': 2e-4,
            'max': 8e-4,
            'scaling': 'log',
        },
        "model.dropout": {
            'feasible_points': [0.05, 0.1, 0.2]
        }
    },
    "lr_v2": {
        "model.lr_init": {
            'min': 2e-4,
            # 'min': 1e-4,
            'max': 1e-3,
            'scaling': 'log',
        },
        "model.dropout": {
            'feasible_points': [0.1, 0.3, 0.5, 0.7] # in lieu of sweeping capacity
        }
    }, # post-mortem. dropout of 0.7 kills unless carefully regulated, don't do this. Sweep hidden size instead.
    "base": {
        # we will use a fixed 6-layer architecture for now, sweep hidden.
        "model.hidden_size": {
            'feasible_points': [128, 256]
        },
        "model.lr_init": {
            'min': 1e-4,
            'max': 3e-3,
            # 'max': 5e-2,
            'scaling': 'log',
        },
        "model.weight_decay": {
            'min': 1e-3,
            'max': 5e-2,
            'scaling': 'log',
        },
        "model.dropout": {
            'feasible_points': [0.1, 0.2, 0.3]
        },
        "seed": {
            'feasible_points': [0, 1, 2]
        },
    },
    "large": {
        "model.hidden_size": {
            'feasible_points': [256, 384, 512, 768]
        },
        "model.lr_init": {
            'min': 1e-4,
            'max': 2e-2,
            'scaling': 'log',
        },
        "model.weight_decay": {
            'min': 1e-3,
            'max': 5e-2,
            'scaling': 'log',
        },
        "model.dropout": {
            'feasible_points': [0.1, 0.2, 0.3]
        },
        "seed": {
            'feasible_points': [0, 1, 2]
        },
    },
    "small_wide": {
        # we will use a fixed 6-layer architecture for now, sweep hidden.
        "model.hidden_size": {
            'feasible_points': [128, 192, 256]
        },
        "model.lr_init": {
            'min': 1e-4,
            'max': 3e-3,
            # 'max': 5e-2,
            'scaling': 'log',
        },
        "model.weight_decay": {
            'min': 1e-3,
            'max': 5e-2,
            'scaling': 'log',
        },
        "model.dropout": {
            'feasible_points': [0.1, 0.2, 0.3]
        },
        "seed": {
            'feasible_points': [0, 1, 2]
        },
    },
}

r"""
- Noted fixed parameters
    - lr schedule (cosine decay) and decay step (interminably long horizon)
    - adam hyperparams besides lr (despite playbook recommendation) - we don't have budget
"""