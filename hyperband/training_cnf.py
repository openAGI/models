import tensorflow as tf
from tefla.core.lr_policy import PolyDecayPolicy

cnf = {
    # 'multilabel': True,
    'classification':True,
    'validation_scores': [('validation accuracy', util.accuracy_wrapper), ('validation kappa', util.kappa_wrapper)],
    # 'validation_scores': [('accuracy', tf.contrib.metrics.f1_score)],
    'num_epochs': 5,
    'lr_policy': PolyDecayPolicy(
        base_lr = 0.001
    ),
    'aug_params': {
        'zoom_range': (1 / 1.15, 1.15),
        'rotation_range': (0, 0),
        'shear_range': (0, 0),
        'translation_range': (0, 0),
        'do_flip': False,
        'allow_stretch': False,
    },
    'input_size': (28*28,),

    }
