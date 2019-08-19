
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import torch
from . import data
from . import models
from . import visual
from . import trainers
import logging.config
__version__ = u'0.4.1'
logging.config.dictConfig({
    u'version': 1,
    u'disable_existing_loggers': False,
    u'formatters': {
        u'standard': {
            u'format': u'%(asctime)-15s %(message)s',
        },
    },
    u'handlers': {
        u'console': {
            u'level': u'INFO',
            u'class': u'logging.StreamHandler',
            u'formatter': u'standard',
            u'stream': u'ext://sys.stdout',
        },
    },
    u'loggers': {
        u'flair': {
            u'handlers': [u'console'],
            u'level': u'INFO',
            u'propagate': False,
        },
    },
    u'root': {
        u'handlers': [u'console'],
        u'level': u'WARNING',
    },
})
logger = logging.getLogger(u'flair')
device = None
if torch.cuda.is_available():
    device = torch.device(u'cuda:0')
else:
    device = torch.device(u'cpu')
