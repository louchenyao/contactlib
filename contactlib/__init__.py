__version__ = '2.1.5'

import logging

logger = logging.getLogger("protein_search")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler()) # output to stderr

#from contactlib.encoder import Encoder
#from contactlib.search import Searcher