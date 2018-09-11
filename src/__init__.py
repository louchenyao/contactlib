import logging

logger = logging.getLogger("protein_search")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler()) # output to stderr

from src.encoder import Encoder
from src.search import Searcher