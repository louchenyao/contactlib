import logging

logger = logging.getLogger("protein_search")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler()) # output to stderr
