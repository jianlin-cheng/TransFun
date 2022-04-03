#!/usr/bin/env python
import pandas as pd
import click as ck
from utils import (
    get_gene_ontology,
    get_anchestors,
    FUNC_DICT,
    EXP_CODES)
from collections import deque
from aaindex import is_ok


DATA_ROOT = 'data/swiss/'