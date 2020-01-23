import numpy as np

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import onnx
#from onnx_tf.backend import prepare
#import tensorflow as tf
from model import SRNN
import argparse

#TRAFFICPREDICT CODE START
parser = argparse.ArgumentParser()

torch.manual_seed(42)
torch.cuda.manual_seed(42)

# RNN size
parser.add_argument(
    "--node_rnn_size",
    type=int,
    default=64,
    help="Size of Human Node RNN hidden state",
)
parser.add_argument(
    "--edge_rnn_size",
    type=int,
    default=128,
    help="Size of Human Human Edge RNN hidden state",
)

# Input and output size
parser.add_argument(
    "--node_input_size", type=int, default=3, help="Dimension of the node features"
)
parser.add_argument(
    "--edge_input_size",
    type=int,
    default=3,
    help="Dimension of the edge features, the 3th parameter is set to 10",
)
parser.add_argument(
    "--node_output_size", type=int, default=5, help="Dimension of the node output"
)

# Embedding size
parser.add_argument(
    "--node_embedding_size",
    type=int,
    default=64,
    help="Embedding size of node features",
)
parser.add_argument(
    "--edge_embedding_size",
    type=int,
    default=64,
    help="Embedding size of edge features",
)

# Attention vector dimension
parser.add_argument("--attention_size", type=int, default=64, help="Attention size")

# Sequence length
parser.add_argument("--seq_length", type=int, default=10, help="Sequence length")
parser.add_argument(
    "--pred_length", type=int, default=6, help="Predicted sequence length"
)

# Batch size
parser.add_argument("--batch_size", type=int, default=8, help="Batch size")

# Number of epochs
parser.add_argument("--num_epochs", type=int, default=300, help="number of epochs")

# Gradient value at which it should be clipped
parser.add_argument(
    "--grad_clip", type=float, default=10.0, help="clip gradients at this value"
)
# Lambda regularization parameter (L2)
parser.add_argument(
    "--lambda_param",
    type=float,
    default=0.00005,
    help="L2 regularization parameter",
)

# Learning rate parameter
parser.add_argument(
    "--learning_rate", type=float, default=0.01, help="learning rate"
)
# Decay rate for the learning rate parameter
parser.add_argument(
    "--decay_rate", type=float, default=0.99, help="decay rate for the optimizer"
)

# Dropout rate
parser.add_argument("--dropout", type=float, default=0, help="Dropout probability")

# Use GPU or CPU
parser.add_argument(
    "--use_cuda", action="store_true", default=True, help="Use GPU or CPU"
)

args = parser.parse_args()
#TRAFFICPREDICT CODE END

tpredict = SRNN(args)
tpredict.load_state_dict(torch.load('C:/TrafficPredict/srnn/save/srnn_model_271.tar'), strict=False)
