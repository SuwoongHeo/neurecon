import torch.nn as nn
import torch as T
import scipy.sparse as sp
from scipy.sparse import coo_matrix
import numpy as np


# https://github.com/tkipf/gcn/blob/master/gcn/utils.py#L122
def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    adj = (adj > 0).astype(adj.dtype)
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0], dtype=adj.dtype))
    return adj_normalized

def batch_mm(matrix, matrix_batch: T.Tensor):
    """
    :param matrix: Sparse or dense matrix, size (m, n).
    :param matrix_batch: Batched dense matrices, size (b, n, k).
    :return: The batched matrix-matrix product, size (m, n) x (b, n, k) = (b, m, k).
    """
    batch_size = matrix_batch.shape[0]
    # Stack the vector batch into columns. (b, n, k) -> (n, b, k) -> (n, b*k)
    vectors = matrix_batch.transpose(0, 1).reshape(matrix.shape[1], -1)

    # A matrix-matrix product is a batched matrix-vector product of the columns.
    # And then reverse the reshaping. (m, n) x (n, b*k) = (m, b*k) -> (m, b, k) -> (b, m, k)
    return matrix.mm(vectors.contiguous()).view(matrix.shape[0], batch_size, -1).transpose(1, 0)


class GConv(nn.Module):
    """Simple GCN layer
    Similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True, activation=None):
        super(GConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation

        self.weight = nn.Parameter(T.zeros((in_features, out_features), dtype=T.float))
        # Following https://github.com/Tong-ZHAO/Pixel2Mesh-Pytorch/blob/a0ae88c4a42eef6f8f253417b97df978db842708/model/gcn_layers.py#L45
        # This seems to be different from the original implementation of P2M
        self.loop_weight = nn.Parameter(T.zeros((in_features, out_features), dtype=T.float))
        if bias:
            self.bias = nn.Parameter(T.zeros((out_features,), dtype=T.float))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight.data)
        nn.init.xavier_uniform_(self.loop_weight.data)

    def forward(self, inputs, adj_mat):
        support: T.Tensor = T.matmul(inputs, self.weight)
        support_loop = T.matmul(inputs, self.loop_weight)
        output = batch_mm(adj_mat, support) + support_loop
        if self.bias is not None:
            ret = output + self.bias
        else:
            ret = output
        return self.activation(ret)

    def extra_repr(self) -> str:
        s = f'{self.in_features}, {self.out_features}, {self.activation}'
        return s

# todo, use GCNII by using torch geometric or implementing by myself
# https://github.com/chennnM/GCNII/
