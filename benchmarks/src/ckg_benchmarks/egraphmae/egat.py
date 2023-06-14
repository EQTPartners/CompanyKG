"""
Copyright (C) eqtgroup.com Ltd 2023
https://github.com/EQTPartners/CompanyKG
License: MIT, https://github.com/EQTPartners/CompanyKG/LICENSE.md
"""

import dgl.function as fn
import torch
import torch.nn as nn
from dgl.ops import edge_softmax
from dgl.utils import expand_as_pair


class EGAT(nn.Module):
    def __init__(
        self,
        in_dim,
        num_hidden,
        out_dim,
        num_layers,
        nhead,
        nhead_out,
        feat_drop,
        attn_drop,
        negative_slope,
        residual,
        norm,
        in_dim_e,
        num_hidden_e,
        out_dim_e,
        concat_out=False,
        encoding=False,
    ):
        super(EGAT, self).__init__()
        self.out_dim = out_dim
        self.num_heads = nhead
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = nn.PReLU()  # fix to PReLU
        self.concat_out = concat_out
        self.in_dim_e = in_dim_e
        self.num_hidden_e = num_hidden_e
        self.out_dim_e = out_dim_e

        last_activation = self.activation if encoding else None
        last_residual = encoding and residual
        last_norm = norm if encoding else None

        if num_layers == 1:
            self.gat_layers.append(
                EGATConv(
                    in_dim,
                    in_dim_e,
                    out_dim,
                    out_dim_e,
                    nhead_out,
                    feat_drop,
                    attn_drop,
                    negative_slope,
                    last_residual,
                    norm=last_norm,
                )
            )
        else:
            # input projection (no residual)
            self.gat_layers.append(
                EGATConv(
                    in_dim,
                    in_dim_e,
                    num_hidden,
                    num_hidden_e,
                    nhead,
                    feat_drop,
                    attn_drop,
                    negative_slope,
                    residual,
                    self.activation,
                    norm=norm,
                )
            )
            # hidden layers
            for l in range(1, num_layers - 1):
                # due to multi-head, the in_dim = num_hidden * num_heads
                self.gat_layers.append(
                    EGATConv(
                        num_hidden * nhead,
                        num_hidden_e * nhead,
                        num_hidden,
                        num_hidden_e,
                        nhead,
                        feat_drop,
                        attn_drop,
                        negative_slope,
                        residual,
                        self.activation,
                        norm=norm,
                    )
                )
            # output projection
            self.gat_layers.append(
                EGATConv(
                    num_hidden * nhead,
                    num_hidden_e * nhead,
                    out_dim,
                    out_dim_e,
                    nhead_out,
                    feat_drop,
                    attn_drop,
                    negative_slope,
                    last_residual,
                    activation=last_activation,
                    norm=last_norm,
                )
            )

        self.head = nn.Identity()

    def forward(self, g, inputs, eputs, return_hidden=False):
        h = inputs
        e = eputs
        hidden_list = []

        for l in range(self.num_layers):
            h, e = self.gat_layers[l](g, h, e)
            hidden_list.append(h)
            # h = h.flatten(1)
        # output projection
        if return_hidden:
            return self.head(h), self.head(e), hidden_list
        else:
            return self.head(h), self.head(e)

    def reset_classifier(self, num_classes):
        self.head = nn.Linear(self.num_heads * self.out_dim, num_classes)


class EGATConv(nn.Module):
    def __init__(
        self,
        in_node_feats,
        in_edge_feats,
        out_node_feats,
        out_edge_feats,
        num_heads,
        feat_drop=0.0,
        attn_drop=0.0,
        negative_slope=0.2,
        residual=False,
        activation=None,
        bias=True,
        norm=None,
    ):

        super().__init__()
        self._num_heads = num_heads
        self._in_src_node_feats, self._in_dst_node_feats = expand_as_pair(in_node_feats)
        self._out_node_feats = out_node_feats
        self._out_edge_feats = out_edge_feats

        if isinstance(in_node_feats, tuple):
            self.fc_node_src = nn.Linear(
                self._in_src_node_feats, out_node_feats * num_heads, bias=False
            )
            self.fc_ni = nn.Linear(
                self._in_src_node_feats, out_edge_feats * num_heads, bias=False
            )
            self.fc_nj = nn.Linear(
                self._in_dst_node_feats, out_edge_feats * num_heads, bias=False
            )
        else:
            self.fc_node_src = nn.Linear(
                self._in_src_node_feats, out_node_feats * num_heads, bias=False
            )
            self.fc_ni = nn.Linear(
                self._in_src_node_feats, out_edge_feats * num_heads, bias=False
            )
            self.fc_nj = nn.Linear(
                self._in_src_node_feats, out_edge_feats * num_heads, bias=False
            )

        self.fc_fij = nn.Linear(in_edge_feats, out_edge_feats * num_heads, bias=False)
        self.attn = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_edge_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if bias:
            self.bias = nn.Parameter(
                torch.FloatTensor(size=(num_heads * out_edge_feats,))
            )
        else:
            self.register_buffer("bias", None)

        if residual:
            if self._in_dst_node_feats != out_node_feats * num_heads:
                self.res_fc = nn.Linear(
                    self._in_dst_node_feats, num_heads * out_node_feats, bias=False
                )
            else:
                self.res_fc = nn.Identity()
        else:
            self.register_buffer("res_fc", None)

        self.reset_parameters()

        self.activation = activation

        self.norm = norm
        if norm is not None:
            self.norm = norm(num_heads * out_node_feats)
            self.norm_e = norm(num_heads * out_edge_feats)

    def reset_parameters(self):
        """
        Reinitialize learnable parameters.
        """
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.fc_node_src.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_ni.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_fij.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_nj.weight, gain=gain)
        nn.init.xavier_normal_(self.attn, gain=gain)
        nn.init.constant_(self.bias, 0)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def forward(self, graph, nfeats, efeats, get_attention=False):

        with graph.local_scope():
            if (graph.in_degrees() == 0).any():
                raise RuntimeError(
                    "There are 0-in-degree nodes in the graph, "
                    "output for those nodes will be invalid. "
                    "This is harmful for some applications, "
                    "causing silent performance regression. "
                    "Adding self-loop on the input graph by "
                    "calling `g = dgl.add_self_loop(g)` will resolve "
                    "the issue."
                )

            # calc edge attention
            # same trick way as in dgl.nn.pytorch.GATConv, but also includes edge feats
            # https://github.com/dmlc/dgl/blob/master/python/dgl/nn/pytorch/conv/gatconv.py
            if isinstance(nfeats, tuple):
                nfeats_src, nfeats_dst = nfeats
                nfeats_src = self.feat_drop(nfeats_src)
                nfeats_dst = self.feat_drop(nfeats_dst)
                dst_prefix_shape = nfeats_dst.shape[:-1]
            else:
                nfeats_src = nfeats_dst = self.feat_drop(nfeats)
                dst_prefix_shape = nfeats.shape[:-1]

            f_ni = self.fc_ni(nfeats_src)
            f_nj = self.fc_nj(nfeats_dst)
            f_fij = self.fc_fij(efeats)

            graph.srcdata.update({"f_ni": f_ni})
            graph.dstdata.update({"f_nj": f_nj})
            # add ni, nj factors
            graph.apply_edges(fn.u_add_v("f_ni", "f_nj", "f_tmp"))
            # add fij to node factor
            f_out = graph.edata.pop("f_tmp") + f_fij
            # f_out = f_fij
            if self.bias is not None:
                f_out = f_out + self.bias
            f_out = self.leaky_relu(f_out)
            f_out = f_out.view(-1, self._num_heads, self._out_edge_feats)
            # compute attention factor
            e = (f_out * self.attn).sum(dim=-1).unsqueeze(-1)
            graph.edata["a"] = self.attn_drop(edge_softmax(graph, e))
            # graph.edata['a'] = e
            graph.srcdata["h_out"] = self.fc_node_src(nfeats_src).view(
                -1, self._num_heads, self._out_node_feats
            )
            # calc weighted sum
            graph.update_all(fn.u_mul_e("h_out", "a", "m"), fn.sum("m", "h_out"))

            h_out = graph.dstdata["h_out"].view(
                -1, self._num_heads, self._out_node_feats
            )

            # residual
            if self.res_fc is not None:
                # Use -1 rather than self._num_heads to handle broadcasting
                resval = self.res_fc(nfeats_dst).view(
                    *dst_prefix_shape, -1, self._out_node_feats
                )
                h_out = h_out + resval

            rst_h_out = h_out.flatten(1)
            rst_f_out = f_out.flatten(1)

            if self.norm is not None:
                rst_h_out = self.norm(rst_h_out)
                rst_f_out = self.norm_e(rst_f_out)

            # activation
            if self.activation:
                rst_h_out = self.activation(rst_h_out)
                rst_f_out = self.activation(rst_f_out)

            if get_attention:
                return rst_h_out, rst_f_out, graph.edata.pop("a")
            else:
                return rst_h_out, rst_f_out
