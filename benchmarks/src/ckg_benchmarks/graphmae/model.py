# This implementation is adapted from GraphMAE
# https://github.com/THUDM/GraphMAE

from functools import partial
from itertools import chain

import dgl
import numpy as np
import torch
from ckg_benchmarks.graphmae.gat import GAT


def sce_loss(x, y, alpha=3):
    x = torch.nn.functional.normalize(x, p=2, dim=-1)
    y = torch.nn.functional.normalize(y, p=2, dim=-1)
    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    return loss.mean()


def mask_edge(graph, mask_prob):
    E = graph.num_edges()
    mask_rates = torch.FloatTensor(np.ones(E) * mask_prob)
    masks = torch.bernoulli(1 - mask_rates)
    mask_idx = masks.nonzero().squeeze(1)
    return mask_idx


def drop_edge(graph, drop_rate, return_edges=False):
    if drop_rate <= 0:
        return graph

    n_node = graph.num_nodes()
    edge_mask = mask_edge(graph, drop_rate)
    src = graph.edges()[0]
    dst = graph.edges()[1]

    nsrc = src[edge_mask]
    ndst = dst[edge_mask]

    ng = dgl.graph((nsrc, ndst), num_nodes=n_node)
    ng = ng.add_self_loop()

    dsrc = src[~edge_mask]
    ddst = dst[~edge_mask]

    if return_edges:
        return ng, (dsrc, ddst)
    return ng


class GraphMAE(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_hidden: int,
        num_layers: int,
        feat_drop: float,
        attn_drop: float,
        nhead: int,
        nhead_out: int = 1,
        mask_rate: float = 0.5,
        drop_edge_rate: float = 0.5,
        replace_rate: float = 0.15,
        alpha_l: float = 3,
        concat_hidden: bool = False,
    ):
        super(GraphMAE, self).__init__()
        self._mask_rate = mask_rate
        self._drop_edge_rate = drop_edge_rate
        self._output_hidden_size = num_hidden
        self._concat_hidden = concat_hidden
        self._norm = torch.nn.LayerNorm  # fix to LayerNorm

        self._replace_rate = replace_rate
        self._mask_token_rate = 1 - self._replace_rate

        assert num_hidden % nhead == 0
        assert num_hidden % nhead_out == 0

        enc_num_hidden = num_hidden // nhead
        enc_nhead = nhead

        dec_in_dim = num_hidden
        dec_num_hidden = num_hidden // nhead_out

        # build encoder
        self.encoder = GAT(
            in_dim=in_dim,
            num_hidden=enc_num_hidden,
            out_dim=enc_num_hidden,
            num_layers=num_layers,
            nhead=enc_nhead,
            nhead_out=enc_nhead,
            concat_out=True,
            feat_drop=feat_drop,
            attn_drop=attn_drop,
            negative_slope=0.2,
            residual=True,
            norm=self._norm,
            encoding=True,
        )

        # build decoder for attribute prediction
        self.decoder = GAT(
            in_dim=dec_in_dim,
            num_hidden=dec_num_hidden,
            out_dim=in_dim,
            num_layers=1,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=True,
            feat_drop=feat_drop,
            attn_drop=attn_drop,
            negative_slope=0.2,
            residual=True,
            norm=self._norm,
            encoding=False,
        )

        self.enc_mask_token = torch.nn.Parameter(torch.zeros(1, in_dim))
        if concat_hidden:
            self.encoder_to_decoder = torch.nn.Linear(
                dec_in_dim * num_layers, dec_in_dim, bias=False
            )
        else:
            self.encoder_to_decoder = torch.nn.Linear(
                dec_in_dim, dec_in_dim, bias=False
            )

        # * setup loss function
        self.criterion = self.setup_loss_fn(alpha_l)

    @property
    def output_hidden_dim(self):
        return self._output_hidden_size

    def setup_loss_fn(self, alpha_l):
        return partial(sce_loss, alpha=alpha_l)

    def encoding_mask_noise(self, g, x, mask_rate=0.3):
        num_nodes = g.num_nodes()
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_rate * num_nodes)

        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[:num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]

        if self._replace_rate > 0:
            num_noise_nodes = int(self._replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[
                perm_mask[: int(self._mask_token_rate * num_mask_nodes)]
            ]
            noise_nodes = mask_nodes[
                perm_mask[-int(self._replace_rate * num_mask_nodes) :]
            ]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[
                :num_noise_nodes
            ]

            out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token
        use_g = g.clone()

        return use_g, out_x, (mask_nodes, keep_nodes)

    def forward(self, g, x):
        # ---- attribute reconstruction ----
        loss = self.mask_attr_prediction(g, x)
        loss_item = {"loss": loss.item()}
        return loss, loss_item

    def mask_attr_prediction(self, g, x):
        pre_use_g, use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(
            g, x, self._mask_rate
        )

        if self._drop_edge_rate > 0:
            use_g, masked_edges = drop_edge(
                pre_use_g, self._drop_edge_rate, return_edges=True
            )
        else:
            use_g = pre_use_g

        enc_rep, all_hidden = self.encoder(use_g, use_x, return_hidden=True)
        if self._concat_hidden:
            enc_rep = torch.cat(all_hidden, dim=1)

        # ---- attribute reconstruction ----
        rep = self.encoder_to_decoder(enc_rep)

        recon = self.decoder(pre_use_g, rep)

        x_init = x[mask_nodes]
        x_rec = recon[mask_nodes]

        loss = self.criterion(x_rec, x_init)
        return loss

    def embed(self, g, x):
        rep = self.encoder(g, x)
        return rep

    @property
    def enc_params(self):
        return self.encoder.parameters()

    @property
    def dec_params(self):
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()])
