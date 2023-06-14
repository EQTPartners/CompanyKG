"""
Copyright (C) eqtgroup.com Ltd 2023
https://github.com/EQTPartners/CompanyKG
License: MIT, https://github.com/EQTPartners/CompanyKG/LICENSE.md
"""

from functools import partial
from itertools import chain

import dgl
import torch
from ckg_benchmarks.egraphmae.egat import EGAT
from ckg_benchmarks.graphmae.model import mask_edge, sce_loss


def drop_edge(graph, drop_rate, e=None, return_edges=False):

    if drop_rate <= 0:
        return graph

    n_node = graph.num_nodes()
    edge_mask = mask_edge(graph, drop_rate)
    src = graph.edges()[0]
    dst = graph.edges()[1]

    nsrc = src[edge_mask]
    ndst = dst[edge_mask]

    ng = dgl.graph((nsrc, ndst), num_nodes=n_node)

    if e is not None:
        ex = e[edge_mask]
        ng.edata["weight"] = ex
    else:
        ex = None

    ng = ng.add_self_loop()

    dsrc = src[~edge_mask]
    ddst = dst[~edge_mask]

    if e is not None:
        if return_edges:
            return ng, ng.edata["weight"], (dsrc, ddst)
        return ng, ng.edata["weight"]
    else:
        if return_edges:
            return ng, None, (dsrc, ddst)
        return ng, None


class EGraphMAE(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_hidden: int,
        num_layers: int,
        feat_drop: float,
        attn_drop: float,
        nhead: int,
        num_edge_features: int,
        num_edge_hidden: int,
        nhead_out: int = 1,
        mask_rate: float = 0.5,
        drop_edge_rate: float = 0.5,
        replace_rate: float = 0.15,
        alpha_l: float = 3,
        concat_hidden: bool = False,
    ):
        super(EGraphMAE, self).__init__()
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
        enc_num_hidden_e = num_edge_hidden // nhead
        # enc_nhead_e = nhead

        dec_in_dim = num_hidden
        dec_num_hidden = num_hidden // nhead_out

        dec_in_dim_e = num_edge_hidden
        dec_num_hidden_e = num_edge_hidden // nhead_out

        # build encoder
        self.encoder = EGAT(
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
            in_dim_e=num_edge_features,
            num_hidden_e=enc_num_hidden_e,
            out_dim_e=enc_num_hidden_e,
            encoding=True,
        )

        # build decoder for attribute prediction
        self.decoder = EGAT(
            in_dim=dec_in_dim,
            num_hidden=dec_num_hidden,
            out_dim=in_dim,
            num_layers=1,
            nhead=nhead,
            nhead_out=nhead_out,
            feat_drop=feat_drop,
            attn_drop=attn_drop,
            negative_slope=0.2,
            residual=True,
            norm=self._norm,
            concat_out=True,
            in_dim_e=dec_in_dim_e,
            num_hidden_e=dec_num_hidden_e,
            out_dim_e=num_edge_features,
            encoding=False,
        )

        self.enc_mask_token = torch.nn.Parameter(torch.zeros(1, in_dim))
        if concat_hidden:
            self.encoder_to_decoder = torch.nn.Linear(
                dec_in_dim * num_layers, dec_in_dim, bias=False
            )
            ## Do not bother to create self.encoder_to_decoder_e
        else:
            self.encoder_to_decoder = torch.nn.Linear(
                dec_in_dim, dec_in_dim, bias=False
            )
            self.encoder_to_decoder_e = torch.nn.Linear(
                dec_in_dim_e, dec_in_dim_e, bias=False
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

    def forward(self, g, x, e):
        # ---- attribute reconstruction ----
        loss = self.mask_attr_prediction(g, x, e)
        loss_item = {"loss": loss.item()}
        return loss, loss_item

    def mask_attr_prediction(self, g, x, e):
        pre_use_g, use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(
            g, x, self._mask_rate
        )

        if self._drop_edge_rate > 0:
            use_g, use_e, masked_edges = drop_edge(
                pre_use_g, self._drop_edge_rate, e, return_edges=True
            )
        else:
            use_g = pre_use_g
            use_e = e

        enc_rep, enc_rep_e, all_hidden = self.encoder(
            use_g, use_x, use_e, return_hidden=True
        )

        if self._concat_hidden:  ## Should be false!
            enc_rep = torch.cat(all_hidden, dim=1)

        # ---- attribute reconstruction ----
        rep = self.encoder_to_decoder(enc_rep)
        rep_e = self.encoder_to_decoder_e(enc_rep_e)

        rep[mask_nodes] = 0

        recon, recon_e = self.decoder(use_g, rep, rep_e)

        x_init = x[mask_nodes]
        x_rec = recon[mask_nodes]

        loss = self.criterion(x_rec, x_init)
        return loss

    def embed(self, g, x, e):
        rep = self.encoder(g, x, e)
        return rep

    @property
    def enc_params(self):
        return self.encoder.parameters()

    @property
    def dec_params(self):
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()])
