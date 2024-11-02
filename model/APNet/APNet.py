import esm
import torch
import torch.nn as nn

from utils.loss import get_loss
from utils.model.Layer import MLP, get_hiddens


class APNet(nn.Module):
    def __init__(self, embed_dim, task_heads, down_mlp=None, feature_mlp=None, esm_model_name='esm2_t33_650M_UR50D',
                 esm_freeze=True, return_contacts=False, **kwargs):
        super(APNet, self).__init__()

        # esm layer
        self.repr_layer = 33
        self.esm_model_name = esm_model_name
        self.esm_freeze = esm_freeze
        self.return_contacts = return_contacts
        self.__init_esm_model__()

        self.embed_dim = embed_dim
        self.down_mlp = down_mlp
        self.feature_mlp = feature_mlp
        self.task_heads = task_heads
        self.__init_submodules__(**kwargs)

    def __init_esm_model__(self):
        # print('ESM model initializing...')
        if self.esm_model_name is None:
            self.esm_model = None
            self.alphabet = None
        else:
            self.esm_model, self.alphabet = esm.pretrained.load_model_and_alphabet(self.esm_model_name)
            if self.esm_freeze:
                for param in self.esm_model.parameters():
                    param.requires_grad = False
            else:
                self.esm_model.train()

    def __init_submodules__(self, **kwargs):
        self.esm_dim = self.esm_model.embed_dim if self.esm_model else 1280

        if self.down_mlp is not None:
            self.down_mlp['hiddens'] = get_hiddens(self.down_mlp['hiddens'],
                                                   input_dim=self.esm_dim,
                                                   output_dim=self.embed_dim)  # (B, 1280) -> (B, E)
            self.down_mlp = MLP(**self.down_mlp)  # (B, 1280) -> (B, E)
        else:
            self.down_mlp = None

        if self.feature_mlp is not None:
            self.feature_mlp['hiddens'] = get_hiddens(self.feature_mlp['hiddens'])
            self.feature_mlp = MLP(**self.feature_mlp)  # (B, F) -> (B, E)
        else:
            self.feature_mlp = None

        for key, value in self.task_heads.items():
            value['hiddens'] = get_hiddens(value['hiddens'])
            self.add_module(key, MLP(**value))  # (B, E) -> (B, c)

        self.task_heads = {key: self._modules[key] for key in self.task_heads.keys()}
        self.ce_loss = get_loss({'name': 'CrossEntropy', 'args': {}})

    def extract_esm_embeddings(self, batch_tokens, add_special_tokens=False):
        if self.esm_model:
            if add_special_tokens:  # add cls and eos tokens, (B, T) -> (B, T+2)
                cls_idx = self.esm_model.cls_idx
                eos_idx = self.esm_model.eos_idx
                cls_pads = cls_idx * torch.ones_like(batch_tokens[:, :1])  # (B, 1)
                eos_pads = eos_idx * torch.ones_like(batch_tokens[:, :1])  # (B, 1)
                x = torch.cat([cls_pads, batch_tokens, eos_pads], dim=-1)  # (B, T+2)
            else:
                x = batch_tokens  # (B, T+2)

            # compute embeddings and contact map
            if self.esm_freeze:
                with torch.no_grad():
                    results = self.esm_model(x, repr_layers=[self.repr_layer],
                                             return_contacts=self.return_contacts)
            else:
                results = self.esm_model(x, repr_layers=[self.repr_layer],
                                         return_contacts=self.return_contacts)

            x = results['representations'][self.repr_layer][:, 1:-1, :]  # (B, T+2, E) -> (B, T, E)
        else:
            x = batch_tokens  # (B, T, E)
        return x

    def feature_fusion(self, embeddings, features=None):
        if self.down_mlp is not None:
            embeddings = self.down_mlp(embeddings)  # (B, T, 1280) -> (B, T, E)
            embeddings = embeddings.mean(dim=1)  # (B, T, E) -> (B, E)

        if self.feature_mlp is not None and features is not None:
            features = self.feature_mlp(features)  # (B, F) -> (B, E)
            features = embeddings * features  # g(x, y) = x * y
        else:
            features = 0

        # feature fusion. f(x, y) = x + g(x, y) like residual connection
        embeddings = embeddings + features
        return embeddings

    def predict_labels(self, z):
        # predict activity and stability for regression and classification tasks
        results = {'latent': z}
        for key, task_head in self.task_heads.items():
            results[key] = task_head(z)
        return results

    def forward(self, tokens, features=None, add_special_tokens=False):
        embeddings = self.extract_esm_embeddings(tokens, add_special_tokens)  # (B, T, E)
        z = self.feature_fusion(embeddings, features)  # (B, F)
        results = self.predict_labels(z)  # {'cls_head': (B, c)}
        return results

    def compute_loss(self, preds, labels):
        # preds: (B, c), labels: (B)
        loss = self.ce_loss(preds, labels)
        return loss
