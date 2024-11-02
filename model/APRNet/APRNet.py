import numpy as np
import pandas as pd

from model.APNet.data_module import DataModule
from model.APNet.lightning_module import LightningModule
from utils.bio.struct_utils import compute_sasa
from utils.bio.struct_utils import get_atom_array, compute_distance_matrix, compute_adjacency_matrix
from utils.lightning import LitModelInference, merge_batch_prediction
from utils.metric.sov import Sov


class APRNet():
    def __init__(self, APNet, pep_len=6, log=False):
        super(APRNet, self).__init__()
        if isinstance(APNet, str):
            self.APNet = LitModelInference(LightningModule, DataModule, APNet)
        else:
            self.APNet = APNet

        self.APNet.set_batch_size(batch_size=1024, num_workers=1)
        self.pep_len = pep_len
        self.log = log

        self.structure_info = {}
        self.score_mapping = None

        self.default_params = {
            'beta': 3.36,
            'delta': 0.4,
            't_start': 0.51,
            't_expand': 0.37,
            't_patience': 9,
            'position_weights': None,
            'radius': 8
        }

    def __call__(self, prot_seqs, prot_pdbs=None, cache=True, **kwargs):
        self.prot_seqs = prot_seqs
        self.prot_pdbs = prot_pdbs

        if self.log:
            print('[step 1] Construct Sliding Window ...')
        all_k_peps, window_map_dict = self.split_protein(prot_seqs)

        if prot_pdbs is not None:
            if self.log:
                print('[step 1] Extract Structural Information ...')
            if self.structure_info is not None and len(self.structure_info) > 0 and cache:
                structure_info = self.structure_info
            else:
                structure_info = self.extract_structural_info(prot_pdbs, **kwargs)
                self.structure_info = structure_info
        else:
            structure_info = {}

        if self.score_mapping is not None and cache:
            score_mapping = self.score_mapping
        else:
            if self.log:
                print('[step 2] Predict Peptide Scores ...')
            pep_scores = self.score_peptide(all_k_peps)
            if self.log:
                print('[step 3] Map Peptide Scores ...')
            score_mapping = self.map_pep_score(prot_seqs, all_k_peps, window_map_dict, pep_scores)
            self.score_mapping = score_mapping

        if self.log:
            print('[step 4] Score Protein ...')
        protein_scores = self.score_sequence(prot_seqs, score_mapping, **structure_info, **kwargs)

        if self.log:
            print('[step 5] Span APR ...')
        pred_labels = self.predict_label(protein_scores, **kwargs)
        return pred_labels, protein_scores

    def split_protein(self, prot_seqs):
        all_k_peps = []
        window_map_dict = {}
        k = self.pep_len
        for i in range(len(prot_seqs)):
            for j in range(len(prot_seqs[i])):
                windows = [prot_seqs[i][t:t + k] for t in range(j - k + 1, j + 1) if 0 <= t < len(prot_seqs[i]) - k + 1]
                window_map_dict[f'seq[{i}]_window[{j}]'] = windows
                all_k_peps.extend(windows)

        # window_map_dict: key="seq[i]_window[j]", value=[pep_1, ..., pep_6]
        # all_k_peps: all hexapeptides in prot_seqs
        all_k_peps = np.array(list(set(all_k_peps)))
        window_map_dict = window_map_dict
        return all_k_peps, window_map_dict

    def extract_structural_info(self, prot_pdbs, **kwargs):
        adjacency_list = []
        distance_list = []
        sasa_list = []
        for idx in range(len(prot_pdbs)):
            radius = kwargs.get('radius', self.default_params['radius'])
            structure = get_atom_array(prot_pdbs[idx])
            adjacency = compute_adjacency_matrix(structure, threshold=radius).astype(np.float32)
            distance = compute_distance_matrix(structure)
            sasa = compute_sasa(structure, level='residue', mode='RSA')
            adjacency_list.append(adjacency)
            distance_list.append(distance)
            sasa_list.append(sasa)

        structure_info = {'adjacency': adjacency_list, 'distance': distance_list, 'sasa': sasa_list}
        return structure_info

    def score_peptide(self, peptides):
        predictions = self.APNet.predict(peptides)
        results = merge_batch_prediction(predictions)
        logits = results['preds'].cpu()
        pep_scores = logits.softmax(dim=-1)[:, -1].numpy()  # (B, 2) -> (B, 1)
        return pep_scores

    def map_pep_score(self, prot_seqs, all_k_peps, window_map_dict, pep_scores):
        score_mapping = {}
        for i in range(len(prot_seqs)):
            for j in range(len(prot_seqs[i])):
                window = window_map_dict[f'seq[{i}]_window[{j}]']
                pep_idx = [np.where(all_k_peps == pep)[0] for pep in window]
                window_scores = pep_scores[pep_idx].reshape(-1)
                if len(window_scores) == 0:
                    raise RuntimeError('len(window_scores) == 0')
                if len(window_scores) != self.pep_len:
                    zero_pad = np.zeros(self.pep_len - len(window_scores))
                    window_scores = [zero_pad, window_scores] if j - self.pep_len < 0 else [window_scores, zero_pad]
                    window_scores = np.concatenate(window_scores, axis=0)
                score_mapping[f'seq[{i}]_window[{j}]'] = window_scores
        return score_mapping

    def score_sequence(self, prot_seqs, score_mapping, **kwargs):
        sasa_list = kwargs.get('sasa', None)
        distance_list = kwargs.get('distance', None)
        adjacency_list = kwargs.get('adjacency', None)
        beta = kwargs.get('beta', self.default_params['beta'])
        delta = kwargs.get('delta', self.default_params['delta'])

        protein_scores = []
        for i in range(len(prot_seqs)):
            agg_score = np.zeros(len(prot_seqs[i]), dtype=float)
            for j in range(len(prot_seqs[i])):
                window_scores = score_mapping[f'seq[{i}]_window[{j}]']
                agg_score[j] = self.score_function(window_scores, **kwargs)  # s_agg

            if all([sasa_list, distance_list, adjacency_list]):
                sasa = sasa_list[i]
                distance = distance_list[i]
                adjacency = adjacency_list[i]
                sasa_score = agg_score * np.exp(beta * (sasa - np.mean(sasa)))
                weight = np.exp(-delta * distance) * adjacency  # w = exp(-deleta * d) * a
                weight_score = weight @ sasa_score  # s_weight = w * s_sasa
                weight_score = (weight_score - np.min(weight_score)) / (np.max(weight_score) - np.min(weight_score))
                agg_score = weight_score

            # for N-terminal and C-terminal residues, the score could not be too high because of the lack of context
            if kwargs.get('reduce_NC_terminals', True):
                agg_score[:3] = agg_score[:3] * 0.5
                agg_score[-3:] = agg_score[-3:] * 0.5
            protein_scores.append(agg_score)
        return protein_scores

    def score_function(self, window_scores, **kwargs):
        position_weights = kwargs.get('position_weights', self.default_params['position_weights'])
        if position_weights is not None:
            position_weights = np.array(position_weights)
        score = np.average(window_scores, weights=position_weights)
        return score

    def predict_label(self, protein_scores, **kwargs):
        return [self.span_APR(protein_scores[i], **kwargs) for i in range(len(protein_scores))]

    def span_APR(self, protein_scores, **kwargs):
        t_start = kwargs.get('t_start', self.default_params['t_start'])
        t_expand = kwargs.get('t_expand', self.default_params['t_expand'])
        t_patience = kwargs.get('t_patience', self.default_params['t_patience'])
        l = len(protein_scores)
        pred_label = np.zeros(l, dtype=int)

        # expand APR according to peaks
        for j, j_score in enumerate(protein_scores):
            if j_score >= t_start:
                # forward expand
                e = j
                patience = t_patience
                while 0 <= e < l:
                    if protein_scores[e] >= t_expand:
                        pred_label[e] = 1
                    else:
                        pred_label[e] = 1
                        patience -= 1
                        if patience == 0:
                            break
                    e += 1  # move forward

                # backward expand
                e = j
                patience = t_patience
                while 0 <= e < l:
                    if protein_scores[e] >= t_expand:
                        pred_label[e] = 1
                    else:
                        pred_label[e] = 1
                        patience -= 1
                        if patience == 0:
                            break
                    e -= 1  # move backward

        pred_label[:3] = 0
        pred_label[-3:] = 0
        return pred_label

    def compute_sov(self, pred_labels, true_labels):
        Sov_Overall_list = []
        Sov_APR_list = []
        Sov_non_APR_list = []
        for i in range(len(pred_labels)):
            assert len(pred_labels[i]) == len(true_labels[i])
            prot_label = true_labels[i][:len(true_labels[i])]
            pred_label = pred_labels[i][:len(pred_labels[i])]
            Sov_Overall, Sov_APR, Sov_non_APR = Sov(prot_label, pred_label)
            Sov_Overall_list.append(Sov_Overall)
            Sov_APR_list.append(Sov_APR)
            Sov_non_APR_list.append(Sov_non_APR)

        Sov_Overall = np.average(Sov_Overall_list)
        Sov_APR = np.average(Sov_APR_list)
        Sov_non_APR = np.average(Sov_non_APR_list)
        Sov_average = (Sov_APR + Sov_non_APR) / 2
        metrics = [Sov_APR, Sov_non_APR, Sov_Overall, Sov_average]
        metrics = metrics + [metrics[-1] + metrics[-2], 0.7 * metrics[0] + 0.3 * metrics[1]]
        index = ['Sov APR', 'Sov non-APR', 'Sov Overall', 'Sov Average', 'Total Score', 'Weighted Average']
        return pd.DataFrame(metrics, index=index).T
