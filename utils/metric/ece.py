import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.notebook import tqdm


class ECELoss(nn.Module):
    '''
    Compute ECE (Expected Calibration Error)
    '''

    def __init__(self, n_bins=15):
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels, softmax=True):
        softmaxes = F.softmax(logits, dim=1) if softmax else logits
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece

    def draw_bars(self, logits, labels, softmax=True):
        num_classes = int((torch.max(labels) + 1).item())
        softmaxes = F.softmax(logits, dim=1) if softmax else logits

        accuracy_in_bins = []
        avg_confidence_in_bins = []

        if num_classes > 2:
            confidences, predictions = torch.max(softmaxes, 1)
            accuracies = predictions.eq(labels)

            for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
                in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                accuracy_in_bins.append(accuracy_in_bin if accuracy_in_bin.item() > 0 else 0)
                avg_confidence_in_bins.append(avg_confidence_in_bin if avg_confidence_in_bin.item() > 0 else 0)

            plt.bar(self.bin_lowers.numpy(),
                    accuracy_in_bins,
                    width=(self.bin_uppers - self.bin_lowers).numpy(),
                    label='Actual',
                    color='blue',
                    alpha=0.5)
            plt.bar(self.bin_lowers.numpy(),
                    self.bin_lowers.numpy(),
                    width=(self.bin_uppers - self.bin_lowers).numpy(),
                    label='Expected',
                    color='pink',
                    alpha=0.85)
            plt.legend(['Actual', 'Expected'])
        elif num_classes == 2:
            confidences, predictions = softmaxes[:, 1], torch.argmax(softmaxes, 1)
            accuracies = predictions.eq(labels)

            for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
                in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                accuracy_in_bins.append(accuracy_in_bin if accuracy_in_bin.item() > 0 else 0)
                avg_confidence_in_bins.append(avg_confidence_in_bin if avg_confidence_in_bin.item() > 0 else 0)

            plt.bar(self.bin_lowers.numpy(),
                    accuracy_in_bins,
                    width=(self.bin_uppers - self.bin_lowers).numpy(),
                    label='Actual',
                    color='blue',
                    alpha=0.45)
            expected_accuracy = torch.cat([1 - self.bin_lowers[: len(self.bin_lowers) // 2], self.bin_uppers[len(self.bin_uppers) // 2:]])
            plt.bar(self.bin_lowers.numpy(),
                    expected_accuracy.numpy(),
                    width=(self.bin_uppers - self.bin_lowers).numpy(),
                    label='Expected',
                    color='pink',
                    alpha=0.85)
            plt.legend(['Actual', 'Expected'], loc='upper center', ncol=2)
            plt.xlim(-0.05, 1.05)
            plt.ylim(0, 1.15)
        else:
            raise ValueError('Number of classes must be greater than 1')
        return plt


class ClassWiseECELoss(nn.Module):
    '''
    Compute Classwise ECE
    '''

    def __init__(self, n_bins=15):
        super(ClassWiseECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels, softmax=True):
        num_classes = int((torch.max(labels) + 1).item())
        softmaxes = F.softmax(logits, dim=1) if softmax else logits
        per_class_sce = None

        for i in range(num_classes):
            class_confidences = softmaxes[:, i]
            class_sce = torch.zeros(1, device=logits.device)
            labels_in_class = labels.eq(i)  # one-hot vector of all positions where the label belongs to the class i

            for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
                in_bin = class_confidences.gt(bin_lower.item()) * class_confidences.le(bin_upper.item())
                prop_in_bin = in_bin.float().mean()
                if prop_in_bin.item() > 0:
                    accuracy_in_bin = labels_in_class[in_bin].float().mean()
                    avg_confidence_in_bin = class_confidences[in_bin].mean()
                    class_sce += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

            if (i == 0):
                per_class_sce = class_sce
            else:
                per_class_sce = torch.cat((per_class_sce, class_sce), dim=0)

        sce = torch.mean(per_class_sce)  # if num_classes = 2, per_class_sce for class 0 and 1 are the same (has been empirically proved)
        return sce, per_class_sce


class AdaptiveECELoss(nn.Module):
    '''
    Compute Adaptive ECE
    '''

    def __init__(self, n_bins=15):
        super(AdaptiveECELoss, self).__init__()
        self.nbins = n_bins

    def histedges_equalN(self, x):
        npt = len(x)
        return np.interp(np.linspace(0, npt, self.nbins + 1),
                         np.arange(npt),
                         np.sort(x))

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)
        n, bin_boundaries = np.histogram(confidences.cpu().detach(), self.histedges_equalN(confidences.cpu().detach()))
        # print(n,confidences,bin_boundaries)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece


class ModelWithTemperature(nn.Module):
    def __init__(self, model, log=True):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = 1.0
        self.log = log

    def forward(self, input, key='logits'):
        logits = self.model(input)[key]
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        return logits / self.temperature

    def set_temperature(self, valid_loader, cross_validate='ece', key='logits'):
        self.cuda()
        self.model.eval()
        nll_criterion = nn.CrossEntropyLoss().cuda()
        ece_criterion = ECELoss().cuda()

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for batch in valid_loader:
                seqs, seq_tokens, seqs_group, mutants, mutant_tokens, mutants_group, labels = batch
                input = seq_tokens.cuda()
                logits = self.model(input)[key]
                logits_list.append(logits)
                labels_list.append(labels)
            logits = torch.cat(logits_list).cuda()
            labels = torch.cat(labels_list).cuda()

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        if self.log:
            print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

        nll_val = 1e9
        ece_val = 1e9
        T_opt_nll = 1.0
        T_opt_ece = 1.0
        T = 0.01
        for i in tqdm(range(1000)):
            self.temperature = T
            self.cuda()
            after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
            after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
            if nll_val > after_temperature_nll:
                T_opt_nll = T
                nll_val = after_temperature_nll

            if ece_val > after_temperature_ece:
                T_opt_ece = T
                ece_val = after_temperature_ece
            T += 0.01

        if cross_validate == 'ece':
            self.temperature = T_opt_ece
        else:
            self.temperature = T_opt_nll
        self.cuda()

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
        if self.log:
            print('Optimal temperature: %.3f' % self.temperature)
            print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))

    def get_temperature(self):
        return self.temperature
