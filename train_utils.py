import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

def train_cdan(feature_extractor, classifier, domain_classifier,
               source_loader, target_loader, optimizer, lr_scheduler,
               total_progress, class_weights, trade_off=1.0, eta=0.1, 
               pseudo_weight=0.5, temperature=2.0, consistency_weight=1.0):
    feature_extractor.train()
    classifier.train()
    domain_classifier.train()
    criterion = nn.CrossEntropyLoss(weight=class_weights).cuda()
    criterion_domain = nn.BCEWithLogitsLoss().cuda()
    criterion_consistency = nn.KLDivLoss(reduction='batchmean').cuda()

    initial_conf_threshold = 0.98
    final_conf_threshold = 0.7
    conf_threshold = initial_conf_threshold - (initial_conf_threshold - final_conf_threshold) * total_progress

    total_cls, total_domain, total_ent, total_pseudo, total_cons = 0., 0., 0., 0., 0.
    correct, total = 0, 0
    source_iter = iter(source_loader)
    target_iter = iter(target_loader)
    len_dataloader = min(len(source_loader), len(target_loader))
    progress_bar = tqdm(range(len_dataloader), desc=f"[Epoch {int(total_progress * len_dataloader)}]")

    grl_lambda = 2. / (1. + np.exp(-10 * total_progress)) - 1
    pseudo_counts = torch.zeros(31).cuda()
    max_pseudo_per_class = 10

    for _ in progress_bar:
        try:
            source_data, source_data_weak, source_label = next(source_iter)
            target_data, target_data_weak, _ = next(target_iter)
        except StopIteration:
            source_iter = iter(source_loader)
            target_iter = iter(target_loader)
            source_data, source_data_weak, source_label = next(source_iter)
            target_data, target_data_weak, _ = next(target_iter)

        source_data, source_label = source_data.cuda(), source_label.cuda()
        target_data, target_data_weak = target_data.cuda(), target_data_weak.cuda()

        source_feature = feature_extractor(source_data)
        source_output = classifier(source_feature)
        target_feature = feature_extractor(target_data)
        target_output = classifier(target_feature)
        target_feature_weak = feature_extractor(target_data_weak)
        target_output_weak = classifier(target_feature_weak)

        cls_loss = criterion(source_output, source_label)
        softmax_target = torch.softmax(target_output / temperature, dim=1)
        entropy_loss = -torch.mean(torch.sum(softmax_target * torch.log(softmax_target + 1e-6), dim=1))

        softmax_source = torch.softmax(source_output / temperature, dim=1)
        source_domain_pred = domain_classifier(source_feature, softmax_source, grl_lambda)
        target_domain_pred = domain_classifier(target_feature, softmax_target, grl_lambda)
        source_domain_label = torch.zeros(source_domain_pred.size(0)).float().cuda()
        target_domain_label = torch.ones(target_domain_pred.size(0)).float().cuda()
        domain_loss = criterion_domain(source_domain_pred, source_domain_label) + \
                      criterion_domain(target_domain_pred, target_domain_label)

        pseudo_loss = torch.tensor(0.).cuda()
        if total_progress > 0.1:
            max_probs, pseudo_labels = torch.max(softmax_target, dim=1)
            confident_mask = max_probs > conf_threshold
            if confident_mask.sum() > 0:
                selected_indices = []
                pseudo_counts.zero_()
                for idx in torch.where(confident_mask)[0]:
                    label = pseudo_labels[idx].item()
                    if pseudo_counts[label] < max_pseudo_per_class:
                        selected_indices.append(idx)
                        pseudo_counts[label] += 1
                if selected_indices:
                    selected_indices = torch.tensor(selected_indices, device='cuda')
                    pseudo_loss = criterion(target_output[selected_indices], pseudo_labels[selected_indices])

        consistency_loss = torch.tensor(0.).cuda()
        if total_progress > 0.1:
            max_probs_weak, pseudo_labels_weak = torch.max(torch.softmax(target_output_weak, dim=1), dim=1)
            confident_mask = max_probs_weak > conf_threshold
            if confident_mask.sum() > 0:
                strong_probs = torch.log_softmax(target_output[confident_mask] / temperature, dim=1)
                weak_probs = torch.softmax(target_output_weak[confident_mask] / temperature, dim=1).detach()
                consistency_loss = criterion_consistency(strong_probs, weak_probs)

        loss = cls_loss + trade_off * domain_loss + eta * entropy_loss + \
               pseudo_weight * pseudo_loss + consistency_weight * consistency_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_cls += cls_loss.item()
        total_domain += domain_loss.item()
        total_ent += entropy_loss.item()
        total_pseudo += pseudo_loss.item()
        total_cons += consistency_loss.item()
        _, predicted = torch.max(source_output, 1)
        total += source_label.size(0)
        correct += (predicted == source_label).sum().item()

        progress_bar.set_postfix({
            'cls_loss': total_cls / (progress_bar.n + 1e-6),
            'domain_loss': total_domain / (progress_bar.n + 1e-6),
            'ent_loss': total_ent / (progress_bar.n + 1e-6),
            'pseudo_loss': total_pseudo / (progress_bar.n + 1e-6),
            'cons_loss': total_cons / (progress_bar.n + 1e-6),
            'conf_thresh': conf_threshold
        })

    train_acc = 100 * correct / total
    return train_acc

def test(feature_extractor, classifier, target_loader):
    feature_extractor.eval()
    classifier.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, _, target in target_loader:
            data, target = data.cuda(), target.cuda()
            feature = feature_extractor(data)
            output = classifier(feature)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    acc = 100 * correct / total
    return acc

def extract_features(loader, feature_extractor, use_weak=False, label_offset=0):
    feature_extractor.eval()
    features, labels, domains = [], [], []
    with torch.no_grad():
        for data, data_weak, target in loader:
            data = data_weak if use_weak else data
            data = data.cuda()
            feature = feature_extractor(data)
            features.append(feature.cpu().numpy())
            labels.append(target.numpy())
            domains.append(np.full(len(target), label_offset))
    return np.concatenate(features), np.concatenate(labels), np.concatenate(domains)