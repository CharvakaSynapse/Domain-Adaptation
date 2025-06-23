import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import json
import numpy as np
import random
from dataset import Office31Dataset, get_data_loaders
from models import Classifier, DomainClassifier
from train_utils import train_cdan, test, extract_features
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def get_transforms():
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
        transforms.RandomGrayscale(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    weak_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return train_transform, weak_transform

def main():
    config = load_config('config.json')
    set_seed(config['seed'])

    train_transform, weak_transform = get_transforms()
    source_loader, target_loader, class_weights = get_data_loaders(
        config['data_root'], config['source_domain'], config['target_domain'],
        config['batch_size'], config['num_classes'], train_transform, weak_transform
    )

    feature_extractor = torchvision.models.resnet50(weights="IMAGENET1K_V1")
    feature_extractor.fc = torch.nn.Identity()
    classifier = Classifier(num_classes=config['num_classes'])
    domain_classifier = DomainClassifier(
        num_classes=config['num_classes'], random=config['random']
    )
    feature_extractor.cuda()
    classifier.cuda()
    domain_classifier.cuda()

    params = [
        {'params': feature_extractor.parameters(), 'lr': 0.001},
        {'params': classifier.parameters(), 'lr': 0.01},
        {'params': domain_classifier.parameters(), 'lr': 0.01}
    ]
    optimizer = optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=int(0.8 * config['num_epochs']), gamma=0.1
    )

    best_acc = 0.0
    current_iteration = 0
    len_dataloader = min(len(source_loader), len(target_loader))
    total_iterations = config['num_epochs'] * len_dataloader

    for epoch in range(config['num_epochs']):
        train_acc = train_cdan(
            feature_extractor, classifier, domain_classifier,
            source_loader, target_loader, optimizer, scheduler,
            total_progress=current_iteration / total_iterations,
            class_weights=class_weights,
            trade_off=config['trade_off'],
            eta=config['eta'],
            pseudo_weight=config['pseudo_weight'],
            temperature=config['temperature'],
            consistency_weight=config['consistency_weight']
        )
        test_acc = test(feature_extractor, classifier, target_loader)
        print(f'[Epoch {epoch}] Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | Best Test Acc: {best_acc:.2f}%')
        best_acc = max(best_acc, test_acc)
        current_iteration += len_dataloader
        scheduler.step()

    print("Extracting features...")
    source_features, source_labels, source_domains = extract_features(
        source_loader, feature_extractor, use_weak=False, label_offset=0
    )
    target_features, target_labels, target_domains = extract_features(
        target_loader, feature_extractor, use_weak=False, label_offset=100
    )
    all_features = np.concatenate([source_features, target_features])
    all_domains = np.concatenate([source_domains, target_domains])

    print("Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    features_2d = tsne.fit_transform(all_features)

    plt.figure(figsize=(8,6))
    plt.scatter(features_2d[all_domains==0, 0], features_2d[all_domains==0, 1], 
                label="Source", alpha=0.5, s=20)
    plt.scatter(features_2d[all_domains==100, 0], features_2d[all_domains==100, 1], 
                label="Target", alpha=0.5, s=20)
    plt.legend()
    plt.title("t-SNE of Features (Source vs. Target)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()