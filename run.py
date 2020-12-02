import os
import argparse
import random

import torch
import torch.optim as optim

from utils import dataset
import model.backbone as backbone

import metric.loss as loss
import metric.pairsampler as pair

from tqdm import tqdm
from torch.utils.data import DataLoader

from metric.batchsampler import MImagesPerClassSampler
from model.embedding import L2NormEmbedding, Embedding
from utils.common import fix_batchnorm, build_transform, recall


def train(
    net, loader, optimizer, criterion, reg_criterion, lambda_mdr=0, nu_mdr=0, ep=0
):
    net.train()
    fix_batchnorm(net)

    train_iter = tqdm(loader, ncols=80)
    loss_all = []
    for images, labels in train_iter:
        images, labels = images.cuda(), labels.cuda()
        embedding = net(images)

        loss_minibatch = criterion(embedding, labels)
        loss_mdr = (
            lambda_mdr * reg_criterion(embedding)
            + nu_mdr * reg_criterion.levels.pow(2).sum()
        )

        optimizer.zero_grad()
        (loss_minibatch + loss_mdr).backward()
        optimizer.step()
        train_iter.set_description(
            "[Train][Epoch %d] Loss: %.5f Reg: %.5f"
            % (ep, loss_minibatch.item(), loss_mdr.item())
        )

        loss_all.append(loss_minibatch.item())

    print("[Epoch %d] Loss: %.5f\n" % (ep, torch.Tensor(loss_all).mean()))


def eval_dml(net, loader, K=[1], ep=0):
    net.eval()
    test_iter = tqdm(loader, ncols=80)
    embeddings_all, labels_all = [], []

    test_iter.set_description("[Eval][Epoch %d]" % ep)
    with torch.no_grad():
        for images, labels in test_iter:
            images, labels = images.cuda(), labels.cuda()
            embedding = net(images)
            embeddings_all.append(embedding.data)
            labels_all.append(labels.data)

        embeddings_all = torch.cat(embeddings_all)
        labels_all = torch.cat(labels_all)
        rec = recall(embeddings_all, labels_all, K=K)

        for k, r in zip(K, rec):
            print("[Epoch %d] Recall@%d: [%.4f]\n" % (ep, k, r))

    return rec[0], K, rec


def build_args():
    parser = argparse.ArgumentParser()
    LookupChoices = type(
        "",
        (argparse.Action,),
        dict(__call__=lambda a, p, n, v, o: setattr(n, a.dest, a.choices[v])),
    )

    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--load", default=None)
    parser.add_argument(
        "--dataset",
        choices=dict(
            cub200=dataset.CUB2011Metric,
            cars196=dataset.Cars196Metric,
            stanford=dataset.StanfordOnlineProductsMetric,
        ),
        default=dataset.CUB2011Metric,
        action=LookupChoices,
    )

    parser.add_argument(
        "--backbone",
        choices=dict(
            bninception=backbone.BNInception,
            resnet18=backbone.ResNet18,
            resnet50=backbone.ResNet50,
        ),
        default=backbone.BNInception,
        action=LookupChoices,
    )

    parser.add_argument("--l2norm", default=False, action="store_true")
    parser.add_argument("--lambda-mdr", type=float, default=0.0)
    parser.add_argument("--nu-mdr", type=float, default=0.0)

    parser.add_argument("--embedding-size", type=int, default=512)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--lr-decay-epochs", type=int, default=[40, 60, 80], nargs="+")
    parser.add_argument("--lr-decay-gamma", default=0.2, type=float)
    parser.add_argument("--batch", default=128, type=int)
    parser.add_argument("--num-image-per-class", default=4, type=int)
    parser.add_argument("--iter-per-epoch", default=100, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--recall", default=[1], type=int, nargs="+")

    parser.add_argument("--seed", default=random.randint(1, 1000), type=int)
    parser.add_argument("--data", default="./dataset/")
    parser.add_argument("--save-dir", default="./result")
    opts = parser.parse_args()

    return opts


if __name__ == "__main__":
    opts = build_args()

    for set_random_seed in [random.seed, torch.manual_seed, torch.cuda.manual_seed_all]:
        set_random_seed(opts.seed)

    base_model = opts.backbone(pretrained=True)

    if opts.l2norm:
        model = L2NormEmbedding(
            base_model,
            feature_size=base_model.output_size,
            embedding_size=opts.embedding_size,
        ).cuda()
    else:
        model = Embedding(
            base_model,
            feature_size=base_model.output_size,
            embedding_size=opts.embedding_size,
        ).cuda()

    if opts.load is not None:
        model.load_state_dict(torch.load(opts.load))
        print("Loaded Model from %s" % opts.load)

    train_transform, test_transform = build_transform(base_model)
    dataset_train = opts.dataset(
        opts.data, train=True, transform=train_transform, download=True
    )
    dataset_train_eval = opts.dataset(
        opts.data, train=True, transform=test_transform, download=True
    )
    dataset_eval = opts.dataset(
        opts.data, train=False, transform=test_transform, download=True
    )

    print("Number of images in Training Set: %d" % len(dataset_train))
    print("Number of images in Test set: %d" % len(dataset_eval))

    loader_train = DataLoader(
        dataset_train,
        batch_sampler=MImagesPerClassSampler(
            dataset_train, opts.batch, m=opts.num_image_per_class, iter_per_epoch=opts.iter_per_epoch
        ),
        pin_memory=True,
        num_workers=8,
    )
    loader_eval = DataLoader(
        dataset_eval,
        shuffle=False,
        batch_size=32,
        drop_last=False,
        pin_memory=True,
        num_workers=8,
    )

    criterion = loss.TripletLoss(sampler=pair.DistanceWeighted(), margin=0.2).cuda()
    reg_criterion = loss.MDRLoss().cuda()

    optimizer = optim.Adam(
        [
            {"lr": opts.lr, "params": model.parameters()},
            {"lr": opts.lr, "params": criterion.parameters()},
            {"lr": opts.lr, "params": reg_criterion.parameters()},
        ],
        weight_decay=1e-5,
    )

    lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=opts.lr_decay_epochs, gamma=opts.lr_decay_gamma
    )

    val_recall, _, _ = eval_dml(model, loader_eval, opts.recall, 0)
    best_rec = val_recall

    if opts.mode == "eval":
        exit(0)

    for epoch in range(1, opts.epochs + 1):
        train(
            model,
            loader_train,
            optimizer,
            criterion,
            reg_criterion,
            lambda_mdr=opts.lambda_mdr,
            nu_mdr=opts.nu_mdr,
            ep=epoch,
        )
        lr_scheduler.step()

        val_recall, val_recall_K, val_recall_all = eval_dml(
            model, loader_eval, opts.recall, epoch
        )

        if best_rec < val_recall:
            best_rec = val_recall
            if opts.save_dir is not None:
                if not os.path.isdir(opts.save_dir):
                    os.mkdir(opts.save_dir)
                torch.save(model.state_dict(), "%s/%s" % (opts.save_dir, "best.pth"))

        if opts.save_dir is not None:
            if not os.path.isdir(opts.save_dir):
                os.mkdir(opts.save_dir)
            torch.save(model.state_dict(), "%s/%s" % (opts.save_dir, "last.pth"))
            with open("%s/result.txt" % opts.save_dir, "w") as f:
                f.write("Best Recall@1: %.4f\n" % best_rec)
                f.write("Final Recall@1: %.4f\n" % val_recall)

        print("Best Recall@1: %.4f" % best_rec)
