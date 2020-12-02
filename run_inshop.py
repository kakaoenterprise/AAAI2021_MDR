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


def eval_inshop(net, loader_query, loader_gallery, K=[1], ep=0):
    net.eval()
    query_iter = tqdm(loader_query, ncols=80)
    gallery_iter = tqdm(loader_gallery, ncols=80)

    query_embeddings_all, query_labels_all = [], []
    gallery_embeddings_all, gallery_labels_all = [], []

    with torch.no_grad():
        for images, labels in query_iter:
            images, labels = images.cuda(), labels.cuda()
            embedding = net(images)
            query_embeddings_all.append(embedding.data)
            query_labels_all.append(labels.data)

        query_embeddings_all = torch.cat(query_embeddings_all)
        query_labels_all = torch.cat(query_labels_all)

        for images, labels in gallery_iter:
            images, labels = images.cuda(), labels.cuda()
            embedding = net(images)
            gallery_embeddings_all.append(embedding.data)

        gallery_embeddings_all = torch.cat(gallery_embeddings_all)

    correct_labels = []
    for query_e, query_l in zip(query_embeddings_all, query_labels_all):
        distance = (gallery_embeddings_all - query_e[None]).pow(2).sum(dim=1)
        knn_ind = distance.topk(max(K), dim=0, largest=False, sorted=True)[1]

        query_label_text = loader_query.dataset.data_labels[query_l.item()]
        gallery_label_text = [loader_gallery.dataset.labels[k.item()] for k in knn_ind]

        cl = [query_label_text == g for g in gallery_label_text]
        correct_labels.append(cl)

    correct_labels = torch.FloatTensor(correct_labels)

    recall_k = []
    for k in K:
        correct_k = 100 * (correct_labels[:, :k].sum(dim=1) > 0).float().mean().item()
        recall_k.append(correct_k)
        print("[Epoch %d] Recall@%d: [%.4f]\n" % (ep, k, correct_k))

    return recall_k[0], K, recall_k


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

    dataset_train = dataset.FashionInshop(
        opts.data, split="train", transform=train_transform
    )
    dataset_query = dataset.FashionInshop(
        opts.data, split="query", transform=test_transform
    )
    dataset_gallery = dataset.FashionInshop(
        opts.data, split="gallery", transform=test_transform
    )

    print("Number of images in Training Set: %d" % len(dataset_train))
    print("Number of images in Query set: %d" % len(dataset_query))
    print("Number of images in Gallery set: %d" % len(dataset_gallery))

    loader_train = DataLoader(
        dataset_train,
        batch_sampler=MImagesPerClassSampler(
            dataset_train, opts.batch, m=opts.num_image_per_class, iter_per_epoch=opts.iter_per_epoch
        ),
        pin_memory=True,
        num_workers=8,
    )
    loader_query = DataLoader(
        dataset_query,
        shuffle=False,
        batch_size=32,
        drop_last=False,
        pin_memory=False,
        num_workers=8,
    )
    loader_gallery = DataLoader(
        dataset_gallery,
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

    val_recall, _, _ = eval_inshop(model, loader_query, loader_gallery, opts.recall, 0)
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

        val_recall, val_recall_K, val_recall_all = eval_inshop(
            model, loader_query, loader_gallery, opts.recall, epoch
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
