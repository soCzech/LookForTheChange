import os
import sys
import torch
import argparse
from datetime import datetime
from torch.utils.data import DataLoader

from method import dataset, model, utils


def main(args):
    if args.checkpoint_dir is not None:
        args.checkpoint_dir = os.path.join(args.checkpoint_dir,
                                           datetime.strftime(datetime.now(), "%Y%m%d-%H%M%S-") + args.category)
        os.makedirs(args.checkpoint_dir)
        txt_file = open(os.path.join(args.checkpoint_dir, "training.log"), "w")
        print("Logging into", os.path.join(args.checkpoint_dir, "training.log"), flush=True)
    else:
        txt_file = sys.stdout

    train_ds = dataset.ChangeItDataset(
        pickle_roots=args.pickle_roots, single_class=args.category, annotation_root=args.annotation_root,
        file_mode="unannotated", noise_adapt_weight_root=args.noise_adapt_weight_root,
        noise_adapt_weight_threshold_file=args.noise_adapt_weight_threshold_file, deterministic=args.ds_no_augment)
    test_ds = dataset.ChangeItDataset(
        pickle_roots=args.pickle_roots, single_class=args.category, annotation_root=args.annotation_root,
        file_mode="annotated", deterministic=True)
    print(train_ds, test_ds, sep="\n", file=txt_file, flush=True)

    train_ds_iter = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=6, pin_memory=False, collate_fn=dataset.changeit_collate_fn)
    test_ds_iter = DataLoader(
        test_ds, batch_size=1, shuffle=True, drop_last=False,
        num_workers=6, pin_memory=False, collate_fn=dataset.changeit_collate_fn)

    # MODEL
    network = model.MultiClassMLP(layers=[2048 + 2048, 512], n_classes=1).cuda()
    optimizer = torch.optim.SGD(network.parameters(), args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = utils.get_cosine_schedule_with_warmup(optimizer, 10 * len(train_ds_iter),
                                                      len(train_ds_iter) * args.n_epochs)
    loss_function = model.LookForTheChangeLoss(args.delta, args.kappa)

    loss_metric = utils.AverageMeter()
    loss_norm_metric = utils.AverageMeter()
    unsup_state_loss_metric = utils.AverageMeter()
    unsup_action_loss_metric = utils.AverageMeter()

    for epoch in range(1, args.n_epochs + 1):
        network.train()
        loss_metric.reset()
        loss_norm_metric.reset()
        unsup_state_loss_metric.reset()
        unsup_action_loss_metric.reset()

        for i, batch in enumerate(train_ds_iter):
            optimizer.zero_grad()

            features = batch["features"].cuda()
            log_pred = network(features)

            log_pred_state = utils.select_correct_classes(log_pred["state"], batch["classes"])
            log_pred_action = utils.select_correct_classes(log_pred["action"], batch["classes"])

            video_loss_weight = None
            if args.video_level_temp is not None:
                assert batch["video_level_scores"] is not None
                video_loss_weight = batch["video_level_scores"] * (-1 / args.video_level_temp)
                video_loss_weight = 1 / (1 + torch.exp(video_loss_weight))
                video_loss_weight = video_loss_weight.cuda()

            unsup_state_loss, unsup_action_loss = \
                loss_function(log_pred_state, log_pred_action, batch["lens"].cuda(), video_loss_weight)
            loss = unsup_state_loss + unsup_action_loss

            loss.backward()

            total_norm = torch.norm(
                torch.stack([torch.norm(p.grad.detach(), 2.0) for p in network.parameters()]), 2.0)
            loss_norm_metric.update(total_norm.item(), len(features))
            torch.nn.utils.clip_grad_norm_(network.parameters(), args.grad_clipping)

            optimizer.step()
            scheduler.step()

            loss_metric.update(loss.item(), len(features))
            unsup_state_loss_metric.update(unsup_state_loss.item(), len(features))
            unsup_action_loss_metric.update(unsup_action_loss.item(), len(features))

        if epoch % args.eval_each != 0:
            continue

        network.eval()
        joint_meter = utils.JointMeter(train_ds.n_classes)

        for i, batch in enumerate(test_ds_iter):
            features = batch["features"].cuda()

            with torch.no_grad():
                log_pred = network(features)

            log_pred_state = utils.select_correct_classes(log_pred["state"], batch["classes"])
            log_pred_action = utils.select_correct_classes(log_pred["action"], batch["classes"])
            test_pred_state = torch.softmax(log_pred_state, -1).cpu().numpy()[0]
            test_pred_action = torch.sigmoid(log_pred_action).cpu().numpy()[0, :, 0]

            joint_meter.log(test_pred_action, test_pred_state, batch["annotations"][0],
                            category=batch["classes"][0].item())

        print(f"Epoch {epoch}/{args.n_epochs} ("
              f"T loss: {loss_metric.value:.3f}, "
              f"T lr: {scheduler.get_last_lr()[0]:.6f}, "
              f"T grad norm: {loss_norm_metric.value:.1f}, "
              f"T unsup state loss: {unsup_state_loss_metric.value:.3f}, "
              f"T unsup action loss: {unsup_action_loss_metric.value:.3f}, "
              f"V state acc: {joint_meter.acc:.1f}%, "
              f"V state prec: {joint_meter.sp:.1f}%, "
              f"V state joint prec: {joint_meter.jsp:.1f}%, "
              f"V action prec: {joint_meter.ap:.1f}%, "
              f"V action joint prec: {joint_meter.jap:.1f}%)", file=txt_file, flush=True)

        print("> {:20} {:>6} {:>6} {:>6} {:>6} {:>6}".format("CATEGORY", "SAcc", "SP", "JtSP", "AP", "JtAP"), file=txt_file)
        print("\n".join([
            "> {:20}{:6.1f}%{:6.1f}%{:6.1f}%{:6.1f}%{:6.1f}%".format(cls_name, *joint_meter[train_ds.classes[cls_name]])
            for cls_name in sorted(train_ds.classes.keys())
        ]), file=txt_file, flush=True)

        if args.checkpoint_dir is not None:
            save_name = f"sa{joint_meter.acc:.1f}-sp{joint_meter.sp:.1f}-" \
                        f"sj{joint_meter.jsp:.1f}-ap{joint_meter.ap:.1f}-aj{joint_meter.jap:.1f}"
            torch.save(network.state_dict(), os.path.join(args.checkpoint_dir, f"model{epoch:03d}-{save_name}.pth.tar"))

    txt_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pickle_roots", type=str, nargs="+", required=True)
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--annotation_root", type=str, default=None)

    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--grad_clipping", type=float, default=2000.)

    parser.add_argument("--n_epochs", default=100, type=int)
    parser.add_argument("--eval_each", default=5, type=int)

    parser.add_argument("--delta", type=int, default=2)
    parser.add_argument("--kappa", type=int, default=60)
    parser.add_argument("--video_level_temp", type=float, default=0.001)

    parser.add_argument("--checkpoint_dir", type=str, default="./logs")
    parser.add_argument("--ds_no_augment", action="store_true")

    parser.add_argument("--noise_adapt_weight_root", type=str, default=None)
    parser.add_argument("--noise_adapt_weight_threshold_file", type=str, default=None)

    main(parser.parse_args())
