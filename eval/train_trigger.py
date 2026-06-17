#!/usr/bin/env python
"""학습 트리거 — grasp 핸드오프 '결정'을 관측에서 학습하는 binary 분류기.

hybrid_eval.py의 GT 트리거(sim 내부 chick 좌표로 XY<0.05 판정)를 대체:
입력 = (thermal agentview 256, wrist RGB 256, proprio 8d) → P(지금 핸드오프).
라벨 = 덤프된 GT 거리 dist < DIST_THR. 평가 시 hybrid_eval --trigger learned가
이 파일의 TriggerNet/load_trigger를 import (전처리 일치: 256→CenterCrop224→norm).

학습 (oft env, GPU 권장):
  /home/capstone/miniconda3/envs/oft/bin/python train_trigger.py \
      --data /home/capstone/openvla_ckpts/trigger_data/v1 \
      --out /home/capstone/openvla_ckpts/trigger_ckpt/trigger_resnet18.pt
"""
import argparse
import glob
import os

import numpy as np
import torch
import torch.nn as nn

DIST_THR = 0.05      # 라벨 임계 = 튜닝 트리거와 동일
IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406, 0.485, 0.456, 0.406]).view(6, 1, 1)
IMNET_STD = torch.tensor([0.229, 0.224, 0.225, 0.229, 0.224, 0.225]).view(6, 1, 1)


class TriggerNet(nn.Module):
    """ResNet18(6ch: agentview+wrist 채널 concat) + proprio MLP → 1 logit."""

    def __init__(self, pretrained=True):
        super().__init__()
        from torchvision.models import resnet18
        try:
            from torchvision.models import ResNet18_Weights
            net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        except Exception:
            net = resnet18(pretrained=pretrained)
        w = net.conv1.weight.data  # (64,3,7,7)
        net.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        net.conv1.weight.data = torch.cat([w, w], dim=1) * 0.5
        net.fc = nn.Identity()
        self.backbone = net
        self.prop = nn.Sequential(nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.head = nn.Linear(512 + 64, 1)

    def forward(self, img6, proprio):
        return self.head(torch.cat([self.backbone(img6), self.prop(proprio)], dim=1)).squeeze(-1)


def preprocess(ag_u8, wr_u8, train=False, rng=None):
    """(...,256,256,3) uint8 ×2 → (...,6,224,224) float normalized.
    train=True면 random crop, 아니면 center crop (평가 경로와 동일)."""
    ag = torch.as_tensor(ag_u8).float().div_(255).permute(0, 3, 1, 2)
    wr = torch.as_tensor(wr_u8).float().div_(255).permute(0, 3, 1, 2)
    x = torch.cat([ag, wr], dim=1)  # (N,6,256,256)
    if train:
        i = int(rng.integers(0, 33)) if rng is not None else 16
        j = int(rng.integers(0, 33)) if rng is not None else 16
    else:
        i = j = 16
    x = x[:, :, i:i + 224, j:j + 224]
    return (x - IMNET_MEAN) / IMNET_STD


def load_trigger(ckpt_path, device="cuda"):
    """평가용 로드. 반환 model은 .predict(ag_u8, wr_u8, proprio8) → prob 제공."""
    blob = torch.load(ckpt_path, map_location=device)
    model = TriggerNet(pretrained=False).to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    p_mean = torch.as_tensor(blob["prop_mean"], device=device)
    p_std = torch.as_tensor(blob["prop_std"], device=device)

    @torch.no_grad()
    def predict(ag_u8, wr_u8, proprio8):
        # [::-1] 뷰는 negative stride → torch 거부. contiguous 복사 필수.
        x = preprocess(np.ascontiguousarray(ag_u8)[None],
                       np.ascontiguousarray(wr_u8)[None]).to(device)
        p = (torch.as_tensor(proprio8, device=device).float()[None] - p_mean) / p_std
        return torch.sigmoid(model(x, p)).item()

    model.predict = predict
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val-frac", type=float, default=0.15)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)

    files = sorted(glob.glob(os.path.join(args.data, "trig_seed*.npz")))
    assert files, f"no npz in {args.data}"
    eps = []
    for f in files:
        d = np.load(f)
        eps.append({"ag": d["agentview"], "wr": d["wrist"],
                    "pr": d["proprio"], "dist": d["dist"]})
    n_val = max(2, int(len(eps) * args.val_frac))
    idx = rng.permutation(len(eps))
    val_set = {int(i) for i in idx[:n_val]}
    tr = [e for i, e in enumerate(eps) if i not in val_set]
    va = [e for i, e in enumerate(eps) if i in val_set]

    def flatten(es):
        return (np.concatenate([e["ag"] for e in es]),
                np.concatenate([e["wr"] for e in es]),
                np.concatenate([e["pr"] for e in es]),
                np.concatenate([(e["dist"] < DIST_THR).astype(np.float32) for e in es]))

    ag_t, wr_t, pr_t, y_t = flatten(tr)
    ag_v, wr_v, pr_v, y_v = flatten(va)
    p_mean, p_std = pr_t.mean(0), pr_t.std(0) + 1e-6
    pos = y_t.sum()
    print(f"episodes train/val={len(tr)}/{len(va)} frames={len(y_t)}/{len(y_v)} "
          f"pos_rate train={pos / len(y_t):.3f} val={y_v.mean():.3f}")

    model = TriggerNet(pretrained=True).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(y_t) - pos) / max(pos, 1)).to(dev))
    pm = torch.as_tensor(p_mean, device=dev)
    ps = torch.as_tensor(p_std, device=dev)

    def run_val():
        model.eval()
        probs = []
        with torch.no_grad():
            for k in range(0, len(y_v), args.batch):
                x = preprocess(ag_v[k:k + args.batch], wr_v[k:k + args.batch]).to(dev)
                p = (torch.as_tensor(pr_v[k:k + args.batch], device=dev) - pm) / ps
                probs.append(torch.sigmoid(model(x, p)).cpu().numpy())
        probs = np.concatenate(probs)
        # AUC (rank 기반, sklearn 없이)
        order = np.argsort(probs)
        ranks = np.empty_like(order, dtype=float); ranks[order] = np.arange(len(probs))
        npos, nneg = y_v.sum(), (1 - y_v).sum()
        auc = (ranks[y_v > 0.5].sum() - npos * (npos - 1) / 2) / max(npos * nneg, 1)
        pred = probs > 0.5
        prec = (pred & (y_v > 0.5)).sum() / max(pred.sum(), 1)
        rec = (pred & (y_v > 0.5)).sum() / max(npos, 1)
        return auc, prec, rec, probs

    best_auc = 0.0
    for ep in range(args.epochs):
        model.train()
        perm = rng.permutation(len(y_t))
        tot = 0.0
        for k in range(0, len(perm), args.batch):
            b = perm[k:k + args.batch]
            x = preprocess(ag_t[b], wr_t[b], train=True, rng=rng).to(dev)
            p = (torch.as_tensor(pr_t[b], device=dev) - pm) / ps
            y = torch.as_tensor(y_t[b], device=dev)
            loss = lossf(model(x, p), y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b)
        auc, prec, rec, _ = run_val()
        print(f"epoch {ep}: loss={tot / len(y_t):.4f} val AUC={auc:.4f} P={prec:.3f} R={rec:.3f}")
        if auc > best_auc:
            best_auc = auc
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "prop_mean": p_mean,
                        "prop_std": p_std, "dist_thr": DIST_THR, "val_auc": auc}, args.out)
            print(f"  saved (best AUC {auc:.4f}) → {args.out}")

    # 트리거 시뮬레이션: val 에피소드에서 prob>0.5 3연속이 실제 어느 GT 거리에서 발화하나
    blob = torch.load(args.out, map_location=dev)
    model.load_state_dict(blob["state_dict"]); model.eval()
    fire_dists, miss = [], 0
    with torch.no_grad():
        for e in va:
            x = preprocess(e["ag"], e["wr"]).to(dev)
            p = (torch.as_tensor(e["pr"], device=dev) - pm) / ps
            probs = torch.sigmoid(model(x, p)).cpu().numpy()
            streak = 0; fired = False
            for prob, dist in zip(probs, e["dist"]):
                streak = streak + 1 if prob > 0.5 else 0
                if streak >= 3:
                    fire_dists.append(float(dist)); fired = True; break
            if not fired:
                miss += 1
    print(f"trigger-sim on val: fired {len(fire_dists)}/{len(va)} "
          f"fire_dist(cm)={[round(100 * d, 1) for d in sorted(fire_dists)]} no-fire={miss}")
    print(f"done. best val AUC={best_auc:.4f}")


if __name__ == "__main__":
    main()
