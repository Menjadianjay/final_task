import os, json, random, itertools, logging, time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TORCH_DYNAMO_DISABLE"] = "1"

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from ViterbiDec import (
    build_transition_matrix,
    evaluate_viterbi,
)

from Shared_resource import (
    Config, Trainer, AlignmentBridge,
    load_bridge, make_loaders, build_manifests,
    normalise, INPUT_DIM, prepare_mfa_label_map,
    NgramLM, build_fuzzy_trie, decode_phones_improved,
    clean_phones, compute_cer,
    _phones_from_labels, _argmax_decode,
)
import Shared_resource as sr

logging.basicConfig(
    format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S", level=logging.INFO
)
log = logging.getLogger(__name__)

MFA_TG_DIR = r"D:\118_Rio\118\src\codeStat\modelling\voiceModelling\kaldialignment\MFA\sampleddata\mfa_textgrid"
OUT_DIR    = r"D:\118_Rio\118\src\codeStat\modelling\voiceModelling\TDNN-TDNN-F\TDNN-F\out"
CSV        = r"D:\118_Rio\118\src\codeStat\preModelling\out\dataBersih.csv"
SPLIT      = r"D:\118_Rio\118\src\codeStat\modelling\voiceModelling\kaldialignment\makeAligner\sampleddata\HasilTrainTestSplit.csv"
LABEL_MAP  = r"D:\118_Rio\118\src\codeStat\modelling\voiceModelling\TDNN-TDNN-F\src\label_map_fixed.csv"
LDA_PKL    = r"D:\118_Rio\118\src\codeStat\modelling\voiceModelling\kaldialignment\makeAligner\sampleddata\lda_model.pkl"
TRAIN_MAN  = f"{OUT_DIR}\\manifest_train.jsonl"
VAL_MAN    = f"{OUT_DIR}\\manifest_val.jsonl"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────

def _semi_ortho_step(W: torch.Tensor, scale: float = 1.0) -> None:
    with torch.no_grad():
        if W.shape[0] > W.shape[1]:
            P = W.T @ W
            I = torch.eye(P.shape[0], device=W.device)
            W -= 0.5 * scale * (W @ (P - I))
        else:
            P = W @ W.T
            I = torch.eye(P.shape[0], device=W.device)
            W -= 0.5 * scale * ((P - I) @ W)


class TDNNFBlock(nn.Module):
    def __init__(
        self,
        dim         : int,
        bottleneck  : int,
        ctx         : List[int],
        bypass_scale: float = 0.66,
        drop        : float = 0.1,
    ):
        super().__init__()
        self.ctx          = sorted(ctx)
        self.bypass_scale = bypass_scale
        self.linear1 = nn.Linear(dim * len(ctx), bottleneck, bias=False)
        self.linear2 = nn.Linear(bottleneck, dim, bias=False)
        self.bn      = nn.BatchNorm1d(dim)
        self.drop    = nn.Dropout(drop)

    def _splice(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        frames  = []
        for off in self.ctx:
            if off < 0:
                pad = torch.zeros(B, -off, D, device=x.device)
                frames.append(torch.cat([pad, x[:, :T + off]], dim=1))
            elif off > 0:
                pad = torch.zeros(B, off, D, device=x.device)
                frames.append(torch.cat([x[:, off:], pad], dim=1))
            else:
                frames.append(x)
        return torch.cat(frames, dim=-1)

    def _bn_safe(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        flat    = x.reshape(B * T, D)
        if self.training and flat.shape[0] < 2:
            return x
        return self.bn(flat).reshape(B, T, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spliced = self._splice(x)
        out     = self.linear2(self.linear1(spliced))
        out     = self.drop(F.relu(self._bn_safe(out)))
        return out + self.bypass_scale * x

    def ortho_step(self, scale: float = 1.0) -> None:
        _semi_ortho_step(self.linear1.weight, scale)


class TDNNFModel(nn.Module):

    _CTX: Dict[int, List[List[int]]] = {
        12: [
            [-2,-1,0,1,2], [-1,0,1], [-3,0,3], [-3,0,3],
            [-3,0,3],       [-3,0,3], [-3,0,3], [-3,0,3],
            [0],             [0],      [0],      [0],
        ],
        15: [
            [-2,-1,0,1,2], [-1,0,1], [-3,0,3], [-3,0,3],
            [-3,0,3],       [-3,0,3], [-3,0,3], [-3,0,3],
            [-3,0,3],       [-3,0,3], [0],      [0],
            [0],             [0],      [0],
        ],
    }

    def __init__(self, cfg: Config, n_blocks: int = 12):
        super().__init__()
        ctxs = self._CTX.get(n_blocks, self._CTX[12])[:n_blocks]

        self.input_proj = nn.Sequential(
            nn.Linear(INPUT_DIM, cfg.tdnnf_hidden),
            nn.BatchNorm1d(cfg.tdnnf_hidden),
            nn.ReLU(),
        )
        self.blocks = nn.ModuleList([
            TDNNFBlock(
                dim          = cfg.tdnnf_hidden,
                bottleneck   = cfg.tdnnf_bottleneck,
                ctx          = ctx,
                bypass_scale = cfg.tdnnf_bypass_scale,
                drop         = cfg.dropout,
            )
            for ctx in ctxs
        ])
        self.prefinal = nn.Sequential(
            nn.Linear(cfg.tdnnf_hidden, cfg.tdnnf_prefinal_dim),
            nn.BatchNorm1d(cfg.tdnnf_prefinal_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
        )
        self.head = nn.Linear(cfg.tdnnf_prefinal_dim, cfg.num_pdfs)

        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        log.info(
            f"TDNN-F: {n_blocks} blok  h={cfg.tdnnf_hidden}  "
            f"btn={cfg.tdnnf_bottleneck}  bypass={cfg.tdnnf_bypass_scale}  "
            f"in={INPUT_DIM}  out={cfg.num_pdfs}  params={n:,}"
        )

    def _apply_seq_bn_safe(self, seq: nn.Sequential, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        flat    = x.reshape(B * T, D)
        if self.training and flat.shape[0] < 2:
            for layer in seq:
                if isinstance(layer, nn.BatchNorm1d):
                    continue
                flat = layer(flat)
        else:
            flat = seq(flat)
        return flat.reshape(B, T, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x   = self._apply_seq_bn_safe(self.input_proj, x)
        for blk in self.blocks:
            x = blk(x)
        out = self._apply_seq_bn_safe(self.prefinal, x)
        return self.head(out)

    def ortho_step(self) -> None:
        for blk in self.blocks:
            blk.ortho_step()


# ─────────────────────────────────────────────────────────────────
# EVALUASI CER  (ganti evaluate_viterbi + WER)
# ─────────────────────────────────────────────────────────────────

def evaluate_cer(
    model      : nn.Module,
    dl,
    id2label   : Dict[int, str],
    fuzzy_trie : Dict,
    lm         : NgramLM,
    device     : str,
    name       : str,
    out_dir    : str,
    prior      : Optional[torch.Tensor] = None,
    log_A      = None,
) -> float:
    """
    Evaluasi model dengan CER (menggantikan evaluate_viterbi + WER).

    Alur per-utterance:
      logits → argmax → _phones_from_labels → clean_phones
      → decode_phones_improved(fuzzy_trie, lm) → compute_cer
    """
    log.info(f"[{name}] Evaluasi CER dimulai")
    t0 = time.time()
    model.eval()

    log_prior = None
    if prior is not None:
        log_prior = torch.log(prior.to(device) + 1e-8)

    cer_list, details = [], []

    with torch.no_grad():
        for batch in dl:
            log_post = F.log_softmax(model(batch["feats"].to(device)), dim=-1)
            if log_prior is not None:
                log_post = log_post - log_prior.unsqueeze(0).unsqueeze(0)

            for b in range(log_post.shape[0]):
                T   = int(batch["lengths"][b])
                ref = normalise(batch["texts"][b]).split()
                if not ref:
                    continue

                labels = _argmax_decode(log_post[b, :T]).cpu().numpy()
                phones = _phones_from_labels(labels, id2label)
                words, _ = decode_phones_improved(phones, lm, fuzzy_trie)
                words    = [w for w in words if len(w) >= 2]

                cer = compute_cer(words, ref)
                cer_list.append(cer)
                details.append({
                    "utt_id"    : batch["audio_ids"][b],
                    "referensi" : " ".join(ref),
                    "phones_raw": " ".join(phones),
                    "hipotesis" : " ".join(words),
                    "ref_len"   : len(ref),
                    "hyp_len"   : len(words),
                    "cer_utt"   : round(cer * 100, 2),
                })

    cer_mean = (sum(cer_list) / len(cer_list) * 100) if cer_list else 100.0
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    csv_path = str(Path(out_dir) / f"{name}_cer_final.csv")
    df       = pd.DataFrame(details)
    summary  = {
        "utt_id"    : "TOTAL",
        "referensi" : "", "phones_raw": "", "hipotesis": "",
        "ref_len"   : int(df["ref_len"].sum()),
        "hyp_len"   : int(df["hyp_len"].sum()),
        "cer_utt"   : round(cer_mean, 2),
    }
    pd.concat([df, pd.DataFrame([summary])], ignore_index=True).to_csv(
        csv_path, index=False, encoding="utf-8"
    )
    log.info(
        f"[{name}] CER = {cer_mean:.2f}%  |  "
        f"{len(cer_list):,} utt  |  {time.time()-t0:.1f}s  →  {csv_path}"
    )
    return cer_mean


# ─────────────────────────────────────────────────────────────────
# UTILITAS
# ─────────────────────────────────────────────────────────────────

@dataclass
class TrialResult:
    trial   : int
    params  : Dict[str, Any]
    cer     : float
    val_loss: float
    time_s  : float


def _val_loss(model: nn.Module, dl) -> float:
    model.eval()
    total, n = 0.0, 0
    dev = next(model.parameters()).device
    with torch.no_grad():
        for batch in dl:
            feats   = batch["feats"].to(dev)
            ali     = batch["ali"].to(dev)
            lengths = batch["lengths"].to(dev)
            logits  = model(feats)
            B, T, C = logits.shape
            mask = (
                torch.arange(T, device=dev).unsqueeze(0) < lengths.unsqueeze(1)
            ) & (ali >= 0)
            if mask.sum() == 0:
                continue
            total += F.cross_entropy(logits[mask], ali[mask], ignore_index=-1).item()
            n += 1
    return total / max(n, 1)


def _build_lm_and_trie(csv_path: str, lm_path: str) -> Tuple[NgramLM, Dict]:
    """Bangun / muat NgramLM dan fuzzy_trie dari CSV kalimat referensi."""
    sentences = (
        pd.read_csv(csv_path)["sentences_bersih"].dropna().tolist()
    )
    if os.path.exists(lm_path):
        lm = NgramLM.load(lm_path)
    else:
        lm = NgramLM(order=3).build(sentences)
        lm.save(lm_path)
    fuzzy_trie = build_fuzzy_trie(sentences)
    return lm, fuzzy_trie


# ─────────────────────────────────────────────────────────────────
# HYPERPARAMETER TUNING
# ─────────────────────────────────────────────────────────────────

def tune(
    cfg,
    bridge,
    id2label        : Dict,
    fuzzy_trie      : Dict,
    lm              : NgramLM,
    log_A,
    search          : str = "random",
    n_trials        : int = 10,
    epochs_per_trial: int = 5,
    seed            : int = 42,
) -> Dict:
    space = {
        "tdnnf_bottleneck"   : [128, 256],
        "tdnnf_bypass_scale" : [0.33, 0.66],
        "lr_start"           : [1e-3, 5e-4],
        "dropout"            : [0.1, 0.2],
        "batch_size"         : [64, 128],
        "n_blocks"           : [12, 15],
    }
    rng = random.Random(seed)

    if search == "grid":
        trials = [dict(zip(space, v)) for v in itertools.product(*space.values())]
    else:
        trials = [{k: rng.choice(v) for k, v in space.items()} for _ in range(n_trials)]

    out_dir = Path(OUT_DIR) / "tdnnf_tune"
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Tuning TDNN-F — {len(trials)} trial x {epochs_per_trial} epoch ({search})")

    results: List[TrialResult] = []
    best_cer, best = float("inf"), {}

    for i, t in enumerate(trials):
        n_blocks = t.pop("n_blocks", 12)
        t_cfg    = Config(**{**cfg.__dict__, **t, "epochs": epochs_per_trial})
        log.info(
            f"Trial {i+1}/{len(trials)}: blk={n_blocks}  "
            f"btn={t_cfg.tdnnf_bottleneck}  bypass={t_cfg.tdnnf_bypass_scale}  "
            f"lr={t_cfg.lr_start}  drop={t_cfg.dropout}"
        )
        t["n_blocks"] = n_blocks
        t0 = time.time()
        try:
            tdl, vdl  = make_loaders(t_cfg, TRAIN_MAN, VAL_MAN, bridge)
            model     = TDNNFModel(t_cfg, n_blocks).to(DEVICE)
            trial_out = str(out_dir / f"trial_{i+1:03d}")
            trainer   = Trainer(
                model, t_cfg, tdl, vdl,
                device=DEVICE, name=f"tdnnf_trial_{i+1:03d}", out=trial_out,
            )
            trainer.train()
            trainer.load_best()
            prior    = trainer.compute_prior()
            cer      = evaluate_cer(
                model, vdl, id2label, fuzzy_trie, lm,
                device=DEVICE, name=f"tdnnf_trial_{i+1:03d}",
                out_dir=trial_out, prior=prior, log_A=log_A,
            )
            val_loss = _val_loss(model, vdl)
        except Exception as e:
            log.error(f"Trial {i+1} gagal: {e}")
            cer, val_loss = 999.0, 999.0

        elapsed = time.time() - t0
        log.info(f"  CER={cer:.2f}%  val_loss={val_loss:.4f}  {elapsed:.0f}s")
        results.append(
            TrialResult(i + 1, dict(t), round(cer, 2), round(val_loss, 5), round(elapsed, 1))
        )
        if cer < best_cer:
            best_cer, best = cer, dict(t)
            log.info("  ★ best baru!")

    rows = [
        {"trial": r.trial, "cer": r.cer, "val_loss": r.val_loss,
         "time_s": r.time_s, **r.params}
        for r in sorted(results, key=lambda x: x.cer)
    ]
    pd.DataFrame(rows).to_csv(out_dir / "all_results.csv", index=False)
    (out_dir / "best.json").write_text(json.dumps(best, indent=2))
    log.info(f"Tuning selesai — Best CER: {best_cer:.2f}%")
    return best


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--mode",   choices=["train", "tune"], default="train")
    p.add_argument("--search", choices=["grid", "random"], default="random")
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--ept",    type=int, default=5)
    p.add_argument("--blocks", type=int, default=12, choices=[12, 15])
    args = p.parse_args()

    log.info(f"TDNN-F ASR Bahasa Jawa — mode={args.mode}  device={DEVICE}")

    # ── Setup label map ──────────────────────────────────────────
    if not os.path.exists(LABEL_MAP):
        log.info("Label map belum ada — membuat dari TextGrid MFA ...")
        prepare_mfa_label_map(MFA_TG_DIR, LABEL_MAP, tier_name="phones")

    # ── Setup manifest ───────────────────────────────────────────
    if not os.path.exists(TRAIN_MAN) or not os.path.exists(VAL_MAN):
        counts = build_manifests(SPLIT, CSV, OUT_DIR)
        if counts.get("train", 0) < 10:
            log.error("Manifest terlalu kecil")
            return
    else:
        log.info("Manifest sudah ada, skip build_manifests")

    # ── Bridge ───────────────────────────────────────────────────
    bridge, id2label, num_pdfs = load_bridge(
        label_map_csv=LABEL_MAP,
        tg_dir=MFA_TG_DIR,
        lda_pkl=LDA_PKL,
    )
    if bridge is None:
        log.error("Bridge gagal dimuat")
        return

    sr.INPUT_DIM = bridge.lda_dim
    global INPUT_DIM
    INPUT_DIM = bridge.lda_dim
    log.info(f"INPUT_DIM = {INPUT_DIM}")

    # ── LM + fuzzy trie ──────────────────────────────────────────
    lm_path        = f"{OUT_DIR}\\lm.pkl"
    lm, fuzzy_trie = _build_lm_and_trie(CSV, lm_path)

    # ── Transition matrix ────────────────────────────────────────
    label2id = {v: k for k, v in id2label.items()}
    log_A    = build_transition_matrix(
        tg_dir           = MFA_TG_DIR,
        id2label         = id2label,
        label2id         = label2id,
        self_loop_weight = 0.5,
    )

    cfg = Config(num_pdfs=num_pdfs or 29, epochs=args.epochs)

    # ── Mode train ───────────────────────────────────────────────
    if args.mode == "train":
        tdl, vdl = make_loaders(cfg, TRAIN_MAN, VAL_MAN, bridge)
        model    = TDNNFModel(cfg, args.blocks).to(DEVICE)
        out_path = f"{OUT_DIR}\\tdnnf"
        trainer  = Trainer(
            model, cfg, tdl, vdl,
            device=DEVICE, name="tdnnf", out=out_path,
        )
        trainer.train()
        trainer.load_best()
        prior = trainer.compute_prior()
        cer   = evaluate_cer(
            model, vdl, id2label, fuzzy_trie, lm,
            device=DEVICE, name="tdnnf", out_dir=out_path,
            prior=prior, log_A=log_A,
        )
        log.info(f"TDNN-F selesai — CER final = {cer:.2f}%")

    # ── Mode tune ────────────────────────────────────────────────
    else:
        best         = tune(
            cfg, bridge, id2label, fuzzy_trie, lm, log_A,
            search=args.search, n_trials=args.trials,
            epochs_per_trial=args.ept,
        )
        n_blocks_best = best.pop("n_blocks", 12)
        final = Config(**{
            **cfg.__dict__,
            **{k: v for k, v in best.items() if k not in ("cer", "time_s")},
            "epochs": args.epochs,
        })
        tdl, vdl = make_loaders(final, TRAIN_MAN, VAL_MAN, bridge)
        model    = TDNNFModel(final, n_blocks_best).to(DEVICE)
        out_path = f"{OUT_DIR}\\tdnnf_best"
        trainer  = Trainer(
            model, final, tdl, vdl,
            device=DEVICE, name="tdnnf_best", out=out_path,
        )
        trainer.train()
        trainer.load_best()
        prior = trainer.compute_prior()
        cer   = evaluate_cer(
            model, vdl, id2label, fuzzy_trie, lm,
            device=DEVICE, name="tdnnf_best", out_dir=out_path,
            prior=prior, log_A=log_A,
        )
        log.info(f"TDNN-F (best params) selesai — CER final = {cer:.2f}%")


if __name__ == "__main__":
    main()