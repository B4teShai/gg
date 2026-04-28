#!/usr/bin/env python3
"""Train recommender baselines on the repository's SelfGNN dataset format.

Supported models:
  - popularity: non-neural item-popularity sanity baseline
  - bprmf: BPR matrix factorization
  - lightgcn: LightGCN over the train interaction graph
  - sasrec: causal Transformer sequential recommender
  - bert4rec: BERT4Rec-style masked-item Transformer

The script reads Datasets/<name>/ produced by dataprocess/* scripts, including
the low/medium/high group datasets.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.sparse.check_sparse_tensor_invariants.disable()


def find_repo_root(start: Path = Path.cwd()) -> Path:
    for path in [start, *start.parents]:
        if (path / "requirements.txt").is_file() and (path / "Datasets").is_dir():
            return path
    return start


def parse_args() -> argparse.Namespace:
    root = find_repo_root()
    parser = argparse.ArgumentParser(description="Train baseline recommenders")
    parser.add_argument(
        "--model",
        required=True,
        choices=["popularity", "bprmf", "lightgcn", "sasrec", "bert4rec"],
    )
    parser.add_argument("--data", required=True, help="dataset name under Datasets/")
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=root / "Results_baselines")
    parser.add_argument("--models-dir", type=Path, default=root / "BaselineModels")
    parser.add_argument("--save-path", default=None)
    parser.add_argument("--device", default="cuda", choices=["cuda", "mps", "cpu"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--reg", type=float, default=1e-5)
    parser.add_argument("--trn-num", type=int, default=10000)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--test-size", type=int, default=1000)

    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--lightgcn-layers", type=int, default=3)
    parser.add_argument("--max-seq-len", type=int, default=200)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    return parser.parse_args()


def set_seed(seed: int, deterministic: bool = False) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def choose_device(name: str) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def fmt(results: dict[str, float]) -> str:
    return " | ".join(f"{k}={v:.4f}" for k, v in results.items())


class RecData:
    def __init__(self, dataset_dir: Path, test_size: int, seed: int):
        self.dataset_dir = dataset_dir
        self.rng = np.random.default_rng(seed)
        with open(dataset_dir / "trn_mat_time", "rb") as f:
            trn_mat_time = pickle.load(f)
        self.train_mat = trn_mat_time[0].astype(np.float32).tocsr()
        self.num_users, self.num_items = self.train_mat.shape
        self.test_size = min(test_size, self.num_items)

        with open(dataset_dir / "sequence", "rb") as f:
            raw_sequence = pickle.load(f)
        with open(dataset_dir / "tst_int", "rb") as f:
            self.test_targets = np.array(pickle.load(f), dtype=object)
        val_path = dataset_dir / "val_int"
        if val_path.is_file():
            with open(val_path, "rb") as f:
                self.val_targets = np.array(pickle.load(f), dtype=object)
        else:
            self.val_targets = np.array([None] * self.num_users, dtype=object)

        self.test_dict = self._load_eval_dict("test_dict")
        self.val_dict = self._load_eval_dict("val_dict")
        self.user_segments, self.segment_meta = self._load_segments()

        self.train_pos: list[list[int]] = []
        self.train_pos_sets: list[set[int]] = []
        for uid in range(self.num_users):
            start, end = self.train_mat.indptr[uid], self.train_mat.indptr[uid + 1]
            items = self.train_mat.indices[start:end].astype(int).tolist()
            self.train_pos.append(items)
            self.train_pos_sets.append(set(items))

        self.sequence: list[list[int]] = []
        for uid in range(self.num_users):
            seq = [int(x) for x in raw_sequence[uid] if int(x) in self.train_pos_sets[uid]]
            if not seq:
                seq = list(self.train_pos[uid])
            self.sequence.append(seq)

        self.all_pos_sets: list[set[int]] = []
        for uid in range(self.num_users):
            pos = set(self.train_pos_sets[uid])
            if self.test_targets[uid] is not None:
                pos.add(int(self.test_targets[uid]))
            if self.val_targets[uid] is not None:
                pos.add(int(self.val_targets[uid]))
            self.all_pos_sets.append(pos)

        self.train_users = np.array(
            [uid for uid, items in enumerate(self.train_pos) if items],
            dtype=np.int64,
        )
        self.test_users = np.array(
            [uid for uid, item in enumerate(self.test_targets) if item is not None],
            dtype=np.int64,
        )
        self.val_users = np.array(
            [uid for uid, item in enumerate(self.val_targets) if item is not None],
            dtype=np.int64,
        )
        self.item_popularity = np.asarray(self.train_mat.sum(axis=0)).reshape(-1).astype(np.float32)

    def _load_eval_dict(self, name: str) -> dict[int, list[int]]:
        path = self.dataset_dir / name
        if not path.is_file():
            return {}
        with open(path, "rb") as f:
            payload = pickle.load(f)
        return {int(k): [int(x) for x in v] for k, v in payload.items()}

    def _load_segments(self) -> tuple[dict[str, list[int]], dict]:
        path = self.dataset_dir / "user_segments.pkl"
        if not path.is_file():
            return {}, {}
        with open(path, "rb") as f:
            payload = pickle.load(f)
        segments = {
            str(k): [int(x) for x in v]
            for k, v in payload.get("segments", {}).items()
        }
        return segments, payload.get("meta", {})

    def sample_negative(self, uid: int) -> int:
        positives = self.all_pos_sets[uid]
        for _ in range(max(self.num_items * 3, 100)):
            item = int(self.rng.integers(0, self.num_items))
            if item not in positives:
                return item
        available = np.array([i for i in range(self.num_items) if i not in positives], dtype=np.int64)
        if len(available) == 0:
            return int(self.rng.integers(0, self.num_items))
        return int(self.rng.choice(available))

    def sample_bpr_batch(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        users = self.rng.choice(self.train_users, size=batch_size, replace=len(self.train_users) < batch_size)
        pos_items = np.empty(batch_size, dtype=np.int64)
        neg_items = np.empty(batch_size, dtype=np.int64)
        for idx, uid in enumerate(users):
            pos_items[idx] = int(self.rng.choice(self.train_pos[int(uid)]))
            neg_items[idx] = self.sample_negative(int(uid))
        return users.astype(np.int64), pos_items, neg_items

    def sample_sequence_batch(
        self,
        batch_size: int,
        max_len: int,
        bert_mask: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        users = self.rng.choice(self.train_users, size=batch_size, replace=len(self.train_users) < batch_size)
        seqs = np.zeros((batch_size, max_len), dtype=np.int64)
        pos_items = np.empty(batch_size, dtype=np.int64)
        neg_items = np.empty(batch_size, dtype=np.int64)
        for row, uid in enumerate(users):
            uid = int(uid)
            seq = self.sequence[uid]
            if len(seq) > 1:
                target_idx = int(self.rng.integers(1, len(seq)))
                history = seq[:target_idx]
                target = int(seq[target_idx])
            else:
                history = []
                target = int(seq[0])

            if bert_mask:
                tokens = (history + [target])[-max_len:]
                shifted = [item + 1 for item in tokens]
                seqs[row, -len(shifted):] = shifted
                seqs[row, -1] = self.num_items + 1
            else:
                shifted = [item + 1 for item in history[-max_len:]]
                if shifted:
                    seqs[row, -len(shifted):] = shifted

            pos_items[row] = target
            neg_items[row] = self.sample_negative(uid)
        return seqs, pos_items, neg_items

    def eval_users_and_targets(self, mode: str) -> tuple[np.ndarray, np.ndarray, dict[int, list[int]]]:
        if mode == "val":
            users = self.val_users
            targets = self.val_targets
            eval_dict = self.val_dict
        else:
            users = self.test_users
            targets = self.test_targets
            eval_dict = self.test_dict
        return users, targets, eval_dict

    def candidate_items(self, uid: int, target: int, eval_dict: dict[int, list[int]]) -> list[int]:
        other_pos = set(self.all_pos_sets[uid])
        other_pos.discard(target)
        negatives = [
            int(x) for x in eval_dict.get(uid, [])
            if int(x) != target and int(x) not in other_pos
        ][: self.test_size - 1]
        if len(negatives) < self.test_size - 1:
            needed = self.test_size - 1 - len(negatives)
            excluded = other_pos | set(negatives) | {target}
            available = np.array(
                [i for i in range(self.num_items) if i not in excluded],
                dtype=np.int64,
            )
            if len(available) == 0:
                available = np.array([i for i in range(self.num_items) if i != target], dtype=np.int64)
            extra = self.rng.choice(available, size=needed, replace=len(available) < needed)
            negatives.extend(extra.astype(int).tolist())
        return negatives + [target]

    def sequence_batch_for_eval(self, users: np.ndarray, max_len: int, bert_mask: bool) -> np.ndarray:
        seqs = np.zeros((len(users), max_len), dtype=np.int64)
        for row, uid in enumerate(users):
            history = self.sequence[int(uid)]
            if bert_mask:
                tokens = [item + 1 for item in history[-(max_len - 1):]]
                values = tokens + [self.num_items + 1]
            else:
                values = [item + 1 for item in history[-max_len:]]
            if values:
                seqs[row, -len(values):] = values
        return seqs


class PopularityModel:
    def __init__(self, scores: np.ndarray):
        self.scores = scores

    def score(self, candidates: np.ndarray) -> np.ndarray:
        return self.scores[candidates]


class BPRMF(nn.Module):
    def __init__(self, num_users: int, num_items: int, dim: int):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, dim)
        self.item_emb = nn.Embedding(num_items, dim)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

    def score(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        user_vec = self.user_emb(users)
        item_vec = self.item_emb(items)
        return (user_vec.unsqueeze(1) * item_vec).sum(dim=-1)

    def bpr_loss(self, users: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor, reg: float) -> torch.Tensor:
        user_vec = self.user_emb(users)
        pos_vec = self.item_emb(pos)
        neg_vec = self.item_emb(neg)
        pos_scores = (user_vec * pos_vec).sum(dim=-1)
        neg_scores = (user_vec * neg_vec).sum(dim=-1)
        loss = -F.logsigmoid(pos_scores - neg_scores).mean()
        if reg > 0:
            loss = loss + reg * (
                user_vec.norm(2).pow(2) + pos_vec.norm(2).pow(2) + neg_vec.norm(2).pow(2)
            ) / users.shape[0]
        return loss


class LightGCN(nn.Module):
    def __init__(self, num_users: int, num_items: int, dim: int, layers: int, train_mat: sp.csr_matrix):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.layers = layers
        self.user_emb = nn.Embedding(num_users, dim)
        self.item_emb = nn.Embedding(num_items, dim)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)
        self.adj = self._build_adj(train_mat)

    def _build_adj(self, train_mat: sp.csr_matrix) -> torch.Tensor:
        coo = train_mat.tocoo()
        rows = np.concatenate([coo.row, coo.col + self.num_users])
        cols = np.concatenate([coo.col + self.num_users, coo.row])
        vals = np.ones(len(rows), dtype=np.float32)
        n_nodes = self.num_users + self.num_items
        deg = np.bincount(rows, weights=vals, minlength=n_nodes).astype(np.float64)
        norm = vals / np.sqrt(deg[rows] + 1e-8) / np.sqrt(deg[cols] + 1e-8)
        idx = torch.LongTensor(np.vstack([rows, cols]))
        return torch.sparse_coo_tensor(idx, torch.FloatTensor(norm), (n_nodes, n_nodes)).coalesce()

    def get_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        all_emb = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        embs = [all_emb]
        adj = self.adj.to(all_emb.device)
        for _ in range(self.layers):
            all_emb = torch.sparse.mm(adj, all_emb)
            embs.append(all_emb)
        out = torch.stack(embs, dim=0).mean(dim=0)
        return out[: self.num_users], out[self.num_users :]

    def score(
        self,
        users: torch.Tensor,
        items: torch.Tensor,
        cached: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        user_emb, item_emb = cached if cached is not None else self.get_embeddings()
        user_vec = user_emb[users]
        item_vec = item_emb[items]
        return (user_vec.unsqueeze(1) * item_vec).sum(dim=-1)

    def bpr_loss(self, users: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor, reg: float) -> torch.Tensor:
        user_emb, item_emb = self.get_embeddings()
        user_vec = user_emb[users]
        pos_vec = item_emb[pos]
        neg_vec = item_emb[neg]
        pos_scores = (user_vec * pos_vec).sum(dim=-1)
        neg_scores = (user_vec * neg_vec).sum(dim=-1)
        loss = -F.logsigmoid(pos_scores - neg_scores).mean()
        if reg > 0:
            raw_user = self.user_emb(users)
            raw_pos = self.item_emb(pos)
            raw_neg = self.item_emb(neg)
            loss = loss + reg * (
                raw_user.norm(2).pow(2) + raw_pos.norm(2).pow(2) + raw_neg.norm(2).pow(2)
            ) / users.shape[0]
        return loss


class SASRec(nn.Module):
    def __init__(
        self,
        num_items: int,
        dim: int,
        max_len: int,
        heads: int,
        blocks: int,
        dropout: float,
    ):
        super().__init__()
        self.num_items = num_items
        self.max_len = max_len
        self.item_emb = nn.Embedding(num_items + 1, dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, dim)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=blocks)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.item_emb.weight, std=0.02)

    def encode(self, seqs: torch.Tensor) -> torch.Tensor:
        batch, length = seqs.shape
        pos = torch.arange(length, device=seqs.device).unsqueeze(0).expand(batch, -1)
        x = self.item_emb(seqs) + self.pos_emb(pos)
        x = self.dropout(x)
        causal_mask = torch.triu(
            torch.ones(length, length, device=seqs.device, dtype=torch.bool),
            diagonal=1,
        )
        x = self.encoder(x, mask=causal_mask)
        x = self.norm(x)
        positions = torch.arange(length, device=seqs.device).unsqueeze(0).expand(batch, -1)
        last_idx = torch.where(seqs != 0, positions, torch.zeros_like(positions)).max(dim=1).values
        return x[torch.arange(batch, device=seqs.device), last_idx]

    def score(self, seqs: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        user_vec = self.encode(seqs)
        item_vec = self.item_emb(items + 1)
        return (user_vec.unsqueeze(1) * item_vec).sum(dim=-1)

    def bpr_loss(self, seqs: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor, reg: float) -> torch.Tensor:
        candidates = torch.stack([pos, neg], dim=1)
        scores = self.score(seqs, candidates)
        loss = -F.logsigmoid(scores[:, 0] - scores[:, 1]).mean()
        if reg > 0:
            loss = loss + reg * self.item_emb(candidates + 1).norm(2).pow(2) / seqs.shape[0]
        return loss


class BERT4Rec(nn.Module):
    def __init__(
        self,
        num_items: int,
        dim: int,
        max_len: int,
        heads: int,
        blocks: int,
        dropout: float,
    ):
        super().__init__()
        self.num_items = num_items
        self.mask_token = num_items + 1
        self.max_len = max_len
        self.item_emb = nn.Embedding(num_items + 2, dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, dim)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=blocks)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.item_emb.weight, std=0.02)

    def encode_mask(self, seqs: torch.Tensor) -> torch.Tensor:
        batch, length = seqs.shape
        pos = torch.arange(length, device=seqs.device).unsqueeze(0).expand(batch, -1)
        x = self.item_emb(seqs) + self.pos_emb(pos)
        x = self.dropout(x)
        x = self.encoder(x)
        x = self.norm(x)
        mask_pos = (seqs == self.mask_token).float().argmax(dim=1).long()
        return x[torch.arange(batch, device=seqs.device), mask_pos]

    def score(self, seqs: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        user_vec = self.encode_mask(seqs)
        item_vec = self.item_emb(items + 1)
        return (user_vec.unsqueeze(1) * item_vec).sum(dim=-1)

    def bpr_loss(self, seqs: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor, reg: float) -> torch.Tensor:
        candidates = torch.stack([pos, neg], dim=1)
        scores = self.score(seqs, candidates)
        loss = -F.logsigmoid(scores[:, 0] - scores[:, 1]).mean()
        if reg > 0:
            loss = loss + reg * self.item_emb(candidates + 1).norm(2).pow(2) / seqs.shape[0]
        return loss


def calc_metrics(preds: np.ndarray, candidates: list[list[int]], targets: list[int]) -> dict[str, float]:
    out = {"HR@10": 0.0, "NDCG@10": 0.0, "HR@20": 0.0, "NDCG@20": 0.0}
    if len(targets) == 0:
        return out
    for row, target in enumerate(targets):
        order = np.argsort(-preds[row])
        ranked = [candidates[row][idx] for idx in order]
        for k in [10, 20]:
            top = ranked[:k]
            if target in top:
                rank = top.index(target)
                out[f"HR@{k}"] += 1.0
                out[f"NDCG@{k}"] += 1.0 / math.log2(rank + 2)
    for key in out:
        out[key] /= len(targets)
    return out


@torch.no_grad()
def evaluate_torch_model(
    model: nn.Module,
    data: RecData,
    device: torch.device,
    model_name: str,
    batch_size: int,
    max_seq_len: int,
    mode: str,
    include_segments: bool = False,
) -> dict | tuple[dict, dict]:
    model.eval()
    users, targets_arr, eval_dict = data.eval_users_and_targets(mode)
    if len(users) == 0:
        return {}

    cached = model.get_embeddings() if model_name == "lightgcn" else None
    all_preds: list[np.ndarray] = []
    all_candidates: list[list[int]] = []
    all_targets: list[int] = []
    all_users: list[int] = []

    for start in range(0, len(users), batch_size):
        batch_users = users[start : start + batch_size]
        candidates = []
        targets = []
        for uid in batch_users:
            target = int(targets_arr[int(uid)])
            candidates.append(data.candidate_items(int(uid), target, eval_dict))
            targets.append(target)
        cand_arr = np.asarray(candidates, dtype=np.int64)
        cand_t = torch.LongTensor(cand_arr).to(device)

        if model_name in {"bprmf", "lightgcn"}:
            user_t = torch.LongTensor(batch_users).to(device)
            if model_name == "lightgcn":
                scores = model.score(user_t, cand_t, cached=cached)
            else:
                scores = model.score(user_t, cand_t)
        elif model_name == "sasrec":
            seq_arr = data.sequence_batch_for_eval(batch_users, max_seq_len, bert_mask=False)
            scores = model.score(torch.LongTensor(seq_arr).to(device), cand_t)
        elif model_name == "bert4rec":
            seq_arr = data.sequence_batch_for_eval(batch_users, max_seq_len, bert_mask=True)
            scores = model.score(torch.LongTensor(seq_arr).to(device), cand_t)
        else:
            raise ValueError(model_name)

        all_preds.append(scores.detach().cpu().numpy())
        all_candidates.extend(candidates)
        all_targets.extend(targets)
        all_users.extend([int(uid) for uid in batch_users])

    preds = np.concatenate(all_preds, axis=0)
    overall = calc_metrics(preds, all_candidates, all_targets)
    if not include_segments or not data.user_segments:
        return overall
    by_segment = {}
    for name, seg_users in data.user_segments.items():
        seg_set = set(seg_users)
        idx = [i for i, uid in enumerate(all_users) if uid in seg_set]
        if not idx:
            continue
        by_segment[name] = calc_metrics(
            preds[idx],
            [all_candidates[i] for i in idx],
            [all_targets[i] for i in idx],
        )
        by_segment[name]["users"] = len(idx)
    return overall, by_segment


def evaluate_popularity(
    model: PopularityModel,
    data: RecData,
    mode: str,
    include_segments: bool = False,
) -> dict | tuple[dict, dict]:
    users, targets_arr, eval_dict = data.eval_users_and_targets(mode)
    candidates = []
    targets = []
    for uid in users:
        target = int(targets_arr[int(uid)])
        candidates.append(data.candidate_items(int(uid), target, eval_dict))
        targets.append(target)
    preds = np.asarray([model.score(np.asarray(row, dtype=np.int64)) for row in candidates])
    overall = calc_metrics(preds, candidates, targets)
    if not include_segments or not data.user_segments:
        return overall
    by_segment = {}
    for name, seg_users in data.user_segments.items():
        seg_set = set(seg_users)
        idx = [i for i, uid in enumerate(users) if int(uid) in seg_set]
        if not idx:
            continue
        by_segment[name] = calc_metrics(
            preds[idx],
            [candidates[i] for i in idx],
            [targets[i] for i in idx],
        )
        by_segment[name]["users"] = len(idx)
    return overall, by_segment


def build_model(args: argparse.Namespace, data: RecData, device: torch.device):
    if args.model == "popularity":
        return PopularityModel(data.item_popularity)
    if args.model == "bprmf":
        return BPRMF(data.num_users, data.num_items, args.embedding_dim).to(device)
    if args.model == "lightgcn":
        return LightGCN(
            data.num_users,
            data.num_items,
            args.embedding_dim,
            args.lightgcn_layers,
            data.train_mat,
        ).to(device)
    if args.model == "sasrec":
        return SASRec(
            data.num_items,
            args.embedding_dim,
            args.max_seq_len,
            args.num_heads,
            args.num_blocks,
            args.dropout,
        ).to(device)
    if args.model == "bert4rec":
        return BERT4Rec(
            data.num_items,
            args.embedding_dim,
            args.max_seq_len,
            args.num_heads,
            args.num_blocks,
            args.dropout,
        ).to(device)
    raise ValueError(args.model)


def train_one_epoch(
    model: nn.Module,
    data: RecData,
    args: argparse.Namespace,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    train_count = min(args.trn_num, len(data.train_users))
    steps = max(1, int(math.ceil(train_count / args.batch)))
    total = 0.0
    for step in range(steps):
        cur_batch = args.batch if step < steps - 1 else max(1, train_count - step * args.batch)
        if args.model in {"bprmf", "lightgcn"}:
            users, pos, neg = data.sample_bpr_batch(cur_batch)
            loss = model.bpr_loss(
                torch.LongTensor(users).to(device),
                torch.LongTensor(pos).to(device),
                torch.LongTensor(neg).to(device),
                args.reg,
            )
        elif args.model in {"sasrec", "bert4rec"}:
            seqs, pos, neg = data.sample_sequence_batch(
                cur_batch,
                args.max_seq_len,
                bert_mask=args.model == "bert4rec",
            )
            loss = model.bpr_loss(
                torch.LongTensor(seqs).to(device),
                torch.LongTensor(pos).to(device),
                torch.LongTensor(neg).to(device),
                args.reg,
            )
        else:
            raise ValueError(args.model)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total += float(loss.item())
    return total / steps


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, default=str)


def main() -> int:
    args = parse_args()
    set_seed(args.seed, args.deterministic)
    root = find_repo_root()
    dataset_dir = args.dataset_dir or (root / "Datasets" / args.data)
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    device = choose_device(args.device)
    save_name = args.save_path or f"{args.data}_{args.model}"
    data = RecData(dataset_dir, args.test_size, args.seed)

    print(f"Model        : {args.model}")
    print(f"Dataset      : {args.data}")
    print(f"Dataset dir  : {dataset_dir}")
    print(f"Device       : {device}")
    print(f"Users/items  : {data.num_users:,}/{data.num_items:,}")
    print(f"Train users  : {len(data.train_users):,}")
    print(f"Val/Test     : {len(data.val_users):,}/{len(data.test_users):,}")
    print(f"Eval size    : {data.test_size}")

    model = build_model(args, data, device)

    if args.model == "popularity":
        val_eval = evaluate_popularity(model, data, mode="val")
        test_eval = evaluate_popularity(model, data, mode="test", include_segments=True)
        if isinstance(test_eval, tuple):
            test_results, test_segments = test_eval
        else:
            test_results, test_segments = test_eval, {}
        print(f"Val:  {fmt(val_eval)}")
        print(f"Test: {fmt(test_results)}")
        result = {
            "model": args.model,
            "dataset": args.data,
            "val_results": val_eval,
            "test_results": test_results,
            "test_segments": test_segments,
            "args": vars(args),
        }
        result_path = args.results_dir / f"{save_name}.json"
        save_json(result_path, result)
        print(f"Results saved to {result_path}")
        return 0

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val_ndcg = -1.0
    best_val_results: dict[str, float] = {}
    best_epoch = 0
    no_improve = 0
    history = []
    args.models_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.models_dir / f"{save_name}.pt"

    print("=" * 72)
    print(f"Training {args.model} for {args.epochs} epochs")
    print("=" * 72)
    for epoch in range(args.epochs):
        start = time.time()
        loss = train_one_epoch(model, data, args, optimizer, device)
        elapsed = time.time() - start
        print(f"Epoch {epoch}/{args.epochs} | loss={loss:.4f} | {elapsed:.1f}s")

        if epoch % args.eval_every != 0:
            continue
        val_results = evaluate_torch_model(
            model,
            data,
            device,
            args.model,
            args.batch,
            args.max_seq_len,
            mode="val",
        )
        print(f"  Val: {fmt(val_results)}")
        history.append({"epoch": epoch, "loss": loss, **{f"val_{k}": v for k, v in val_results.items()}})
        val_ndcg = float(val_results.get("NDCG@10", 0.0))
        if val_ndcg > best_val_ndcg:
            best_val_ndcg = val_ndcg
            best_val_results = dict(val_results)
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), model_path)
            print(f"  New best Val NDCG@10={best_val_ndcg:.4f}. Saved {model_path}")
        else:
            no_improve += 1
            print(f"  No improvement. Patience {no_improve}/{args.patience}")
            if no_improve >= args.patience:
                print("Early stopping.")
                break

    if model_path.is_file():
        model.load_state_dict(torch.load(model_path, map_location=device))
    test_eval = evaluate_torch_model(
        model,
        data,
        device,
        args.model,
        args.batch,
        args.max_seq_len,
        mode="test",
        include_segments=True,
    )
    if isinstance(test_eval, tuple):
        test_results, test_segments = test_eval
    else:
        test_results, test_segments = test_eval, {}
    print("=" * 72)
    print(f"Best epoch: {best_epoch}")
    print(f"Best Val : {fmt(best_val_results)}")
    print(f"Test     : {fmt(test_results)}")
    if test_segments:
        print("Test by segment:")
        for name, metrics in test_segments.items():
            users = metrics.get("users", 0)
            clean = {k: v for k, v in metrics.items() if k != "users"}
            print(f"  {name} (n={users}): {fmt(clean)}")

    result = {
        "model": args.model,
        "dataset": args.data,
        "best_epoch": best_epoch,
        "val_results": best_val_results,
        "test_results": test_results,
        "test_segments": test_segments,
        "args": vars(args),
        "train_history": history,
    }
    result_path = args.results_dir / f"{save_name}.json"
    save_json(result_path, result)
    print(f"Results saved to {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())