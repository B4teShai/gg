import pickle
import ast
import json
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import pandas as pd
import os
import torch

torch.sparse.check_sparse_tensor_invariants.disable()


def build_sparse_adj(rows, cols, vals, shape):
    """Build PyTorch sparse tensor from COO arrays."""
    idx = torch.stack([torch.LongTensor(rows), torch.LongTensor(cols)])
    v = torch.FloatTensor(vals)
    return torch.sparse_coo_tensor(idx, v, size=shape).coalesce()


def build_binary_adj(mat, shape):
    """Convert scipy sparse matrix to symmetrically-normalised PyTorch sparse tensor.

    A_norm[r,c] = 1 / (sqrt(row_deg[r]) * sqrt(col_deg[c]))
    Passing mat.T gives the transpose with degrees swapped automatically.
    """
    coo = sp.coo_matrix(mat)
    rows = coo.row.astype(np.int64)
    cols = coo.col.astype(np.int64)
    row_deg = np.array(mat.sum(axis=1)).flatten() + 1e-8
    col_deg = np.array(mat.sum(axis=0)).flatten() + 1e-8
    vals = (1.0 / np.sqrt(row_deg[rows])) * (1.0 / np.sqrt(col_deg[cols]))
    return build_sparse_adj(rows, cols, vals.astype(np.float32), shape)


def build_weighted_adj(mat, edge_weights_arr, shape, log_stats=False):
    """Build weighted adjacency from scipy sparse + weight array.

    The edge_weights file is now produced by feature_extraction_*.ipynb with
    values already normalised to [0, 1] via either Bayesian-shrunk min-max
    (Yelp rating) or log-quantile rank (finance/synthetic amount). So we
    skip the historical sigmoid(log(1+x)) squeeze — that would compress the
    dynamic range again, exactly the effect critic.md §3 flagged ("edge
    weights all ended up in [0.67, 0.86]").

    Instead we clip to [1e-3, 1] and apply symmetric GCN normalisation.
    When log_stats=True the raw and post-norm distributions are printed so
    the paper can report that weights now span the full [0, 1] range.
    """
    coo = sp.coo_matrix((mat != 0).astype(np.float32))
    rows = coo.row.astype(np.int64)
    cols = coo.col.astype(np.int64)

    # Vectorized weight lookup via CSR matrix
    if sp.issparse(edge_weights_arr):
        raw = np.asarray(edge_weights_arr[rows, cols]).flatten()
        matched = int((raw > 0).sum())
        total = len(raw)
        coverage = 100.0 * matched / max(total, 1)
        if log_stats:
            print(f'  edge-weight coverage: {matched:,}/{total:,} ({coverage:.2f}%)')
        # Missing weights fall back to the neutral mid-point so the edge
        # still carries signal but is not artificially inflated.
        raw = np.where(raw > 0, raw, 0.5)
    else:
        edge_weights = edge_weights_arr
        raw = np.full(len(rows), 0.5, dtype=np.float64)
        for idx in range(len(rows)):
            key = (int(rows[idx]), int(cols[idx]))
            if key in edge_weights:
                raw[idx] = edge_weights[key]

    w_hat = np.clip(raw, 1e-3, 1.0)

    if log_stats:
        p = np.percentile(w_hat, [5, 25, 50, 75, 95])
        print(f'  edge-weight distribution (post-clip): '
              f'min={w_hat.min():.4f} max={w_hat.max():.4f} '
              f'mean={w_hat.mean():.4f} std={w_hat.std():.4f}')
        print(f'  percentiles  p05={p[0]:.4f} p25={p[1]:.4f} '
              f'p50={p[2]:.4f} p75={p[3]:.4f} p95={p[4]:.4f}')

    row_deg = np.bincount(rows, weights=w_hat,
                          minlength=shape[0]).astype(np.float64) + 1e-8
    col_deg = np.bincount(cols, weights=w_hat,
                          minlength=shape[1]).astype(np.float64) + 1e-8
    w_norm = w_hat / np.sqrt(row_deg[rows]) / np.sqrt(col_deg[cols])

    if log_stats:
        print(f'  sym-norm weights: min={w_norm.min():.4e} '
              f'max={w_norm.max():.4e} mean={w_norm.mean():.4e}')

    adj = build_sparse_adj(rows, cols, w_norm.astype(np.float32), shape)
    adj_t = build_sparse_adj(cols, rows, w_norm.astype(np.float32), (shape[1], shape[0]))
    return adj, adj_t


def _split_feature_groups(raw):
    groups = []
    for part in str(raw).replace('+', ',').split(','):
        part = part.strip()
        if part:
            groups.append(part)
    return groups or ['all']


def _resolve_feature_columns(meta, side, requested, total_cols):
    """Resolve requested named groups to original feature column indices."""
    group_key = f'{side}_feature_groups'
    groups = meta.get(group_key)
    names = meta.get(f'{side}_feature_names', [])
    if not groups:
        return list(range(total_cols)), names[:total_cols]

    requested_groups = _split_feature_groups(requested)
    expanded = []
    for group in requested_groups:
        if group == 'all':
            expanded.extend(meta.get('default_node_feature_groups') or [
                g for g in meta.get('group_order', groups.keys()) if g != 'degree'
            ])
        elif group == 'all_plus_degree':
            expanded.extend(meta.get('group_order') or list(groups.keys()))
        else:
            expanded.append(group)

    wanted = set()
    unknown = []
    for group in expanded:
        cols = groups.get(group)
        if cols is None:
            unknown.append(group)
            continue
        wanted.update(int(c) for c in cols)
    if unknown:
        print(f'WARNING: unknown {side} feature groups ignored: {unknown}')

    cols = [c for c in range(total_cols) if c in wanted]
    selected_names = [names[c] if c < len(names) else f'col_{c}' for c in cols]
    return cols, selected_names


class DataHandler:
    def __init__(self, args):
        self.args = args
        _script_dir = os.path.dirname(os.path.abspath(__file__))
        self.predir = os.path.join(_script_dir, '..', 'Datasets', args.data) + os.sep

    def load_data(self):
        args = self.args

        # ---- trn_mat_time: [global_mat, subMat_list, timeMat] ----
        with open(self.predir + 'trn_mat_time', 'rb') as f:
            trnMat = pickle.load(f)
        args.user, args.item = trnMat[0].shape
        print(f'Users: {args.user}, Items(Merchants): {args.item}')

        self.subMat = trnMat[1]

        # ---- sequence ----
        with open(self.predir + 'sequence', 'rb') as f:
            self.sequence = pickle.load(f)

        # Build overall binary train matrix from sequences
        row, col, data = [], [], []
        for uid, item_list in enumerate(self.sequence):
            for iid in item_list:
                row.append(uid)
                col.append(iid)
                data.append(1)
        self.trnMat = csr_matrix(
            (np.array(data), (np.array(row), np.array(col))),
            shape=(args.user, args.item)
        )

        # ---- tst_int ----
        with open(self.predir + 'tst_int', 'rb') as f:
            tstInt = pickle.load(f)
        self.tstInt = np.array(tstInt, dtype=object)
        tstStat = np.array([x is not None for x in tstInt])
        self.tstUsrs = np.where(tstStat)[0]
        print(f'Test users: {len(self.tstUsrs)}')

        # ---- val_int ----
        val_path = self.predir + 'val_int'
        if os.path.isfile(val_path):
            with open(val_path, 'rb') as f:
                valInt = pickle.load(f)
            self.valInt = np.array(valInt, dtype=object)
            valStat = np.array([x is not None for x in valInt])
            self.valUsrs = np.where(valStat)[0]
            print(f'Val users: {len(self.valUsrs)}')
        else:
            self.valInt = None
            self.valUsrs = np.array([], dtype=int)
            print('No val_int found, validation disabled.')

        # ---- test_dict ----
        if os.path.isfile(self.predir + 'test_dict'):
            with open(self.predir + 'test_dict', 'rb') as f:
                self.test_dict = pickle.load(f)
        else:
            self.test_dict = {}

        # ---- val_dict ----
        val_dict_path = self.predir + 'val_dict'
        val_csv_path = self.predir + 'val_yelp_merchant.csv'
        if os.path.isfile(val_dict_path):
            with open(val_dict_path, 'rb') as f:
                self.val_dict = pickle.load(f)
            print(f'Val dict loaded: {len(self.val_dict)} users')
        elif os.path.isfile(val_csv_path):
            self.val_dict = {}
            df_val = pd.read_csv(val_csv_path, sep='\t')
            for _, row in df_val.iterrows():
                uid = int(row['user_id']) - 1
                negs = ast.literal_eval(str(row['neg_merchants']))
                self.val_dict[uid] = negs
            print(f'Val dict from CSV: {len(self.val_dict)} users')
        else:
            self.val_dict = {}

        # ---- Load edge weights (optional) ----
        ew_csr = None
        if args.use_edge_features:
            ew_path = self.predir + 'edge_weights.pkl'
            if os.path.isfile(ew_path):
                with open(ew_path, 'rb') as f:
                    edge_weights = pickle.load(f)
                print(f'Edge weights loaded: {len(edge_weights):,} pairs — building CSR...')
                ew_rows = np.array([k[0] for k in edge_weights], dtype=np.int32)
                ew_cols = np.array([k[1] for k in edge_weights], dtype=np.int32)
                ew_vals = np.array(list(edge_weights.values()), dtype=np.float32)
                ew_csr = csr_matrix((ew_vals, (ew_rows, ew_cols)),
                                    shape=(args.user, args.item))
                del edge_weights
                print('  CSR weight matrix built.')
            else:
                print('WARNING: edge_weights.pkl not found, using binary adjacency')

        # ---- Build PyTorch sparse adjacency ----
        self.sub_adj = []
        self.sub_adj_t = []
        for i in range(len(self.subMat)):
            mat = self.subMat[i]
            if args.use_edge_features and ew_csr is not None:
                adj, adj_t = build_weighted_adj(
                    mat, ew_csr, (args.user, args.item),
                    log_stats=(i == 0))
            else:
                adj = build_binary_adj(mat, (args.user, args.item))
                adj_t = build_binary_adj(mat.T, (args.item, args.user))
            self.sub_adj.append(adj)
            self.sub_adj_t.append(adj_t)

        actual_graphs = len(self.subMat)
        if args.graphNum > actual_graphs:
            print(f'Warning: graphNum={args.graphNum} > sub-graphs={actual_graphs}, clamping')
            args.graphNum = actual_graphs
        elif args.graphNum < actual_graphs:
            print(f'Warning: graphNum={args.graphNum} < sub-graphs={actual_graphs}; '
                  f'using the first {args.graphNum} sub-graphs only')
        print(f'Sub-graphs: {args.graphNum}')

        # Clamp testSize to item catalog size (prevents broken eval on small catalogs)
        if args.testSize > args.item:
            print(f'testSize={args.testSize} > item catalog {args.item}, clamping to {args.item}')
            args.testSize = args.item

        # ---- Load node features (optional) ----
        self.user_features = None
        self.merchant_features = None
        if args.use_node_features:
            uf_path = self.predir + 'user_features.npy'
            mf_path = self.predir + 'merchant_features.npy'
            if os.path.isfile(uf_path) and os.path.isfile(mf_path):
                user_np = np.load(uf_path)
                merch_np = np.load(mf_path)
                meta = {}
                meta_path = self.predir + 'feature_meta.json'
                if os.path.isfile(meta_path):
                    with open(meta_path, encoding='utf-8') as f:
                        meta = json.load(f)

                u_cols, u_names = _resolve_feature_columns(
                    meta, 'user', args.node_feature_groups, user_np.shape[1])
                m_cols, m_names = _resolve_feature_columns(
                    meta, 'merchant', args.node_feature_groups, merch_np.shape[1])
                if not u_cols or not m_cols:
                    print('WARNING: selected node feature groups produced no columns; '
                          'disabling node features')
                    args.use_node_features = False
                    return

                user_np = user_np[:, u_cols].astype(np.float32, copy=True)
                merch_np = merch_np[:, m_cols].astype(np.float32, copy=True)

                if args.use_edge_features and not args.keep_node_value_with_edges:
                    user_value_names = set(meta.get('user_value_feature_names', []))
                    merch_value_names = set(meta.get('merchant_value_feature_names', []))
                    if user_value_names or merch_value_names:
                        u_zero = [
                            i for i, name in enumerate(u_names)
                            if name.startswith('value:')
                            or name in user_value_names
                            or name.split(':', 1)[-1] in user_value_names
                        ]
                        m_zero = [
                            i for i, name in enumerate(m_names)
                            if name.startswith('value:')
                            or name in merch_value_names
                            or name.split(':', 1)[-1] in merch_value_names
                        ]
                    else:
                        u_zero = [int(meta.get('user_value_col', 1))]
                        m_zero = [int(meta.get('merchant_value_col', 1))]
                    for col in u_zero:
                        if 0 <= col < user_np.shape[1]:
                            user_np[:, col] = 0.0
                    for col in m_zero:
                        if 0 <= col < merch_np.shape[1]:
                            merch_np[:, col] = 0.0
                    if u_zero or m_zero:
                        print(f'  Zeroed node value feature columns because edge weights '
                              f'are enabled: user={u_zero}, merchant={m_zero}')

                self.user_features = torch.FloatTensor(user_np)
                self.merchant_features = torch.FloatTensor(merch_np)
                args.d_u = self.user_features.shape[1]
                args.d_v = self.merchant_features.shape[1]
                args.selected_user_feature_names = u_names
                args.selected_merchant_feature_names = m_names
                print(f'User features: {self.user_features.shape}, '
                      f'Merchant features: {self.merchant_features.shape}')
                print(f'  Node feature groups: {args.node_feature_groups}')

            else:
                print('WARNING: feature files not found, disabling node features')
                args.use_node_features = False

        # ---- Optional: low/mid/high user segments for segmented eval ----
        self.user_segments = None
        self.segment_meta  = None
        seg_path = self.predir + 'user_segments.pkl'
        if os.path.isfile(seg_path):
            with open(seg_path, 'rb') as f:
                payload = pickle.load(f)
            raw_seg = payload.get('segments', {}) or {}
            self.user_segments = {
                k: [int(u) for u in v] for k, v in raw_seg.items()
            }
            self.segment_meta = payload.get('meta', {})
            seg_counts = {k: len(v) for k, v in self.user_segments.items()}
            print(f'User segments loaded: {seg_counts}')

    # ------------------------------------------------------------------ #
    #  Sampling helpers (identical to selfGNN-Base)                       #
    # ------------------------------------------------------------------ #

    def neg_sample(self, label_row, samp_size, num_items, exclude):
        negs = []
        max_tries = num_items * 3
        tries = 0
        while len(negs) < samp_size and tries < max_tries:
            r = np.random.randint(num_items)
            if label_row[r] == 0 and r not in exclude:
                negs.append(r)
            tries += 1
        # Pad with last found item if sampling exhausted (very dense users)
        if negs:
            while len(negs) < samp_size:
                negs.append(negs[-1])
        else:
            negs = [np.random.randint(num_items)] * samp_size
        return negs

    def sample_train_batch(self, bat_ids):
        args = self.args
        label_mat = self.trnMat[bat_ids].toarray()
        batch = len(bat_ids)

        pos_u, pos_i, pos_seq = [], [], []
        neg_u, neg_i, neg_seq = [], [], []
        sequences = np.zeros((args.batch, args.pos_length), dtype=np.int64)
        masks = np.zeros((args.batch, args.pos_length), dtype=np.float32)

        for i in range(batch):
            uid = bat_ids[i]
            posset = list(self.sequence[uid])
            tst_item = self.tstInt[uid]
            samp_num = min(args.sampNum, len(posset))
            choose = 1

            if samp_num == 0:
                pos_items = [np.random.randint(args.item)]
                neg_items = [pos_items[0]]
                samp_num = 1
            else:
                choose = np.random.randint(
                    1, max(min(args.pred_num + 1, len(posset) - 3), 1) + 1)
                pos_items = [posset[-choose]] * samp_num
                exclude = set()
                last_item = self.sequence[uid][-1]
                exclude.add(last_item)
                if tst_item is not None:
                    exclude.add(int(tst_item))
                neg_items = self.neg_sample(
                    label_mat[i], samp_num, args.item, exclude)

            for j in range(samp_num):
                pos_u.append(uid);  pos_i.append(pos_items[j]);  pos_seq.append(i)
                neg_u.append(uid);  neg_i.append(neg_items[j]);  neg_seq.append(i)

            seq = posset[:-choose] if choose < len(posset) else posset
            if len(seq) == 0:
                seq = [0]
            if len(seq) <= args.pos_length:
                sequences[i, -len(seq):] = seq
                masks[i, -len(seq):] = 1.0
            else:
                sequences[i] = seq[-args.pos_length:]
                masks[i] = 1.0

        all_u = pos_u + neg_u
        all_i = pos_i + neg_i
        all_seq = pos_seq + neg_seq
        return (np.array(all_u), np.array(all_i),
                sequences, masks, np.array(all_seq))

    def sample_ssl_batch(self, bat_ids):
        args = self.args
        su_locs = [[] for _ in range(args.graphNum)]
        si_locs = [[] for _ in range(args.graphNum)]

        for k in range(args.graphNum):
            label = self.subMat[k][bat_ids].toarray()
            label_binary = (label != 0).astype(np.float32)
            for i, uid in enumerate(bat_ids):
                pos_items = np.where(label_binary[i] != 0)[0]
                ssl_num = min(args.sslNum, len(pos_items) // 2)
                if ssl_num == 0:
                    rand_item = np.random.randint(args.item)
                    su_locs[k].extend([uid, uid])
                    si_locs[k].extend([rand_item, rand_item])
                else:
                    chosen = np.random.choice(pos_items, ssl_num * 2, replace=False)
                    for j in range(ssl_num):
                        su_locs[k].extend([uid, uid])
                        si_locs[k].extend([int(chosen[j]), int(chosen[ssl_num + j])])

        return ([np.array(s) for s in su_locs],
                [np.array(s) for s in si_locs])

    def sample_eval_batch(self, bat_ids, mode='test'):
        args = self.args
        batch = len(bat_ids)

        if mode == 'val':
            eval_int = self.valInt
            eval_dict = self.val_dict
        else:
            eval_int = self.tstInt
            eval_dict = self.test_dict

        u_locs, i_locs, u_locs_seq = [], [], []
        tst_locs = []
        sequences = np.zeros((args.batch, args.pos_length), dtype=np.int64)
        masks = np.zeros((args.batch, args.pos_length), dtype=np.float32)

        for i in range(batch):
            uid = bat_ids[i]
            pos_item = eval_int[uid]

            # Filter out pos_item from neg list (preprocessing may include it when
            # the test merchant was only visited once and removed from training).
            neg_items = np.array(
                [x for x in eval_dict.get(uid, []) if x != pos_item][:args.testSize - 1])
            if len(neg_items) < args.testSize - 1:
                # Pad from existing negatives only (never from all items — that
                # would let visited/positive items re-enter the ranking pool).
                extra = args.testSize - 1 - len(neg_items)
                if len(neg_items) > 0:
                    pad = np.random.choice(neg_items, extra, replace=True)
                else:
                    available = np.array([j for j in range(args.item) if j != pos_item])
                    pad = np.random.choice(available, extra,
                                           replace=len(available) < extra)
                neg_items = np.concatenate([neg_items, pad])
            loc_set = np.concatenate([neg_items, np.array([pos_item])])
            tst_locs.append(loc_set)

            for j in range(len(loc_set)):
                u_locs.append(uid)
                i_locs.append(int(loc_set[j]))
                u_locs_seq.append(i)

            posset = list(self.sequence[uid])
            if len(posset) <= args.pos_length:
                sequences[i, -len(posset):] = posset
                masks[i, -len(posset):] = 1.0
            else:
                sequences[i] = posset[-args.pos_length:]
                masks[i] = 1.0

        return (np.array(u_locs), np.array(i_locs),
                sequences, masks, np.array(u_locs_seq),
                tst_locs)
