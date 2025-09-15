import nltk
import evaluate

import torch
import torch.nn.functional as F
import numpy as np

from utils.streaming_utils import process_batch
from sklearn.metrics import precision_score, recall_score

import matplotlib.pyplot as plt
import seaborn as sns

from utils.tokenization import get_tokenizer_max_length


def text_to_embedding(text, flag, encoder, normalize_embeddings, max_length=32, device='cpu', batch_size=1024):
    max_length = min(get_tokenizer_max_length(encoder.tokenizer), max_length)
    text = [t[:max_length * 5] for t in text]
    output = {}

    tt = encoder.tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    output.update({f"{flag}_{key}": value for key, value in tt.items()})
    if "token_name_idxs" in output: output.pop("token_name_idxs")
    batch = { k: v.to(device) for k,v in output.items()}

    # process_batch in batches of size batch_size. process_batch[flag] is a tensor of embeddings
    output_embs = []
    for i in range(0, len(text), batch_size):
        batch_slice = {k: v[i:i+batch_size] for k, v in batch.items()}
        output_embs.append(process_batch(batch_slice, {flag: encoder}, normalize_embeddings, device)[flag])
    return torch.cat(output_embs)

    # return process_batch(batch, {flag: encoder}, normalize_embeddings, device)[flag]


def generate_text(inverter, embeddings, max_seq_length=32):
    gen_kwargs = {
        "early_stopping": False,
        "num_beams": 4,
        "do_sample": False,
        "no_repeat_ngram_size": 0,
        'min_length': 1,
        'max_length': max_seq_length,
    }
    regenerated = inverter.generate(
        inputs={
            "frozen_embeddings": embeddings,
        },
        generation_kwargs=gen_kwargs,
    )

    output_strings = inverter.tokenizer.batch_decode(
        regenerated, skip_special_tokens=True
    )
    return output_strings


def _calculate_token_f1(predictions, references):
    true_words = set(nltk.tokenize.word_tokenize(references))
    pred_words = set(nltk.tokenize.word_tokenize(predictions))

    TP = len(true_words & pred_words)
    FP = len(true_words) - len(true_words & pred_words)
    FN = len(pred_words) - len(true_words & pred_words)

    precision = (TP) / (TP + FP + 1e-20)
    recall = (TP) / (TP + FN + 1e-20)

    try:
        f1 = (2 * precision * recall) / (precision + recall + 1e-20)
    except ZeroDivisionError:
        f1 = 0.0
    return f1


def calculate_scores(score_flag, target_text, translation_text):
    if score_flag == 'bleu':
        bleu = evaluate.load("sacrebleu")
        score_func = lambda p, r: bleu.compute(predictions=[p], references=[r])['score']
    elif score_flag == 'f1':
        score_func = _calculate_token_f1
    else:
        raise ValueError(f"Unknown score_flag: {score_flag}")

    return np.mean([score_func(p, r) for p, r in zip(target_text, translation_text)])


def eval_batch(ins, recons, translations):
    recon_res = {}
    translation_res = {}
    for target_flag, emb in ins.items():
        emb = emb / emb.norm(dim=1, keepdim=True)
        in_distances = 1 - (emb @ emb.T)
        rec = recons[target_flag]
        rec = rec / rec.norm(dim=1, keepdim=True)
        rec_distances = 1 - (rec @ rec.T)
        recon_res[target_flag] = {
            "mse": F.mse_loss(emb, rec).item(),
            "cos": F.cosine_similarity(emb, rec).mean().item(),
            "std": rec.std(dim=0).mean().item(),
            "vsp": (in_distances - rec_distances).abs().mean().item(),
            "cos_var": F.cosine_similarity(emb, rec).var().item(),
            "vsp_var": (in_distances - rec_distances).abs().var().item()
        }
        translation_res[target_flag] = {}
        for flag, trans in translations[target_flag].items():
            trans = trans / trans.norm(dim=1, keepdim=True)
            out_distances = 1 - (trans @ trans.T)
            translation_res[target_flag][flag] = {
                "mse": F.mse_loss(emb, trans).item(),
                "cos": F.cosine_similarity(emb, trans).mean().item(),
                "std": trans.std(dim=0).mean().item(),
                "vsp": (in_distances - out_distances).abs().mean().item(),
                "cos_var": F.cosine_similarity(emb, trans).var().item(),
                "vsp_var": (in_distances - out_distances).abs().var().item()
            }
    return recon_res, translation_res


def text_batch(ins, recons, translations, inverters, encoders, normalize_embeddings, max_seq_length=32, device='cpu'):
    recon_res = {}
    translation_res = {}

    for target_flag, inverter in inverters.items():
        gt = generate_text(inverter, ins[target_flag], max_seq_length=max_seq_length)
        gt_emb = text_to_embedding(gt, target_flag, encoders[target_flag], normalize_embeddings, max_seq_length, device)

        rec = recons[target_flag]
        rec = rec / rec.norm(dim=1, keepdim=True)
        rec_text = generate_text(inverter, rec, max_seq_length=max_seq_length)
        rec_emb = text_to_embedding(rec_text, target_flag, encoders[target_flag], normalize_embeddings, max_seq_length, device)
        recon_res[target_flag] = {
            "bleu": calculate_scores('bleu', gt, rec_text),
            "f1": calculate_scores('f1', gt, rec_text),
            "t_cos": F.cosine_similarity(gt_emb, rec_emb).mean().item(),
        }
        print('gt:', gt[0])
        print('rec:', rec_text[0])
        translation_res[target_flag] = {}
        for flag, trans in translations[target_flag].items():
            trans = trans / trans.norm(dim=1, keepdim=True)
            trans_text = generate_text(inverter, trans, max_seq_length=max_seq_length)
            trans_emb = text_to_embedding(trans_text, target_flag, encoders[target_flag], normalize_embeddings, max_seq_length, device)
            translation_res[target_flag][flag] = {
                "bleu": calculate_scores('bleu', gt, trans_text),
                "f1": calculate_scores('f1', gt, trans_text),
                "t_cos": F.cosine_similarity(trans_emb, gt_emb).mean().item(),
            }
            print(f'trans ({flag} -> {target_flag}):', trans_text[0])
    return recon_res, translation_res


def merge_dicts(full, incremental):
    def recursive_merge(f, i):
        for key, val in i.items():
            if isinstance(val, dict):
                if key not in f or not isinstance(f[key], dict):
                    f[key] = {}
                recursive_merge(f[key], val)
            else:
                if key not in f:
                    f[key] = []
                f[key].append(val)
    
    recursive_merge(full, incremental)


def mean_dicts(full, ses=False, bs=1):
    """
    Recursively replace every leaf-list in `full` by its mean.
    If include_std=True, also add a sibling leaf with suffix '_std'.
    """
    def _recurse(d):
        for key in list(d.keys()):
            val = d[key]
            if isinstance(val, dict):
                _recurse(val)
            elif isinstance(val, list):
                if len(val) > 1:
                    d[key] = np.mean(val)
                else:
                    d[key] = val[0]
                if ses and 'var' in key:
                    d[f"{key.replace('var', 'se')}"] = np.sqrt(d[key]) / np.sqrt(len(val) * bs)
            else:
                d[key] = val

    _recurse(full)
    return full
    


def top_k_accuracy(sims, k=1):
    top_k_preds = np.argsort(sims, axis=1)[:, -k:]  # Get indices of top-k predictions
    correct = np.arange(sims.shape[0])[:, None]  # Ground truth indices
    return np.mean(np.any(top_k_preds == correct, axis=1))  # Check if correct label is in top-k


def get_avg_rank(sims: np.ndarray) -> float:
    ranks = (np.argsort(-sims) == np.arange(sims.shape[0])[:, None])
    amax = ranks.argmax(1)
    return amax.mean() + 1, amax.var()


def create_heatmap(translator, ins, tgt_emb, src_emb, top_k_size, heatmap_size=None, k=16) -> dict:
    res = {}
    ins = {k: v[:top_k_size] for k, v in ins.items()}
    # TODO: Can we just pass in translations?
    trans = translator.translate_embeddings(ins[src_emb], src_emb, tgt_emb)
    ins_norm = F.normalize(ins[tgt_emb].cpu(), p=2, dim=1)
    trans_norm = F.normalize(trans.cpu(), p=2, dim=1)
    sims = (ins_norm @ trans_norm.T).numpy()
    res[f'{src_emb}_{tgt_emb}_top_1_acc'] = (sims.argmax(axis=1) == np.arange(sims.shape[0])).mean()
    res[f'{src_emb}_{tgt_emb}_top_{k}_acc'] = top_k_accuracy(sims, k)
    avg_rank, rank_var = get_avg_rank(sims)
    res[f"{src_emb}_{tgt_emb}_rank"] = avg_rank
    res[f"{src_emb}_{tgt_emb}_rank_var"] = rank_var
    
    if heatmap_size is not None:
        sims = sims[:heatmap_size, :heatmap_size]
        sims_softmax = F.softmax(torch.tensor(sims) * 100, dim=1).numpy()
        res[f'{src_emb}_{tgt_emb}_heatmap_top_1_acc'] = top_k_accuracy(sims, 1)
        if heatmap_size > k:
            res[f'{src_emb}_{tgt_emb}_heatmap_top_{k}_acc'] = top_k_accuracy(sims, k)

        # plot sims
        fig, ax = plt.subplots(figsize=(6,5))
        sns.heatmap(sims, vmin=0, vmax=1, cmap='coolwarm', ax=ax)
        ax.set_title('Heatmap of cosine similarities')
        ax.set_xlabel(f'Fake ({src_emb}->{tgt_emb})')
        ax.set_ylabel(f'Real ({tgt_emb})')
        plt.tight_layout()
        res[f'{src_emb}_{tgt_emb}_heatmap'] = fig 
        plt.close(fig)
    
        # plot sims w/ softmax
        fig, ax = plt.subplots(figsize=(8,7))
        sns.heatmap(sims_softmax, cmap='Purples', ax=ax)
        ax.set_title('Heatmap of cosine similarities (softmaxed)')
        ax.set_xlabel(f'Fake ({src_emb}->{tgt_emb})')
        ax.set_ylabel(f'Real ({tgt_emb})')
        plt.tight_layout()
        res[f'{src_emb}_{tgt_emb}_heatmap_softmax'] = fig 
        plt.close(fig)

    return res


def compute_topk_accuracy_and_avg_rank(sorted_indices: torch.Tensor, ground_truth: torch.Tensor, k: int):
    """
    Given pre-sorted indices (in descending order) for each sample and the ground truth,
    compute both the top-k accuracy and the average rank of the ground truth labels.
    
    top-k accuracy: For each sample, if any ground truth label is in the top k predictions, count it as correct.
    average rank: For each ground truth label, determine its rank in the sorted predictions (1-indexed) and average over the batch.
    """
    n = sorted_indices.shape[0]
    topk_correct = 0
    all_ranks = []
    for i in range(n):
        # Get the indices for ground truth labels for this sample.
        gt_indices = (ground_truth[i] == 1).nonzero(as_tuple=True)[0]
        # Get top-k predictions for this sample.
        top_k_preds = sorted_indices[i, :k]
        # If any ground truth label is in top-k predictions, count this sample as correct.
        if any(label.item() in top_k_preds.tolist() for label in gt_indices):
            topk_correct += 1
        # For each ground truth label, compute its rank (1-indexed) in the full sorted list.
        for label in gt_indices:
            rank = (sorted_indices[i] == label).nonzero(as_tuple=True)[0].item() + 1
            all_ranks.append(rank)
    topk_accuracy = topk_correct / n
    avg_rank = sum(all_ranks) / len(all_ranks) if all_ranks else 0.0
    return topk_accuracy, avg_rank


def top_k_p_r(predictions: np.ndarray, ground_truth: np.ndarray, k: int):
    n, C = ground_truth.shape
    pred_one_hot = np.zeros((n, C), dtype=int)
    
    # Convert top-k predictions to one-hot encoding
    for i in range(n):
        pred_one_hot[i, predictions[i]] = 1
    
    # Compute precision and recall for multilabel
    precision_k = precision_score(ground_truth, pred_one_hot, average='samples', zero_division=0)
    recall_k = recall_score(ground_truth, pred_one_hot, average='samples', zero_division=0)
    
    return precision_k, recall_k


def classification_batch(ins, translations, labels, gt_mapping, k=1):
    res = {}
    for target_model in labels.keys():
        labels_norm = F.normalize(labels[target_model], p=2, dim=1)
        ins_norm = F.normalize(ins[target_model], p=2, dim=1)
        sims = ins_norm @ labels_norm.T
        # Compute sorted indices once (in descending order)
        sorted_indices = torch.argsort(sims, dim=1, descending=True)
        topk_preds = sorted_indices[:, :k]
        
        # Compute top-k accuracy and average rank in one pass.
        acc, avg_rank = compute_topk_accuracy_and_avg_rank(sorted_indices, gt_mapping, k)
        res[f'{target_model}_top_{k}_acc'] = acc
        res[f'{target_model}_avg_rank'] = avg_rank
        
        p, r = top_k_p_r(topk_preds.cpu().numpy(), gt_mapping.cpu().numpy(), k)
        res[f'{target_model}_P@{k}'] = p
        res[f'{target_model}_R@{k}'] = r
        
        for model, trans in translations[target_model].items():
            trans_norm = F.normalize(trans, p=2, dim=1)
            sims = trans_norm @ labels_norm.T
            sorted_indices = torch.argsort(sims, dim=1, descending=True)
            topk_preds = sorted_indices[:, :k]
            
            acc, avg_rank = compute_topk_accuracy_and_avg_rank(sorted_indices, gt_mapping, k)
            res[f'{model}_{target_model}_top_{k}_acc'] = acc
            res[f'{model}_{target_model}_avg_rank'] = avg_rank
            
            p, r = top_k_p_r(topk_preds.cpu().numpy(), gt_mapping.cpu().numpy(), k)
            res[f'{model}_{target_model}_P@{k}'] = p
            res[f'{model}_{target_model}_R@{k}'] = r

    return res


def eval_loop_(
    cfg, translator, encoders, data_iter, inverters=None, pbar=None, device='cpu', labels=None
):
    recon_res = {}
    translation_res = {}
    heatmap_res = {}
    text_recon_res = {}
    text_translation_res = {}
    classification_res = {}

    top_k_batches = cfg.top_k_batches if hasattr(cfg, 'top_k_batches') else 0
    text_batches = cfg.text_batches if hasattr(cfg, 'text_batches') else 0
    with torch.no_grad():
        for i, batch in enumerate(data_iter):
            ins = process_batch(batch, encoders, cfg.normalize_embeddings, device)
            recons, translations = translator(ins, include_reps=False)
            
            r_res, t_res = eval_batch(ins, recons, translations)
            merge_dicts(recon_res, r_res)
            merge_dicts(translation_res, t_res)
            if i < top_k_batches and hasattr(cfg, 'top_k_size') and hasattr(cfg, 'k') and cfg.top_k_size > 0:
                heatmap_size = cfg.heatmap_size if i == top_k_batches - 1 else None
                batch_res = create_heatmap(translator, ins, cfg.sup_emb, cfg.unsup_emb, cfg.top_k_size, heatmap_size, cfg.k)
                batch_res.update(create_heatmap(translator, ins, cfg.unsup_emb, cfg.sup_emb, cfg.top_k_size, heatmap_size, cfg.k))
                merge_dicts(heatmap_res, batch_res)
            if i < text_batches and inverters is not None and (cfg.sup_emb in inverters or cfg.unsup_emb in inverters):
                t_r_res, t_t_res = text_batch(ins, recons, translations, inverters, encoders, cfg.normalize_embeddings, cfg.max_seq_length, device)
                merge_dicts(text_recon_res, t_r_res)
                merge_dicts(text_translation_res, t_t_res)
            if labels is not None:
                c_res = classification_batch(ins, translations, labels, batch['label'], k=cfg.k)
                merge_dicts(classification_res, c_res)
            if pbar is not None:
                pbar.update(1)

        recon_res = mean_dicts(recon_res, ses=True, bs=cfg.val_bs)
        translation_res = mean_dicts(translation_res, ses=True, bs=cfg.val_bs)
        heatmap_res = mean_dicts(heatmap_res, ses=True, bs=cfg.val_bs)
        text_recon_res = mean_dicts(text_recon_res)
        text_translation_res = mean_dicts(text_translation_res)
        classification_res = mean_dicts(classification_res, ses=True, bs=cfg.val_bs)
        return recon_res, translation_res, heatmap_res, text_recon_res, text_translation_res, classification_res


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0, increase=True):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.opt_val = float('-inf') if increase else float('inf')
        self.increase = increase

    def early_stop(self, val):
        if self.increase and val > (self.opt_val + self.min_delta):
            self.opt_val = val
            self.counter = 0
        elif not self.increase and val < (self.opt_val - self.min_delta):
            self.opt_val = val
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

    def __call__(self, val):
        return self.early_stop(val)
