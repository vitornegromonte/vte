from typing import Optional

import torch
from sentence_transformers import SentenceTransformer, models


MODEL_PATH = '/private/home/jxm/supervised_translation/model_weights/'
# MODEL_PATH = "/home/rishi/code/data/model_weights/"

HF_FLAGS = {
    'gtr': 'sentence-transformers/gtr-t5-base',
    'gte': 'thenlper/gte-base',
    'gist': 'avsolatorio/GIST-Embedding-v0',
    'stella': 'infgrad/stella-base-en-v2',
    'sentence-t5': 'sentence-transformers/sentence-t5-base',
    'e5': 'intfloat/e5-base-v2',
    'ember': 'llmrails/ember-v1',
    'snowflake': 'Snowflake/snowflake-arctic-embed-m-long',
    'sbert': 'sentence-transformers/all-MiniLM-L12-v2',
    'clip': 'sentence-transformers/clip-ViT-B-32',
    'jina': 'jinaai/jina-embeddings-v2-base-en',
    'bert-nli': 'sentence-transformers/bert-base-nli-mean-tokens',
    'dpr': 'sentence-transformers/facebook-dpr-ctx_encoder-single-nq-base',
    'granite': 'ibm-granite/granite-embedding-278m-multilingual',
    'modernbert': 'answerdotai/ModernBERT-base',
    'modernbert-large': 'answerdotai/ModernBERT-large',
    'nomicbert': 'nomic-ai/nomic-bert-2048',
    'qwen': 'Qwen/Qwen3-Embedding-0.6B',
}


def load_encoder(model_flag, device: str = 'cpu', mixed_precision: Optional[str] = None):
    f = HF_FLAGS.get(model_flag, model_flag)

    model_kwargs = {}
    if mixed_precision is not None:
        if mixed_precision == 'bf16':
            model_kwargs['torch_dtype'] = torch.bfloat16
        elif mixed_precision == 'fp16':
            model_kwargs['torch_dtype'] = torch.float16
        elif mixed_precision == 'no':
            model_kwargs['torch_dtype'] = torch.float32
        else:
            raise ValueError(f"Unknown mixed precision flag {mixed_precision}")
    else:
        model_kwargs['torch_dtype'] = torch.float32
    

    # special loading for gpt-2
    if model_flag.startswith("gpt2"):
        print(f"Loading gpt-2 model {model_flag}")
        transformer = models.Transformer("sentence-transformers/all-MiniLM-L6-v2", max_seq_length=256)
        normalize = models.Normalize()
        if model_flag == "gpt2_mean":
            pooling = models.Pooling(transformer.get_word_embedding_dimension(), pooling_mode="mean")
        elif model_flag == "gpt2_last":
            pooling = models.Pooling(transformer.get_word_embedding_dimension(), pooling_mode="lasttoken")
        else:
            raise ValueError(f"Unknown gpt-2 model {model_flag}")
        encoder = SentenceTransformer(modules=[transformer, pooling, normalize])
    else:
        encoder = SentenceTransformer(f, device=device, trust_remote_code=True, model_kwargs=model_kwargs)
    return encoder.eval()


def get_sentence_embedding_dimension(encoder):
    dim = encoder.get_sentence_embedding_dimension()
    if dim is not None:
        return dim

    # special handling for CLIP models
    dim = encoder[0].model.text_model.config.hidden_size

    return dim