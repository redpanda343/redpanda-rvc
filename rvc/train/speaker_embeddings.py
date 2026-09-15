import torch


def match_speaker_embedding_scale(fresh_embeddings, pretrained_embeddings):
    if fresh_embeddings.ndim != 2 or pretrained_embeddings.ndim != 2:
        raise ValueError("Speaker embeddings must be two-dimensional.")
    if fresh_embeddings.shape[0] == 0 or pretrained_embeddings.shape[0] == 0:
        raise ValueError("Speaker embeddings must contain at least one row.")
    if fresh_embeddings.shape[1] != pretrained_embeddings.shape[1]:
        raise ValueError("Speaker embedding dimensions do not match.")
    if (
        not fresh_embeddings.is_floating_point()
        or not pretrained_embeddings.is_floating_point()
    ):
        raise TypeError("Speaker embeddings must use a floating-point dtype.")

    work_dtype = (
        torch.float64
        if fresh_embeddings.dtype == torch.float64
        or pretrained_embeddings.dtype == torch.float64
        else torch.float32
    )
    fresh = fresh_embeddings.detach().to(dtype=work_dtype)
    pretrained = pretrained_embeddings.detach().to(
        device=fresh.device,
        dtype=work_dtype,
    )
    fresh_mean_norm = fresh.norm(dim=1).mean()
    pretrained_mean_norm = pretrained.norm(dim=1).mean()

    if not torch.isfinite(fresh_mean_norm) or fresh_mean_norm <= 0:
        raise ValueError("Fresh speaker embedding scale must be finite and positive.")
    if not torch.isfinite(pretrained_mean_norm) or pretrained_mean_norm <= 0:
        raise ValueError(
            "Pretrained speaker embedding scale must be finite and positive."
        )

    return (fresh * (pretrained_mean_norm / fresh_mean_norm)).to(
        dtype=fresh_embeddings.dtype
    )
