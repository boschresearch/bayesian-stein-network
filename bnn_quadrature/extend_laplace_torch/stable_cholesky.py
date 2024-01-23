import torch


def stable_cholesky(M: torch.Tensor):
    jitter = 1e-8
    max_jitter = 1e3
    has_converged = False
    while not has_converged and jitter < max_jitter:
        try:
            print(f"Using jitter value: {jitter}")
            M = M + torch.diag(torch.ones(M.shape[-1]) * jitter)
            Lf = torch.linalg.cholesky(M)
            has_converged = True
        except RuntimeError as e:
            jitter = jitter * 10
            has_converged = False
    if not has_converged:
        raise RuntimeError(
            f"{e} - To ensure PSD matrix consider raising the jitter valued. "
            f"Current jitter value: {jitter}."
        )
    return Lf
