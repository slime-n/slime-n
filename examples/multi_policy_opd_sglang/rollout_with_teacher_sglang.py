"""Custom rollout for multi-policy OPD with an SGLang-backend teacher.

Pipeline:
    1. Generate the student's response via the student's SGLang engine
       (delegated to slime's stock rollout).
    2. POST the full token sequence to teacher_sglang's /generate endpoint
       with ``return_logprob=True`` and ``max_new_tokens=0`` — prefix-only
       scoring, no new generation. The teacher returns per-token logprobs
       for the entire prompt+response prefix.
    3. Slice to response-only tokens, store as a float32 tensor on
       ``sample.teacher_log_probs``. Downstream
       ``_convert_samples_to_train_data`` (slime/ray/rollout.py:995-996)
       hoists per-sample teacher_log_probs into
       ``train_data["teacher_log_probs"]``, which the student's train_actor
       reads via rollout_data so ``apply_opd_kl_to_advantages`` applies it
       as a reverse-KL term.

The POST shape mirrors ``slime.rollout.on_policy_distillation.reward_func``
verbatim, which is the proven single-policy SGLang-OPD path. The only
difference: instead of reading a hardcoded ``args.rm_url``, the teacher's
URL is resolved through the framework's per-model router map via
``get_model_url(args, "teacher_sglang")``, which RolloutManager populates
on ``args.sglang_model_routers`` at startup (slime/ray/rollout.py:1381).
"""

from __future__ import annotations

import torch

from slime.rollout.sglang_rollout import generate, get_model_url
from slime.utils.http_utils import post
from slime.utils.types import Sample


async def generate_with_teacher_sglang(
    args, sample: Sample, sampling_params, evaluation: bool = False
) -> Sample:
    sample = await generate(args, sample, sampling_params)

    # Route this sample to the student's training buffer. Required because
    # both `student` and `teacher_sglang` are registered with RolloutManager
    # (teacher_sglang because it hosts an engine), so the framework's
    # auto-tag fallback in slime/ray/rollout.py:657 does not fire (it only
    # triggers when there's exactly one registered policy). teacher_sglang
    # itself never receives training samples — it only scores them below.
    sample.policy_name = "student"

    if sample.status == Sample.Status.ABORTED or sample.response_length == 0:
        return sample

    teacher_url = get_model_url(args, "teacher_sglang")
    payload = {
        "input_ids": sample.tokens,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
    }
    resp = await post(teacher_url, payload)

    # meta_info.input_token_logprobs is a list of [logprob, token_id] pairs
    # over the full input. Skip index 0 (BOS-like; no preceding context to
    # score), keep the float at index 0 of each pair, then slice to the
    # response-only tail.
    all_logprobs = torch.tensor(
        [item[0] for item in resp["meta_info"]["input_token_logprobs"][1:]],
        dtype=torch.float32,
    )
    sample.teacher_log_probs = all_logprobs[-sample.response_length:]
    return sample
