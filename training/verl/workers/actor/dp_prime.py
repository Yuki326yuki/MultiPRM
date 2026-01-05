# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process PRM
"""
from typing import Iterable

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.workers.actor import BasePPOActor
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import logprobs_from_logits, log_probs_from_logits_all_rmpad
from verl.models.registry import check_model_support_rmpad
            
from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis

__all__ = ['DataParallelPRIME']
PRIME_LOSS = {
    'ce': core_algos.compute_ce_dpo_loss_rm
}

def _pairwise_bce_loss(delta_scores: torch.Tensor) -> torch.Tensor:
    """-log sigmoid(delta). delta can be any real-valued tensor."""
    return torch.nn.functional.softplus(-delta_scores).mean()

class DataParallelPRIME(BasePPOActor):

    def __init__(
        self,
        config,
        reward_module: nn.Module,
        reference_module: nn.Module,
        reward_optimizer: torch.optim.Optimizer = None,
        prime_loss_fn='ce',
        tokenizer=None,
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.reward_module = reward_module
        self.reference_module = reference_module
        self.reward_optimizer = reward_optimizer
        self.prime_loss_fn = PRIME_LOSS[prime_loss_fn]
        self.tokenizer = tokenizer

        # Optional offline PRM training module (disabled by default)
        self.offline_cfg = (self.config.prime_model.get('offline_data', None) or {})
        self.offline_enabled = bool(self.offline_cfg.get('enabled', False))
        self._offline_sampler = None
        if self.offline_enabled:
            if self.tokenizer is None:
                raise ValueError("offline_data.enabled=True requires passing a tokenizer to DataParallelPRIME")
            from verl.utils.offline_prm import OfflinePRMSampler
            self._offline_sampler = OfflinePRMSampler(cfg=self.offline_cfg, tokenizer=self.tokenizer)
            print(f"[PRIME] Offline PRM module enabled. sources={self._offline_sampler.summary()}")
        self.use_remove_padding = self.config.prime_model.get('use_remove_padding', False)
        print(f'PRM use_remove_padding={self.use_remove_padding}')

        # Online PRM multi-signal losses (CLS/REG/PREF). Defaults preserve original behavior (CLS-only).
        self.online_loss_cfg = (self.config.prime_model.get('online_losses', None) or {})
        self.online_enable_reg = bool(self.online_loss_cfg.get('enable_reg', False))
        self.online_enable_pref = bool(self.online_loss_cfg.get('enable_pref', False))
        self.w_online_cls = float(self.online_loss_cfg.get('loss_weight_cls', 1.0))
        self.w_online_reg = float(self.online_loss_cfg.get('loss_weight_reg', 1.0))
        self.w_online_pref = float(self.online_loss_cfg.get('loss_weight_pref', 0.5))
        self.online_reg_key = str(self.online_loss_cfg.get('reg_key', 'reg'))
        # If True, PREF pairs are built using (correct vs incorrect) when possible; otherwise fall back to reg-score ranking.
        self.pref_use_outcome_when_available = bool(self.online_loss_cfg.get('pref_use_outcome_when_available', True))

    def _make_minibatch_iterator(self, data: DataProto) -> Iterable[DataProto]:
        """Make minibatch iterator for updating the actor
        See PPO paper for details. https://arxiv.org/abs/1707.06347
        """
        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'acc', 'old_log_probs']
        # Optional online signals for PRM training
        if hasattr(data, 'batch'):
            if 'reg' in data.batch:
                select_keys.append('reg')
            if 'score' in data.batch:
                select_keys.append('score')
        data = data.select(batch_keys=select_keys)
        return data.make_iterator(mini_batch_size=self.config.mini_batch_size,
                                  epochs=1)

    def _optimizer_step(self):
        assert self.config.prime_model.optim.grad_clip is not None

        if isinstance(self.reward_module, FSDP):
            grad_norm = self.reward_module.clip_grad_norm_(self.config.prime_model.optim.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.reward_module.parameters(), max_norm=self.config.prime_model.optim.grad_clip)
        self.reward_optimizer.step()
        return grad_norm

    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        raise NotImplementedError

    def _forward_micro_batch(self, module, micro_batch, prompt_length, no_grad=False):
        response_length = micro_batch['responses'].size(-1)
        grad_context = torch.no_grad() if no_grad else torch.enable_grad()
        with grad_context, torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch['attention_mask']
            position_ids = micro_batch['position_ids']
            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, max_seqlen_in_batch = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                      indices).transpose(0, 1)
                output = module(input_ids=input_ids_rmpad,
                                        attention_mask=None,
                                        position_ids=position_ids_rmpad,
                                        use_cache=False)
                logits_rmpad = output.logits.squeeze(0)
                logprobs = log_probs_from_logits_all_rmpad(input_ids_rmpad=input_ids_rmpad,
                                                            logits_rmpad=logits_rmpad,
                                                            indices=indices,
                                                            batch_size=batch_size,
                                                            seqlen=seqlen,
                                                            response_length=response_length)  # (batch, seqlen)
            else:
                output = module(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        position_ids=position_ids,
                                        use_cache=False)
                logits = output.logits[:, -response_length - 1:-1]
                logprobs = logprobs_from_logits(logits, micro_batch['responses'])

        return logprobs


    def compute_implicit_reward(self, micro_batch, log_probs, ref_log_probs):
        response_length = micro_batch['responses'].shape[-1]
        max_positions = micro_batch['attention_mask'][:, -response_length:].sum(-1)

        ref_log_probs.to(log_probs.dtype)
        q = log_probs[:, -response_length:] - ref_log_probs[:, -response_length:]  # this is actually diff of q

        # reward computation does not need gradient. only q needs
        with torch.no_grad():
            step_ends = []
            if self.config.prime_granularity == 'token':
                for i in range(micro_batch['input_ids'].shape[0]):
                    step_ends.append(list(range(max_positions[i])))
            elif self.config.prime_granularity == 'whole':
                for i in range(micro_batch['input_ids'].shape[0]):
                    step_ends.append([max_positions[i] - 1])
            else:
                raise NotImplementedError

            token_level_score = torch.zeros_like(micro_batch['input_ids'][:, -response_length:]).to(torch.float32)
            # the strategy of translating q to reward function:
            for i, step_end in enumerate(step_ends):
                for j in range(len(step_end)):
                    step_range = [min(step_end[j - 1] + 1, response_length - 1) if j > 0 else 0,
                                  min(response_length - 1, step_end[j])]
                    token_level_score[i, step_range[1]] = q[i, step_range[0]:step_range[1] + 1].sum()

        return token_level_score, q

    def _offline_prm_update(self, metrics: dict):
        """Optional offline PRM update.

        This is intentionally *decoupled* from PRIME's online rollout flow:
        - It only updates the PRM (self.reward_module)
        - It does NOT affect the returned token_level_scores for policy advantage
        - It is fully optional via config.prime_model.offline_data.enabled
        """
        if (not self.offline_enabled) or (self._offline_sampler is None):
            return

        cfg = self.offline_cfg
        steps = int(cfg.get('steps_per_iter', 1))
        batch_size = int(cfg.get('batch_size', self.config.micro_batch_size))
        micro_bs = int(cfg.get('micro_batch_size', self.config.micro_batch_size))
        assert batch_size % micro_bs == 0, "offline_data.batch_size must be divisible by offline_data.micro_batch_size"
        grad_accum = batch_size // micro_bs
        beta = float(cfg.get('beta_train', self.config.prime_model.get('beta_train', 0.05)))
        w_pref = float(cfg.get('loss_weight_pref', 0.5))
        w_cls = float(cfg.get('loss_weight_cls', 1.0))
        w_reg = float(cfg.get('loss_weight_reg', 1.0))

        # Mix proportions inside an offline batch
        p_pref = float(cfg.get('mix_pref', 0.2))
        p_cls = float(cfg.get('mix_cls', 0.5))
        p_reg = float(cfg.get('mix_reg', 0.3))
        # Normalize in case user doesn't sum to 1
        denom = max(p_pref + p_cls + p_reg, 1e-8)
        p_pref, p_cls, p_reg = p_pref/denom, p_cls/denom, p_reg/denom

        for _ in range(steps):
            # Build one mixed offline batch
            n_pref = int(round(batch_size * p_pref))
            n_cls = int(round(batch_size * p_cls))
            n_reg = max(batch_size - n_pref - n_cls, 0)

            pref_samples = self._offline_sampler.sample_pref(n_pref) if n_pref > 0 else []
            cls_samples = self._offline_sampler.sample_cls(n_cls) if n_cls > 0 else []
            reg_samples = self._offline_sampler.sample_reg(n_reg) if n_reg > 0 else []

            # Tokenize & compute losses
            self.reward_optimizer.zero_grad()
            total_loss = 0.0

            # (1) CLS/REG: reuse PRIME CE loss (BCE on sigmoid(sum(q)*beta))
            def _run_pointwise(samples, label_key: str, weight: float, tag: str):
                nonlocal total_loss
                if not samples:
                    return
                batch = self._offline_sampler.collate_pointwise(samples)
                bs = batch['input_ids'].shape[0]
                for start in range(0, bs, micro_bs):
                    mb = {k: (v[start:start+micro_bs].cuda() if torch.is_tensor(v) else v) for k, v in batch.items()}
                    prompt_len = int(mb['prompt_length'].max().item())
                    log_prob = self._forward_micro_batch(module=self.reward_module, micro_batch=mb, prompt_length=prompt_len)
                    if self.reference_module is not None:
                        ref_log_prob = self._forward_micro_batch(module=self.reference_module, micro_batch=mb, prompt_length=prompt_len, no_grad=True)
                    else:
                        # If no reference module, fall back to using current log_prob as ref (zero reward). Not recommended.
                        ref_log_prob = log_prob.detach()

                    response_length = mb['responses'].shape[-1]
                    q = log_prob[:, -response_length:] - ref_log_prob[:, -response_length:]
                    eos_mask = mb['eos_mask']
                    labels = mb[label_key].to(torch.float32)
                    loss = (weight * core_algos.compute_ce_dpo_loss_rm(q, labels, eos_mask=eos_mask, beta=beta)) / grad_accum
                    loss.backward()
                    total_loss = total_loss + loss.detach().item()

            _run_pointwise(cls_samples, 'label', w_cls, 'cls')
            _run_pointwise(reg_samples, 'score', w_reg, 'reg')

            # (2) PREF: pairwise logistic on sequence-level rewards
            if pref_samples:
                batch_pos, batch_neg = self._offline_sampler.collate_pairwise(pref_samples)
                # micro-batch over pairs
                for start in range(0, batch_pos['input_ids'].shape[0], micro_bs):
                    mb_pos = {k: v[start:start+micro_bs].cuda() for k, v in batch_pos.items()}
                    mb_neg = {k: v[start:start+micro_bs].cuda() for k, v in batch_neg.items()}

                    def _seq_reward(mb):
                        prompt_len = int(mb['prompt_length'].max().item())
                        log_prob = self._forward_micro_batch(module=self.reward_module, micro_batch=mb, prompt_length=prompt_len)
                        if self.reference_module is not None:
                            ref_log_prob = self._forward_micro_batch(module=self.reference_module, micro_batch=mb, prompt_length=prompt_len, no_grad=True)
                        else:
                            ref_log_prob = log_prob.detach()
                        response_length = mb['responses'].shape[-1]
                        q = log_prob[:, -response_length:] - ref_log_prob[:, -response_length:]
                        # length-normalized sequence score
                        seq = (q * mb['eos_mask']).sum(dim=1) / (mb['eos_mask'].sum(dim=1).clamp_min(1.0))
                        return seq

                    s_pos = _seq_reward(mb_pos)
                    s_neg = _seq_reward(mb_neg)
                    delta = beta * (s_pos - s_neg)
                    loss = (w_pref * _pairwise_bce_loss(delta)) / grad_accum
                    loss.backward()
                    total_loss = total_loss + loss.detach().item()

            grad_norm = self._optimizer_step()
            append_to_dict(metrics, {
                'reward_model/offline_enabled': 1.0,
                'reward_model/offline_total_loss': float(total_loss),
                'reward_model/offline_grad_norm': grad_norm.detach().item(),
            })



    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.reward_module.train()
        beta = self.config.prime_model.get('beta_train', 0.05)
        n_samples = data.meta_info['n_samples']
        prompt_length = data.batch['prompts'].shape[-1]
        acc = data.batch['acc']
        attention_mask = data.batch['attention_mask']
        eos_mask = attention_mask[:, prompt_length:]

        assert self.config.mini_batch_size % self.config.micro_batch_size == 0
        self.gradient_accumulation = self.config.mini_batch_size // self.config.micro_batch_size

        dataloader = self._make_minibatch_iterator(data=data)

        metrics = {}
        token_level_scores = []
        for batch_idx, data in enumerate(dataloader):
            # split batch into micro_batches
            # batch = data.batch#.cuda()
            micro_batches = data.batch.split(self.config.micro_batch_size)

            self.reward_optimizer.zero_grad()

            for data in micro_batches:
                data = data.cuda()  # actor device is cpu when using offload
                batch_attention_mask = data['attention_mask']
                batch_eos_mask = batch_attention_mask[:, prompt_length:]
                batch_acc = data['acc']

                log_prob = torch.cat([self._forward_micro_batch(module=self.reward_module, micro_batch=data[i:i + 1], prompt_length=prompt_length) for i in range(len(data))])
                if self.reference_module is not None:
                    ref_log_prob = torch.cat([self._forward_micro_batch(module=self.reference_module, micro_batch=data[i:i + 1], prompt_length=prompt_length, no_grad=True) for i in range(len(data))])
                else:
                    ref_log_prob = data['old_log_probs']

                token_level_score, q = self.compute_implicit_reward(data, log_prob, ref_log_prob)
                token_level_scores.append(token_level_score)

                # ----- Online PRM multi-signal update (CLS/REG/PREF) -----
                # CLS (outcome correctness) is always available (batch_acc).
                loss_cls = self.prime_loss_fn(q, batch_acc, eos_mask=batch_eos_mask, beta=beta)

                # Optional REG: continuous score in [0,1] if present (e.g., step-score).
                loss_reg = None
                if self.online_enable_reg:
                    if self.online_reg_key in data:
                        reg_labels = data[self.online_reg_key].to(batch_acc.dtype)
                        loss_reg = self.prime_loss_fn(q, reg_labels, eos_mask=batch_eos_mask, beta=beta)
                    elif 'score' in data:
                        reg_labels = data['score'].to(batch_acc.dtype)
                        loss_reg = self.prime_loss_fn(q, reg_labels, eos_mask=batch_eos_mask, beta=beta)

                # Optional PREF: build preference pairs within each prompt-group of size n_samples.
                loss_pref = None
                if self.online_enable_pref and (n_samples is not None) and int(n_samples) > 1:
                    ns = int(n_samples)
                    if (data['responses'].shape[0] % ns) != 0:
                        raise ValueError(
                            f"online PREF requires micro_batch_size to be a multiple of n_samples. "
                            f"Got micro_bs={data['responses'].shape[0]} n_samples={ns}"
                        )
                    # Sequence-level length-normalized implicit reward (before beta scaling).
                    seq_score = (q.to(torch.float32) * batch_eos_mask.to(torch.float32)).sum(dim=1) / (
                        batch_eos_mask.to(torch.float32).sum(dim=1).clamp_min(1.0)
                    )
                    # Optional reg values (for fallback pairing)
                    reg_vals = None
                    if self.online_enable_reg:
                        if self.online_reg_key in data:
                            reg_vals = data[self.online_reg_key].to(torch.float32)
                        elif 'score' in data:
                            reg_vals = data['score'].to(torch.float32)
                    deltas = []
                    for g in range(0, seq_score.shape[0], ns):
                        acc_g = batch_acc[g:g+ns]
                        score_g = seq_score[g:g+ns]
                        pos_j = neg_j = None
                        if self.pref_use_outcome_when_available and (acc_g.max() > 0) and (acc_g.min() < 1):
                            pos_candidates = (acc_g > 0).nonzero(as_tuple=True)[0]
                            neg_candidates = (acc_g <= 0).nonzero(as_tuple=True)[0]
                            pos_j = pos_candidates[score_g[pos_candidates].argmax()]
                            neg_j = neg_candidates[score_g[neg_candidates].argmax()]
                        elif reg_vals is not None:
                            reg_g = reg_vals[g:g+ns]
                            if (reg_g.max() - reg_g.min()) > 1e-6:
                                pos_j = reg_g.argmax()
                                neg_j = reg_g.argmin()
                        if (pos_j is not None) and (neg_j is not None) and (pos_j != neg_j):
                            deltas.append(score_g[pos_j] - score_g[neg_j])
                    if deltas:
                        delta = torch.stack(deltas, dim=0)
                        loss_pref = _pairwise_bce_loss(beta * delta)

                # Combine losses (defaults keep CLS-only behavior).
                total_loss = self.w_online_cls * loss_cls
                if loss_reg is not None:
                    total_loss = total_loss + (self.w_online_reg * loss_reg)
                if loss_pref is not None:
                    total_loss = total_loss + (self.w_online_pref * loss_pref)

                prime_loss = total_loss
                loss = total_loss / self.gradient_accumulation
                loss.backward()

            grad_norm = self._optimizer_step()
            data = {
                'reward_model/prm_loss':prime_loss.detach().item(),
                'reward_model/grad_norm': grad_norm.detach().item(),
                }
            append_to_dict(metrics, data)

        self.reward_optimizer.zero_grad()
        torch.cuda.empty_cache()
        

        token_level_scores = torch.cat(token_level_scores, 0).cpu()
        dpo_acc_before = core_algos.compute_dpo_accuracy(token_level_scores, acc, eos_mask=eos_mask,
                                                         n_samples=n_samples)
        data = {
            'reward_model/dpo_acc_before': dpo_acc_before.detach().item(),
        }
        append_to_dict(metrics, data)

        # Optional offline PRM update (doesn't affect returned token_level_scores)
        self._offline_prm_update(metrics)

        if self.config.prime_model.update == "before":
            token_level_scores = []
            dataloader = self._make_minibatch_iterator(data=data)
            for batch_idx, data in enumerate(dataloader):
                micro_batches = data.batch.split(self.config.micro_batch_size)
                for data in micro_batches:
                    data = data.cuda()
                    batch_attention_mask = data['attention_mask']
                    batch_eos_mask = batch_attention_mask[:, prompt_length::]
                    batch_acc = data['acc']

                    log_prob = torch.cat([self._forward_micro_batch(module=self.reward_module, micro_batch=data[i:i + 1], prompt_length=prompt_length) for i in range(len(data))])
                    if self.reference_module is not None:
                        ref_log_prob = torch.cat([self._forward_micro_batch(module=self.reference_module, micro_batch=data[i:i + 1], prompt_length=prompt_length, no_grad=True) for i in range(len(data))])
                    else:
                        ref_log_prob = data['old_log_probs']

                    token_level_score, q = self.compute_implicit_reward(data, log_prob, ref_log_prob, prompt_length)
                    token_level_scores.append(token_level_score)

            token_level_scores = torch.cat(token_level_scores, 0).cpu()
            dpo_acc_after = core_algos.compute_dpo_accuracy(token_level_scores, acc, eos_mask=eos_mask,
                                                            n_samples=n_samples)
            data = {
                'reward_model/dpo_acc_after': dpo_acc_after.detach().item(),
            }
            append_to_dict(metrics, data)
            torch.cuda.empty_cache()

        if self.config.prime_norm == 'batch_norm':  # this method will still consider the relative value of rewards. The key is to control the absolute value of RETURN from being too high. so the normalization is done by controlling the maximum of reverse cumulative sum
            reverse_cumsum = torch.cumsum(token_level_scores.flip(dims=[1]),dim=-1).flip(dims=[1])
            token_level_scores = token_level_scores/(reverse_cumsum.abs().max()+1e-6)
        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        return token_level_scores, metrics
