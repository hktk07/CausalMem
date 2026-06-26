import os

import torch

from .llava_arch_v3 import FOSSCache as BaseFOSSCache
from .llava_arch_v3 import LlavaMetaForCausalLM as BaseLlavaMetaForCausalLM
from .llava_arch_v3 import LlavaMetaModel


class FOSSCache(BaseFOSSCache):  # 主方法
    def __init__(
        self,
        *args,
        time_weight=0.2,
        time_power=1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        time_weight = float(time_weight)
        time_weight = min(max(time_weight, 0.0), 1.0)

        self.time_weight = time_weight
        self.energy_weight = 1.0 - time_weight
        self.time_power = float(time_power)
        print("self.time_power", self.time_power)
        print("self.time_weight", self.time_weight)

    def _normalize_scores(self, scores):
        if scores.numel() == 0:
            return scores.float()

        scores = scores.float()
        score_min = scores.min()
        score_max = scores.max()
        denom = torch.clamp(score_max - score_min, min=1e-6)
        return (scores - score_min) / denom

    def _compute_time_scores(self, current_frame_idx):
        if self.mem1_frame_ids.numel() == 0:
            return torch.empty(0, device=self.device, dtype=torch.float32)

        # Recent frames receive larger time scores; older frames receive smaller ones.
        denom = max(1, current_frame_idx + 1)
        time_scores = (self.mem1_frame_ids.float() + 1.0) / float(denom)

        if self.time_power != 1.0:
            time_scores = torch.pow(time_scores.clamp(min=0.0), self.time_power)

        return time_scores.clamp(0.0, 1.0).to(device=self.device, dtype=torch.float32)

    @torch.no_grad()
    def process_frame(self, X_new, frame_idx=None, verbose=False, total_num_frames=None, flag=0):
        if X_new is None or X_new.numel() == 0:
            if flag == 1:
                self.memory_buffer = self._assemble_and_sort()
                return self.memory_buffer
            return None

        X_new = X_new.to(device=self.device, dtype=self.dtype)
        N, D = X_new.shape
        if frame_idx is None:
            raise ValueError("process_frame must receive frame_idx")

        self._update_basis(X_new, N, D)

        self.mem1_buffer = torch.cat([self.mem1_buffer, X_new], dim=0)
        new_ids = torch.full((N,), frame_idx, device=self.device, dtype=torch.long)
        self.mem1_frame_ids = torch.cat([self.mem1_frame_ids, new_ids], dim=0)

        current_total = self.mem1_buffer.shape[0]
        if current_total > self.budget:
            if self.U.shape[1] > 0:
                C_all = torch.matmul(self.mem1_buffer, self.U)
                X_hat_all = torch.matmul(C_all, self.U.T)
                energy_all = torch.norm((self.mem1_buffer - X_hat_all).float(), dim=-1)

                energy_scores = self._normalize_scores(energy_all)
                time_scores = self._compute_time_scores(frame_idx)
                combined_scores = (
                    self.energy_weight * energy_scores
                    + self.time_weight * time_scores
                )

                keep_budget = min(self.budget, self.mem1_buffer.size(0))
                _, keep_indices = torch.topk(combined_scores, keep_budget)
                keep_mask = torch.zeros(
                    self.mem1_buffer.size(0),
                    dtype=torch.bool,
                    device=self.device,
                )
                keep_mask[keep_indices] = True

                # Evicted mem1 tokens are discarded directly; no mem2 summary is kept.
                self.mem1_buffer = self.mem1_buffer[keep_mask]
                self.mem1_frame_ids = self.mem1_frame_ids[keep_mask]
            else:
                excess = current_total - self.budget
                self.mem1_buffer = self.mem1_buffer[excess:]
                self.mem1_frame_ids = self.mem1_frame_ids[excess:]

        if flag == 1:
            self.memory_buffer = self._assemble_and_sort()
            return self.memory_buffer
        return None

    def _assemble_and_sort(self):
        return self.mem1_buffer

    @torch.no_grad()
    def get_frame_token_counts(self, total_num_frames=None):
        if self.mem1_frame_ids.numel() == 0:
            return []

        if total_num_frames is None:
            total_num_frames = int(self.mem1_frame_ids.max().item()) + 1

        counts = torch.bincount(self.mem1_frame_ids, minlength=total_num_frames)
        return counts.tolist()


class LlavaMetaForCausalLM(BaseLlavaMetaForCausalLM):
    def stream_compress_video_features(self, image_feature, verbose=True):
        print("v3_time_only_mem1")
        cache = FOSSCache(
            budget=int(os.getenv("FOSS_BUDGET", getattr(self.config, "foss_budget", 12000))),
            dim=image_feature.shape[-1],
            k_max=int(os.getenv("FOSS_K_MAX", getattr(self.config, "foss_k_max", 256))),
            decay=float(os.getenv("FOSS_DECAY", getattr(self.config, "foss_decay", 0.9))),
            device=image_feature.device,
            dtype=image_feature.dtype,
            update_ratio=float(
                os.getenv("FOSS_UPDATE_RATIO", getattr(self.config, "foss_update_ratio", 0.1))
            ),
            max_new_basis=int(
                os.getenv("FOSS_MAX_NEW_BASIS", getattr(self.config, "foss_max_new_basis", 8))
            ),
            time_weight=float(
                os.getenv("FOSS_TIME_WEIGHT", getattr(self.config, "foss_time_weight", 0.2))
            ),
            time_power=float(
                os.getenv("FOSS_TIME_POWER", getattr(self.config, "foss_time_power", 1.0))
            ),
        )

        num_frames = image_feature.shape[0]
        for t in range(num_frames):
            frame_tokens = image_feature[t]
            flag = 1 if t == num_frames - 1 else 0
            cache.process_frame(
                frame_tokens,
                frame_idx=t,
                verbose=verbose,
                total_num_frames=num_frames,
                flag=flag,
            )

        final_counts = cache.get_frame_token_counts(total_num_frames=num_frames)
        return cache.memory_buffer, final_counts


__all__ = ["FOSSCache", "LlavaMetaModel", "LlavaMetaForCausalLM"]
