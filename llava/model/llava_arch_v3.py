from abc import ABC, abstractmethod

import math
import re
import torch
import torch.nn as nn
from .multimodal_encoder.builder import build_vision_tower
from .multimodal_resampler.builder import build_vision_resampler
from .multimodal_projector.builder import build_vision_projector

from llava.constants import (
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
import os
from llava.mm_utils import get_anyres_image_grid_shape
from llava.utils import rank0_print
import random


import torch
import torch.nn as nn

class FOSSCache:
    def __init__(
        self,
        budget=12000,
        dim=3584,
        k_max=256,
        decay=0.9,
        device="cuda",
        dtype=torch.bfloat16,
        update_ratio=0.1,
        max_new_basis=8,
    ):
        self.budget = budget
        self.dim = dim
        self.k_max = k_max
        self.decay = decay
        self.device = device
        self.dtype = dtype
        self.update_ratio = update_ratio
        self.max_new_basis = max_new_basis
        print(self.max_new_basis,self.k_max,self.decay)
       
        self.mem1_buffer = torch.empty((0, dim), device=device, dtype=dtype)
        self.mem1_frame_ids = torch.empty((0,), device=device, dtype=torch.long)

        
        self.mem2_bg_vectors = {} # {frame_idx: tensor([dim])}
        self.mem2_bg_counts = {}  # {frame_idx: int} 

        
        self.U = torch.empty((dim, 0), device=device, dtype=dtype)
        
        self.S = torch.empty(0, device=device, dtype=torch.float32)

        
        self.memory_buffer = torch.empty((0, dim), device=device, dtype=dtype)

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

        
        current_total = self.mem1_buffer.shape[0] + len(self.mem2_bg_vectors)

        if current_total > self.budget:
            if self.U.shape[1] > 0:
                
                C_all = torch.matmul(self.mem1_buffer, self.U)
                X_hat_all = torch.matmul(C_all, self.U.T)
                energy_all = torch.norm((self.mem1_buffer - X_hat_all).float(), dim=-1)
               
                fg_budget = max(100, self.budget - len(self.mem2_bg_vectors))
                
                
                _, keep_indices = torch.topk(energy_all, min(fg_budget, self.mem1_buffer.size(0)))
                keep_mask = torch.zeros(self.mem1_buffer.size(0), dtype=torch.bool, device=self.device)
                keep_mask[keep_indices] = True
                
                
                drop_mask = ~keep_mask
                dropped_tokens = self.mem1_buffer[drop_mask]
                dropped_fids = self.mem1_frame_ids[drop_mask]
                
                if dropped_tokens.size(0) > 0:
                    self._merge_to_mem2(dropped_tokens, dropped_fids)

                
                self.mem1_buffer = self.mem1_buffer[keep_mask]
                self.mem1_frame_ids = self.mem1_frame_ids[keep_mask]
            else:
                
                excess = current_total - self.budget
                self.mem1_buffer = self.mem1_buffer[excess:]
                self.mem1_frame_ids = self.mem1_frame_ids[excess:]

        
        if flag == 1:
            self.memory_buffer = self._assemble_and_sort()
            
            return self.memory_buffer
        else:
            
            return None

    def _update_basis(self, X_new, N, D):
   
        if self.U.shape[1] == 0:
            q = min(self.k_max, N, D)
            if q > 0:
                _, _, V = torch.svd_lowrank(X_new.float(), q=q)
                self.U = V.to(self.dtype)
                self.S = torch.ones(self.U.shape[1], device=self.device, dtype=torch.float32)
        else:
            C_new = torch.matmul(X_new, self.U)
            X_hat = torch.matmul(C_new, self.U.T)
            R_new = X_new - X_hat
            energy_new = torch.norm(R_new.float(), dim=-1)
            
            usage = torch.mean(torch.abs(C_new.float()), dim=0)
            
            self.S = self.decay * self.S + (1.0 - self.decay) * usage

            
            k_update = self.max_new_basis
            # print('k_update',k_update)
            _, update_idx = torch.topk(energy_new, k=k_update)
            # _, update_idx = torch.topk(energy_new, k=196)
            # print(update_idx)
            cand = R_new[update_idx].T.float()
            if cand.numel() > 0:
                Q_new, _ = torch.linalg.qr(cand, mode="reduced")
                if self.U.shape[1] > 0 and Q_new.shape[1] > 0:
                    proj = self.U.float() @ (self.U.float().T @ Q_new)
                    Q_new = Q_new - proj
                    if Q_new.numel() > 0:
                        Q_new, _ = torch.linalg.qr(Q_new, mode="reduced")
                if Q_new.numel() > 0:
                    col_norm = torch.norm(Q_new, dim=0)
                    Q_new = Q_new[:, col_norm > 1e-6].to(self.dtype)
                    self.U = torch.cat([self.U, Q_new], dim=1)
                    self.S = torch.cat([self.S, torch.ones(Q_new.shape[1], device=self.device)], dim=0)

            if self.U.shape[1] > self.k_max:
                _, topk_idx = torch.topk(self.S, self.k_max)
                topk_idx, _ = torch.sort(topk_idx)
                self.U = self.U[:, topk_idx]
                self.S = self.S[topk_idx]

    def _merge_to_mem2(self, tokens, fids):
       
        unique_fids = torch.unique(fids)
        for fid in unique_fids:
            fid_item = fid.item()
            mask = (fids == fid)
            new_data = tokens[mask]
            new_sum = new_data.sum(dim=0)
            new_count = new_data.size(0)

            if fid_item in self.mem2_bg_vectors:
                
                old_sum = self.mem2_bg_vectors[fid_item] * self.mem2_bg_counts[fid_item]
                total_count = self.mem2_bg_counts[fid_item] + new_count
                self.mem2_bg_vectors[fid_item] = (old_sum + new_sum) / total_count
                self.mem2_bg_counts[fid_item] = total_count
            else:
                
                self.mem2_bg_vectors[fid_item] = new_sum / new_count
                self.mem2_bg_counts[fid_item] = new_count

    def _assemble_and_sort(self):
        
        if len(self.mem2_bg_vectors) == 0:
            return self.mem1_buffer

       
        bg_keys = sorted(self.mem2_bg_vectors.keys())
        bg_tensor = torch.stack([self.mem2_bg_vectors[k] for k in bg_keys])
        bg_fids = torch.tensor(bg_keys, device=self.device, dtype=torch.long)

        
        all_tokens = torch.cat([self.mem1_buffer, bg_tensor], dim=0)
        all_fids = torch.cat([self.mem1_frame_ids, bg_fids], dim=0)
        
       
        fg_types = torch.zeros(self.mem1_buffer.size(0), device=self.device, dtype=torch.long)
        bg_types = torch.ones(bg_tensor.size(0), device=self.device, dtype=torch.long)
        all_types = torch.cat([fg_types, bg_types], dim=0)

        
        sort_scores = all_fids * 2 + all_types
        sort_idx = torch.argsort(sort_scores, stable=True)

        return all_tokens[sort_idx]

    @torch.no_grad()
    def get_frame_token_counts(self, total_num_frames=None):
        """
        获取最终每一帧保留的 Token 总数 (前景数 + 1个背景)
        """
        if self.mem1_frame_ids.numel() == 0 and len(self.mem2_bg_vectors) == 0:
            return []
        
        
        if total_num_frames is None:
            max_f1 = self.mem1_frame_ids.max().item() if self.mem1_frame_ids.numel() > 0 else 0
            max_f2 = max(self.mem2_bg_vectors.keys()) if self.mem2_bg_vectors else 0
            total_num_frames = int(max(max_f1, max_f2)) + 1

        counts = torch.bincount(self.mem1_frame_ids, minlength=total_num_frames)
        
        
        for fid in self.mem2_bg_vectors.keys():
            if fid < total_num_frames:
                counts[fid] += 1
                
        return counts.tolist()



class LlavaMetaModel:
    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            delay_load = getattr(config, "delay_load", False)
            self.vision_tower = build_vision_tower(config, delay_load=delay_load)
            self.vision_resampler = build_vision_resampler(config, vision_tower=self.vision_tower)
            self.mm_projector = build_vision_projector(config, vision_cfg=self.vision_tower.config)

            if "unpad" in getattr(config, "mm_patch_merge_type", ""):
                self.image_newline = nn.Parameter(torch.empty(config.hidden_size, dtype=self.dtype))

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower
        self.config.vision_tower_pretrained = getattr(model_args, "vision_tower_pretrained", "")

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)
            vision_resampler = build_vision_resampler(model_args, vision_tower=vision_tower)
            for k, v in vision_resampler.config.items():
                setattr(self.config, k, v)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
                self.vision_resampler = [vision_resampler]
            else:
                self.vision_tower = vision_tower
                self.vision_resampler = vision_resampler
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_resampler = self.vision_resampler[0]
                vision_tower = self.vision_tower[0]
            else:
                vision_resampler = self.vision_resampler
                vision_tower = self.vision_tower
            vision_tower.load_model()

            for p in self.vision_resampler.parameters():
                p.requires_grad = True

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, "mm_projector_type", "linear")
        self.config.mm_hidden_size = getattr(vision_resampler, "hidden_size", vision_tower.hidden_size)
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if not hasattr(self.config, "add_faster_video"):
            if model_args.add_faster_video:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.faster_token = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )

        if getattr(self, "mm_projector", None) is None:
            self.mm_projector = build_vision_projector(self.config, vision_cfg=vision_tower.config)

            if "unpad" in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location="cpu")

            def get_w(weights, keyword):
                return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}

            incompatible_keys = self.mm_projector.load_state_dict(get_w(mm_projector_weights, "mm_projector"))
            rank0_print(
                f"Loaded mm projector weights from {pretrain_mm_mlp_adapter}. "
                f"Incompatible keys: {incompatible_keys}"
            )
            incompatible_keys = self.vision_resampler.load_state_dict(
                get_w(mm_projector_weights, "vision_resampler"), strict=False
            )
            rank0_print(
                f"Loaded vision resampler weights from {pretrain_mm_mlp_adapter}. "
                f"Incompatible keys: {incompatible_keys}"
            )


def unpad_image(tensor, original_size):
    """
    tensor: C x H x W
    original_size: (width, height)
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding : current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features


    def get_2dPool(self, image_feature, stride=2):
        height = width = self.get_vision_tower().num_patches_per_side
        num_frames, num_tokens, num_dim = image_feature.shape

        image_feature = image_feature.view(num_frames, height, width, num_dim)
        image_feature = image_feature.permute(0, 3, 1, 2)  # [T, D, H, W]

        if self.config.mm_spatial_pool_mode == "average":
            image_feature = nn.functional.avg_pool2d(
                image_feature, kernel_size=stride, stride=stride
            )

        elif self.config.mm_spatial_pool_mode == "max":
            image_feature = nn.functional.max_pool2d(
                image_feature, kernel_size=stride, stride=stride
            )

        elif self.config.mm_spatial_pool_mode == "bilinear":
            h, w = image_feature.shape[2:]
            scaled_shape = [math.ceil(h / stride), math.ceil(w / stride)]
            frame_chunk = getattr(self.config, "mm_pool_frame_chunk", 4)

            pooled_chunks = []
            for start in range(0, num_frames, frame_chunk):
                chunk = image_feature[start:start + frame_chunk].contiguous()
                pooled_chunk = nn.functional.interpolate(
                    chunk,
                    size=scaled_shape,
                    mode="bilinear",
                    align_corners=False,
                )
                pooled_chunks.append(pooled_chunk)

            image_feature = torch.cat(pooled_chunks, dim=0)

        else:
            raise ValueError(f"Unexpected mm_spatial_pool_mode: {self.config.mm_spatial_pool_mode}")

        image_feature = image_feature.permute(0, 2, 3, 1)
        image_feature = image_feature.reshape(num_frames, -1, num_dim)
        return image_feature

    # ---------------------------------------------------------------------
    # streaming helpers
    # ---------------------------------------------------------------------
    def add_token_per_grid_single_frame(self, frame_feature):
        """
        frame_feature: [num_tokens, dim]
        return: [tokens_with_row_newline, dim]
        """
        model_obj = self.get_model()
        resize_h = int(math.sqrt(frame_feature.shape[0]))
        feature_dim = frame_feature.shape[-1]

        frame_feature = frame_feature.view(1, resize_h, resize_h, feature_dim)      # [1, h, h, d]
        frame_feature = frame_feature.permute(3, 0, 1, 2).contiguous()              # [d, 1, h, h]
        frame_feature = frame_feature.squeeze(1)                                     # [d, h, h]
        frame_feature = torch.cat(
            (
                frame_feature,
                model_obj.image_newline[:, None, None].expand(feature_dim, resize_h, 1).to(frame_feature.device),
            ),
            dim=-1,
        )                                                                            # [d, h, h+1]
        frame_feature = frame_feature.permute(1, 2, 0).contiguous()                 # [h, h+1, d]
        frame_feature = frame_feature.view(-1, feature_dim)                         # [h*(h+1), d]
        return frame_feature

    def add_token_per_frame_single_frame(self, frame_feature):
        """
        frame_feature: [num_tokens, dim]
        return: [num_tokens + 1, dim]
        """
        model_obj = self.get_model()
        frame_feature = torch.cat(
            (
                frame_feature,
                model_obj.image_newline[None].to(frame_feature.device),
            ),
            dim=0,
        )
        return frame_feature


    def stream_compress_video_features(self, image_feature, verbose=True):
        
        print('v3')
    

        cache = FOSSCache(
        budget=int(os.getenv("FOSS_BUDGET", getattr(self.config, "foss_budget", 12000))),
        dim=image_feature.shape[-1],
        k_max=int(os.getenv("FOSS_K_MAX", getattr(self.config, "foss_k_max", 256))),
        decay=float(os.getenv("FOSS_DECAY", getattr(self.config, "foss_decay", 0.9))),
        device=image_feature.device,
        dtype=image_feature.dtype,
        update_ratio=float(os.getenv("FOSS_UPDATE_RATIO", getattr(self.config, "foss_update_ratio", 0.1))),
        max_new_basis=int(os.getenv("FOSS_MAX_NEW_BASIS", getattr(self.config, "foss_max_new_basis", 8))),
    )
        num_frames = image_feature.shape[0]
        for t in range(num_frames):
            frame_tokens = image_feature[t]   # [pooled_tokens, dim]
            if t==num_frames-1:
                flag=1
            else:
                flag=0
            cache.process_frame(
                frame_tokens,
                frame_idx=t,
                verbose=verbose,
                total_num_frames=num_frames,
                flag=flag,
            )

        final_counts = cache.get_frame_token_counts(total_num_frames=num_frames)
       
        return cache.memory_buffer, final_counts

    # ---------------------------------------------------------------------
    # main multimodal prep
    # ---------------------------------------------------------------------
    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        labels,
        images,
        modalities=["image"],
        image_sizes=None,
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if isinstance(modalities, str):
            modalities = [modalities]

        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]

            video_idx_in_batch = []
            for i in range(len(modalities)):
                if modalities[i] == "video":
                    video_idx_in_batch.append(i)

            images_list = []
            for image in images:
                if image.ndim == 4:
                    images_list.append(image)
                else:
                    images_list.append(image.unsqueeze(0))

            concat_images = torch.cat([image for image in images_list], dim=0)
            split_sizes = [image.shape[0] for image in images_list]

            # -------------------------------------------------------------
            # chunked image encoding to avoid OOM
            # -------------------------------------------------------------
            mm_encode_batch_size = getattr(self.config, "mm_encode_batch_size", 32)
            encoded_image_features_list = []
            num_total_images = concat_images.shape[0]

            for start in range(0, num_total_images, mm_encode_batch_size):
                
                print('start',start)
                end = min(start + mm_encode_batch_size, num_total_images)
                image_chunk = concat_images[start:end]
                chunk_features = self.encode_images(image_chunk)  # [chunk_frames, num_tokens, dim]
                encoded_image_features_list.append(chunk_features)

            encoded_image_features = torch.cat(encoded_image_features_list, dim=0)
            encoded_image_features = torch.split(encoded_image_features, split_sizes)

            # image_features[i] : [num_frames, num_tokens, dim] for video
            # or [num_images, num_tokens, dim] for multi-image
            image_features = []


            # cur_mm_spatial_pool_stride = getattr(self.config, "mm_spatial_pool_stride", 1)

            for idx, image_feat in enumerate(encoded_image_features):
                if idx in video_idx_in_batch:
                    # print(image_feat.shape)  729
                    pooled_feat = self.get_2dPool(image_feat)
                else:
                    pooled_feat = image_feat
                image_features.append(pooled_feat)

            # print(image_features[0].shape) 
            mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
            image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")
            mm_newline_position = getattr(self.config, "mm_newline_position", "one_token")
            # import pdb
            # pdb.set_trace()
            if mm_patch_merge_type == "flat":
                image_features = [x.flatten(0, 1) for x in image_features]

            elif mm_patch_merge_type.startswith("spatial"):
                new_image_features = []
                # print('image_features',len(image_features),image_features[0].shape)
                # exit(0)
                for image_idx, image_feature in enumerate(image_features):
                    # -----------------------------------------------------
                    # video branch: streaming frame-by-frame FOSS compression
                    # image_feature shape: [num_frames, num_tokens, dim]
                    # -----------------------------------------------------
                    # print()
                    if image_idx in video_idx_in_batch:
                        mm_newline_position = 'one_token'
                        # compressed_video_feature = self.stream_compress_video_features(image_feature)
                     
                        compressed_video_feature, frame_token_counts = self.stream_compress_video_features(
                            image_feature,
                            verbose=True
                        )
                        

                        # print('new_image_features',len(new_image_features),new_image_features[0].shape)
                        #new_image_features 1 torch.Size([6000, 3584])
                        # compressed_video_feature = compressed_video_feature.flatten(0, 1)
                        if 'unpad' in mm_patch_merge_type:
                            compressed_video_feature = torch.cat((
                                compressed_video_feature,
                                self.model.image_newline[None].to(image_feature.device)
                            ), dim=0)
                        print('com', compressed_video_feature.shape)
                        new_image_features.append(compressed_video_feature)  
                        # exit(0)
                    # -----------------------------------------------------
                    # multi-patch / multi-image branch (keep original logic)
                    # -----------------------------------------------------
                    elif image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]

                        matched_anyres_max_num_patches = None
                        if "anyres_max" in image_aspect_ratio:
                            matched_anyres_max_num_patches = re.match(
                                r"anyres_max_(\d+)", image_aspect_ratio
                            )
                            if matched_anyres_max_num_patches:
                                max_num_patches = int(matched_anyres_max_num_patches.group(1))

                        if image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
                            if hasattr(self.get_vision_tower(), "image_size"):
                                vision_tower_image_size = self.get_vision_tower().image_size
                            else:
                                raise ValueError("vision_tower_image_size is not found in the vision tower.")
                            try:
                                num_patch_width, num_patch_height = get_anyres_image_grid_shape(
                                    image_sizes[image_idx],
                                    self.config.image_grid_pinpoints,
                                    vision_tower_image_size,
                                )
                            except Exception as e:
                                rank0_print(f"Error: {e}")
                                num_patch_width, num_patch_height = 2, 2
                            image_feature = image_feature.view(
                                num_patch_height, num_patch_width, height, width, -1
                            )
                        else:
                            image_feature = image_feature.view(2, 2, height, width, -1)

                        if "maxpool2x2" in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = nn.functional.max_pool2d(image_feature, 2)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)

                        elif (
                            "unpad" in mm_patch_merge_type
                            and "anyres_max" in image_aspect_ratio
                            and matched_anyres_max_num_patches
                        ):
                            unit = image_feature.shape[2]
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            c, h, w = image_feature.shape
                            times = math.sqrt(h * w / (max_num_patches * unit**2))
                            if times > 1.1:
                                image_feature = image_feature[None]
                                image_feature = nn.functional.interpolate(
                                    image_feature,
                                    [int(h // times), int(w // times)],
                                    mode="bilinear",
                                )[0]
                            image_feature = torch.cat(
                                (
                                    image_feature,
                                    self.get_model().image_newline[:, None, None].expand(
                                        *image_feature.shape[:-1], 1
                                    ).to(image_feature.device),
                                ),
                                dim=-1,
                            )
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)

                        elif "unpad" in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat(
                                (
                                    image_feature,
                                    self.get_model().image_newline[:, None, None].expand(
                                        *image_feature.shape[:-1], 1
                                    ).to(image_feature.device),
                                ),
                                dim=-1,
                            )
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)

                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)

                        if "nobase" not in mm_patch_merge_type:
                            image_feature = torch.cat((base_image_feature, image_feature), dim=0)

                        new_image_features.append(image_feature)

                    # -----------------------------------------------------
                    # single image branch
                    # -----------------------------------------------------
                    else:
                        image_feature = image_feature[0]
                        if "unpad" in mm_patch_merge_type:
                            image_feature = torch.cat(
                                (image_feature, self.get_model().image_newline[None].to(image_feature.device)),
                                dim=0,
                            )
                        new_image_features.append(image_feature)

                image_features = new_image_features

            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")

        else:
            image_features = self.encode_images(images)

        if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(
            self.config, "mm_use_im_start_end", False
        ):
            raise NotImplementedError

        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()

        if position_ids is None:
            position_ids = torch.arange(
                0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
            )

        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        input_ids = [
            cur_input_ids[cur_attention_mask]
            for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)
        ]
        labels = [
            cur_labels[cur_attention_mask]
            for cur_labels, cur_attention_mask in zip(labels, attention_mask)
        ]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0

        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()

            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat(
                    [cur_input_embeds_1, cur_image_features[0:0]], dim=0
                )
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = (
                [-1]
                + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
                + [cur_input_ids.shape[0]]
            )

            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []

            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(
                    cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]]
                )
                cur_labels_noim.append(
                    cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]]
                )

            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)

            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])

                if i < num_images:
                    try:
                        cur_image_features = image_features[cur_image_idx]
                    except IndexError:
                        cur_image_features = image_features[cur_image_idx - 1]

                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(
                        torch.full(
                            (cur_image_features.shape[0],),
                            IGNORE_INDEX,
                            device=cur_labels.device,
                            dtype=cur_labels.dtype,
                        )
                    )

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)

        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full(
            (batch_size, max_len),
            IGNORE_INDEX,
            dtype=new_labels[0].dtype,
            device=new_labels[0].device,
        )
        attention_mask = torch.zeros(
            (batch_size, max_len),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        position_ids = torch.zeros(
            (batch_size, max_len),
            dtype=position_ids.dtype,
            device=position_ids.device,
        )

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]

            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                            cur_new_embed,
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                    )
            else:
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            cur_new_embed,
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                    )

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        if getattr(self.config, "use_pos_skipping", False) and self.training:
            position_ids = torch.arange(new_input_embeds.size(1), device=new_input_embeds.device).unsqueeze(0)
            split_position = random.randint(0, new_input_embeds.size(1))
            left_add = random.randint(0, self.config.pos_skipping_range)
            right_add = random.randint(left_add, self.config.pos_skipping_range)
            position_ids[:, :split_position] += left_add
            position_ids[:, split_position:] += right_add

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN],
                special_tokens=True,
            )
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(
                    model_args.pretrain_mm_mlp_adapter,
                    map_location="cpu",
                )
                embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]
                assert num_new_tokens == 2

                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(
                        f"Unexpected embed_tokens_weight shape. "
                        f"Pretrained: {embed_tokens_weight.shape}. "
                        f"Current: {input_embeddings.shape}. "
                        f"Number of new tokens: {num_new_tokens}."
                    )

        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False