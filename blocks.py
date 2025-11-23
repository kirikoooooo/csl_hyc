import time

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from collections import OrderedDict

from utils import generate_binomial_mask
import logs


def _calc_parameter_memory_bytes(module: nn.Module) -> int:
    total = 0
    for param in module.parameters(recurse=True):
        total += param.numel() * param.element_size()
    return total


class MinEuclideanDistBlock(nn.Module):

    def __init__(self, shapelets_size, num_shapelets, in_channels=1, to_cuda=True, checkpoint=False):
        super(MinEuclideanDistBlock, self).__init__()
        self.to_cuda = to_cuda
        self.num_shapelets = num_shapelets
        self.shapelets_size = shapelets_size
        self.in_channels = in_channels
        self.checkpoint = checkpoint
        self._first_forward_skipped = False  # 用于标记是否已经跳过第一次
        # if not registered as parameter, the optimizer will not be able to see the parameters
        shapelets = torch.randn(self.in_channels, self.num_shapelets, self.shapelets_size, requires_grad=True,
                                dtype=torch.float)
        if self.to_cuda:
            shapelets = shapelets.cuda()
        self.shapelets = nn.Parameter(shapelets).contiguous()
        # otherwise gradients will not be backpropagated
        self.shapelets.retain_grad()

    def forward(self, x, masking=False):

        current_backward_mem = torch.cuda.max_memory_allocated()
        logs.global_backward_peak_mem = max(logs.global_backward_peak_mem, current_backward_mem)

        # ✅ 只记录一次 forward 前的显存（首次模块 forward）
        if logs.global_pre_forward_mem is None:
            logs.global_pre_forward_mem = torch.cuda.memory_allocated()

        start_time = time.time()
        logs.epoch_max_allocated = max(logs.epoch_max_allocated, torch.cuda.max_memory_allocated())
        torch.cuda.reset_peak_memory_stats()
        pre_mem = torch.cuda.memory_allocated()

        x = x.unfold(2, self.shapelets_size, 1).contiguous()
        if logs.cdist_euclidean_mem is None:
            torch.cuda.empty_cache()
            logs.epoch_max_allocated = max(logs.epoch_max_allocated, torch.cuda.max_memory_allocated())
            torch.cuda.reset_peak_memory_stats()
            begin = torch.cuda.memory_allocated()
            x = torch.cdist(x, self.shapelets, p=2, compute_mode='use_mm_for_euclid_dist')
            torch.cuda.synchronize()
            end = torch.cuda.memory_allocated()
            delta = end - begin

            if delta > 0:
                logs.cdist_euclidean_mem = delta
                print(f"[CDIST] 收集成功，torch.cdist 显存消耗: {delta} ")

        else:
            x = torch.cdist(x, self.shapelets, p=2, compute_mode='use_mm_for_euclid_dist')

        x = torch.sum(x, dim=1, keepdim=True).transpose(2, 3)
        x, _ = torch.min(x, 3)

        torch.cuda.synchronize()
        end_time = time.time()

        duration = end_time - start_time
        length = self.shapelets_size

        if logs.skip_first_forward and not self._first_forward_skipped:
            self._first_forward_skipped = True
            return x

        if length not in logs.block_forward_stats_by_type["euclidean"]:
            logs.block_forward_stats_by_type["euclidean"][length] = {
                "forward_total_time": 0.0,
                "forward_calls": 1,
                "peak_mem": None,
            }
        stats = logs.block_forward_stats_by_type["euclidean"][length]
        if stats["forward_calls"] < logs.global_iter_count:
            stats["forward_total_time"] += duration
            stats["forward_calls"] += 1
        if stats["peak_mem"] is None:
            stats["peak_mem"] = torch.cuda.max_memory_allocated() - pre_mem
            stats["final_mem"] = 0

        return x


class MaxCosineSimilarityBlock(nn.Module):

    def __init__(self, shapelets_size, num_shapelets, in_channels=1, to_cuda=True):
        super(MaxCosineSimilarityBlock, self).__init__()
        self.to_cuda = to_cuda
        self.num_shapelets = num_shapelets
        self._first_forward_skipped = False
        self.shapelets_size = shapelets_size
        self.in_channels = in_channels
        self.relu = nn.ReLU()
        # if not registered as parameter, the optimizer will not be able to see the parameters
        shapelets = torch.randn(self.in_channels, self.num_shapelets, self.shapelets_size, requires_grad=True,
                                dtype=torch.float)
        if self.to_cuda:
            shapelets = shapelets.cuda()
        self.shapelets = nn.Parameter(shapelets).contiguous()
        # otherwise gradients will not be backpropagated
        self.shapelets.retain_grad()

    def forward(self, x, masking=False):
        start_time = time.time()
        logs.epoch_max_allocated = max(logs.epoch_max_allocated, torch.cuda.max_memory_allocated())
        torch.cuda.reset_peak_memory_stats()
        pre_mem = torch.cuda.memory_allocated()
        """
        n_dims = x.shape[1]
        shapelets_norm = self.shapelets / self.shapelets.norm(p=2, dim=2, keepdim=True).clamp(min=1e-8)
        shapelets_norm = shapelets_norm.transpose(1, 2).half()
        out = torch.zeros((x.shape[0],
                           1,
                           x.shape[2] - self.shapelets_size + 1,
                           self.num_shapelets),
                        dtype=torch.float)
        if self.to_cuda:
            out = out.cuda()
        for i_dim in range(n_dims):
            x_dim = x[:, i_dim : i_dim + 1, :].half()
            x_dim = x_dim.unfold(2, self.shapelets_size, 1).contiguous()
            x_dim = x_dim / x_dim.norm(p=2, dim=3, keepdim=True).clamp(min=1e-8)
            out += torch.matmul(x_dim, shapelets_norm[i_dim : i_dim + 1, :, :]).float()

        x = out.transpose(2, 3) / n_dims
        """


        # prev_allocated = torch.cuda.memory_allocated()
        # unfold time series to emulate sliding window
        x = x.unfold(2, self.shapelets_size, 1).contiguous()

        # normalize with l2 norm
        x = x / x.norm(p=2, dim=3, keepdim=True).clamp(min=1e-8)

        shapelets_norm = (self.shapelets / self.shapelets.norm(p=2, dim=2, keepdim=True).clamp(min=1e-8))

        # calculate cosine similarity via dot product on already normalized ts and shapelets
        x = torch.matmul(x, shapelets_norm.transpose(1, 2))
        # add up the distances of the channels in case of
        # multivariate time series
        # Corresponds to the approach 1 and 3 here: https://stats.stackexchange.com/questions/184977/multivariate-time-series-euclidean-distance
        n_dims = x.shape[1]

        x = torch.sum(x, dim=1, keepdim=True).transpose(2, 3) / n_dims

        # ignore negative distances
        x = self.relu(x)

        x, _ = torch.max(x, 3)

        torch.cuda.synchronize()
        end_time = time.time()

        duration = end_time - start_time
        length = self.shapelets_size

        if logs.skip_first_forward and not self._first_forward_skipped:
            self._first_forward_skipped = True
            return x

        if length not in logs.block_forward_stats_by_type["cosine"]:
            logs.block_forward_stats_by_type["cosine"][length] = {
                "forward_total_time": 0.0,
                "forward_calls": 1,
                "peak_mem": None,
            }

        stats = logs.block_forward_stats_by_type["cosine"][length]
        if stats["forward_calls"] < logs.global_iter_count :
            stats["forward_total_time"] += duration
            stats["forward_calls"] += 1

        if stats["peak_mem"] is None:
            stats["peak_mem"] = torch.cuda.max_memory_allocated() - pre_mem
            stats["final_mem"] = 0
        return x


class MaxCrossCorrelationBlock(nn.Module):

    def __init__(self, shapelets_size, num_shapelets, in_channels=1, to_cuda=True):
        super(MaxCrossCorrelationBlock, self).__init__()
        self.shapelets = nn.Conv1d(in_channels, num_shapelets, kernel_size=shapelets_size)
        self.num_shapelets = num_shapelets
        self.shapelets_size = shapelets_size
        self._first_forward_skipped = False
        self.to_cuda = to_cuda
        if self.to_cuda:
            self.cuda()

    def forward(self, x, masking=False):
        start_time = time.time()

        x = self.shapelets(x)
        if masking:
            mask = generate_binomial_mask(x.shape)
            x *= mask
        x, _ = torch.max(x, 2, keepdim=True)
        torch.cuda.synchronize()
        end_time = time.time()

        duration = end_time - start_time
        length = self.shapelets_size

        if logs.skip_first_forward and not self._first_forward_skipped:
            self._first_forward_skipped = True
            return x.transpose(2, 1)

        if length not in logs.block_forward_stats_by_type['cross']:
            logs.block_forward_stats_by_type['cross'][length] = {
                "forward_total_time": 0.0,
                "forward_calls": 1,
            }

        stats = logs.block_forward_stats_by_type['cross'][length]
        if stats["forward_calls"] < logs.global_iter_count :
            stats["forward_total_time"] += duration
            stats["forward_calls"] += 1

        return x.transpose(2, 1)


class ShapeletsDistBlocks(nn.Module):

    def __init__(self, shapelets_size_and_len, in_channels=1, dist_measure='euclidean', to_cuda=True, checkpoint=False):
        super(ShapeletsDistBlocks, self).__init__()
        self.checkpoint = checkpoint
        self.to_cuda = to_cuda
        self.shapelets_size_and_len = OrderedDict(sorted(shapelets_size_and_len.items(), key=lambda x: x[0]))
        self.in_channels = in_channels
        self.dist_measure = dist_measure
        shapelet_lengths = list(shapelets_size_and_len.keys())

        if dist_measure == 'euclidean':
            self.euclidean_checkpoint_shapelet_lengths = shapelet_lengths
        elif dist_measure == 'cosine':
            self.cosine_checkpoint_shapelet_lengths = shapelet_lengths

        if dist_measure == 'euclidean':
            self.blocks = nn.ModuleList(
                [MinEuclideanDistBlock(shapelets_size=shapelets_size, num_shapelets=num_shapelets,
                                       in_channels=in_channels, to_cuda=self.to_cuda, checkpoint=self.checkpoint)
                 for shapelets_size, num_shapelets in self.shapelets_size_and_len.items()])
        elif dist_measure == 'cross-correlation':
            self.blocks = nn.ModuleList(
                [MaxCrossCorrelationBlock(shapelets_size=shapelets_size, num_shapelets=num_shapelets,
                                          in_channels=in_channels, to_cuda=self.to_cuda)
                 for shapelets_size, num_shapelets in self.shapelets_size_and_len.items()])
        elif dist_measure == 'cosine':
            self.blocks = nn.ModuleList(
                [MaxCosineSimilarityBlock(shapelets_size=shapelets_size, num_shapelets=num_shapelets,
                                          in_channels=in_channels, to_cuda=self.to_cuda)
                 for shapelets_size, num_shapelets in self.shapelets_size_and_len.items()])
        elif dist_measure == 'mix':
            module_list = []
            for shapelets_size, num_shapelets in self.shapelets_size_and_len.items():
                module_list.append(
                    MinEuclideanDistBlock(shapelets_size=shapelets_size, num_shapelets=num_shapelets // 3,
                                          in_channels=in_channels, to_cuda=self.to_cuda))
                module_list.append(
                    MaxCosineSimilarityBlock(shapelets_size=shapelets_size, num_shapelets=num_shapelets // 3,
                                             in_channels=in_channels, to_cuda=self.to_cuda))
                module_list.append(MaxCrossCorrelationBlock(shapelets_size=shapelets_size,
                                                            num_shapelets=num_shapelets - 2 * num_shapelets // 3,
                                                            in_channels=in_channels, to_cuda=self.to_cuda))
            self.blocks = nn.ModuleList(module_list)

        else:
            raise ValueError("dist_measure must be either of 'euclidean', 'cross-correlation', 'cosine'")

        self._record_parameter_memory()

    def forward(self, x, masking=False):

        out = torch.tensor([], dtype=torch.float).cuda() if self.to_cuda else torch.tensor([], dtype=torch.float)
        # for block in self.blocks:
        #
        #     if self.checkpoint and self.dist_measure != 'cross-correlation' :
        #         out = torch.cat((out, checkpoint(block, x, masking, use_reentrant=False)), dim=2)
        #
        #     else:
        #         out = torch.cat((out, block(x, masking)), dim=2)
        for i, (shapelets_size, _) in enumerate(self.shapelets_size_and_len.items()):
            block = self.blocks[i]

            if self.checkpoint and self.dist_measure == 'euclidean' and shapelets_size in logs.euclidean_checkpoint_shapelet_lengths:
                out = torch.cat((out, checkpoint(block, x, masking, use_reentrant=False)), dim=2)
            elif self.checkpoint and self.dist_measure == 'cosine' and shapelets_size in logs.cosine_checkpoint_shapelet_lengths:
                out = torch.cat((out, checkpoint(block, x, masking, use_reentrant=False)), dim=2)
            else:
                out = torch.cat((out, block(x, masking)), dim=2)

        return out

    def _record_parameter_memory(self):
        lengths = list(self.shapelets_size_and_len.keys())
        if not lengths:
            return

        for idx, block in enumerate(self.blocks):
            if isinstance(block, MinEuclideanDistBlock):
                mem_key = "euclidean"
            elif isinstance(block, MaxCosineSimilarityBlock):
                mem_key = "cosine"
            elif isinstance(block, MaxCrossCorrelationBlock):
                mem_key = "cross"
            else:
                continue

            if self.dist_measure == 'mix':
                length_idx = min(idx // 3, len(lengths) - 1)
            else:
                length_idx = min(idx, len(lengths) - 1)

            shapelets_size = lengths[length_idx]
            memory_bytes = _calc_parameter_memory_bytes(block)
            logs.block_param_memory_by_type[mem_key][shapelets_size] = memory_bytes


class LearningShapeletsModel(nn.Module):

    def __init__(self, shapelets_size_and_len, in_channels=1, num_classes=2, dist_measure='euclidean',
                 to_cuda=True, checkpoint=False):
        super(LearningShapeletsModel, self).__init__()

        self.to_cuda = to_cuda
        self.checkpoint = checkpoint
        self.shapelets_size_and_len = shapelets_size_and_len
        self.num_shapelets = sum(shapelets_size_and_len.values())
        self.shapelets_blocks = ShapeletsDistBlocks(in_channels=in_channels,
                                                    shapelets_size_and_len=shapelets_size_and_len,
                                                    dist_measure=dist_measure, to_cuda=to_cuda, checkpoint=checkpoint)
        self.linear = nn.Linear(self.num_shapelets, num_classes)

        self.projection = nn.Sequential(nn.BatchNorm1d(num_features=self.num_shapelets),
                                        # nn.Linear(self.model.num_shapelets, 256),
                                        # nn.ReLU(),
                                        # nn.Linear(self.num_shapelets, 128)
                                        )

        self.projection2 = nn.Sequential(nn.Linear(self.num_shapelets, 256),
                                         nn.ReLU(),
                                         nn.Linear(256, 128))

        if self.to_cuda:
            self.cuda()

    def forward(self, x, optimize='acc', masking=False):

        x = self.shapelets_blocks(x, masking)

        x = torch.squeeze(x, 1)

        # test torch.cat
        # x = torch.cat((x[:, :x.shape[1] // 2], x[:, x.shape[1] // 2:]), dim=1)

        x = self.projection(x)

        if optimize == 'acc':
            x = self.linear(x)

        return x


class LearningShapeletsModelMixDistances(nn.Module):

    def __init__(self, shapelets_size_and_len, in_channels=1, num_classes=2, dist_measure='mix',
                 to_cuda=True, checkpoint=False):
        super(LearningShapeletsModelMixDistances, self).__init__()

        self.checkpoint = checkpoint
        self.to_cuda = to_cuda
        self.shapelets_size_and_len = shapelets_size_and_len
        self.num_shapelets = sum(shapelets_size_and_len.values())

        self.shapelets_euclidean = ShapeletsDistBlocks(in_channels=in_channels,
                                                       shapelets_size_and_len={item[0]: item[1] // 3 for item in
                                                                               shapelets_size_and_len.items()},
                                                       dist_measure='euclidean', to_cuda=to_cuda,
                                                       checkpoint=self.checkpoint)

        self.shapelets_cosine = ShapeletsDistBlocks(in_channels=in_channels,
                                                    shapelets_size_and_len={item[0]: item[1] // 3 for item in
                                                                            shapelets_size_and_len.items()},
                                                    dist_measure='cosine', to_cuda=to_cuda, checkpoint=checkpoint)

        self.shapelets_cross_correlation = ShapeletsDistBlocks(in_channels=in_channels,
                                                               shapelets_size_and_len={
                                                                   item[0]: item[1] - 2 * (item[1] // 3) for item in
                                                                   shapelets_size_and_len.items()},
                                                               dist_measure='cross-correlation', to_cuda=to_cuda,
                                                               checkpoint=checkpoint)

        self.linear = nn.Linear(self.num_shapelets, num_classes)

        self.projection = nn.Sequential(nn.BatchNorm1d(num_features=self.num_shapelets),
                                        # nn.Linear(self.model.num_shapelets, 256),
                                        # nn.ReLU(),
                                        # nn.Linear(self.num_shapelets, 128)
                                        )

        self.bn1 = nn.BatchNorm1d(num_features=sum(num // 3 for num in self.shapelets_size_and_len.values()))
        self.bn2 = nn.BatchNorm1d(num_features=sum(num // 3 for num in self.shapelets_size_and_len.values()))
        self.bn3 = nn.BatchNorm1d(
            num_features=sum(num - 2 * (num // 3) for num in self.shapelets_size_and_len.values()))

        self.projection2 = nn.Sequential(nn.Linear(self.num_shapelets, 256),
                                         nn.ReLU(),
                                         nn.Linear(256, 128))

        if self.to_cuda:
            self.cuda()

    def forward(self, x, optimize='acc', masking=False):

        # start_time = time.time()

        n_samples = x.shape[0]
        num_lengths = len(self.shapelets_size_and_len)

        out = torch.tensor([], dtype=torch.float).cuda() if self.to_cuda else torch.tensor([], dtype=torch.float)

        x_out = self.shapelets_euclidean(x, masking)
        x_out = torch.squeeze(x_out, 1)
        # x_out = torch.nn.functional.normalize(x_out, dim=1)
        x_out = self.bn1(x_out)
        x_out = x_out.reshape(n_samples, num_lengths, -1)
        # print(x_out.shape)
        out = torch.cat((out, x_out), dim=2)

        x_out = self.shapelets_cosine(x, masking)
        x_out = torch.squeeze(x_out, 1)
        # x_out = torch.nn.functional.normalize(x_out, dim=1)
        x_out = self.bn2(x_out)
        x_out = x_out.reshape(n_samples, num_lengths, -1)
        # print(x_out.shape)
        out = torch.cat((out, x_out), dim=2)

        x_out = self.shapelets_cross_correlation(x, masking)
        x_out = torch.squeeze(x_out, 1)
        # x_out = torch.nn.functional.normalize(x_out, dim=1)
        x_out = self.bn3(x_out)
        x_out = x_out.reshape(n_samples, num_lengths, -1)
        # print(x_out.shape)
        out = torch.cat((out, x_out), dim=2)

        out = out.reshape(n_samples, -1)

        # print(out.shape)
        # out = self.projection(out)

        if optimize == 'acc':
            out = self.linear(out)
        # end_time = time.time()
        # print(f"{end_time - start_time:.6f} ")

        return out


