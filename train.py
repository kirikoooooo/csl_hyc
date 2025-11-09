import os
import time
import warnings
import logging
import numpy as np
import torch
from manager import Manager, cast_forward
import tsaug
from torch import nn

from tqdm import tqdm

from blocks import LearningShapeletsModel, LearningShapeletsModelMixDistances
import logs
from datetime import datetime
from sko.GA import GA
from KGA import *
from solver.checkmate_solver import *
from solver.monet_solver import *
def print_cuda_memory(tag=""):
    allocated = torch.cuda.memory_allocated() / 1024 ** 2  # MB
    reserved = torch.cuda.memory_reserved() / 1024 ** 2
    max_allocated = torch.cuda.max_memory_allocated() / 1024 ** 2
    max_reserved = torch.cuda.max_memory_reserved() / 1024 ** 2
    print(
        f"[{tag}] Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB | Max Allocated: {max_allocated:.2f} MB | Max Reserved: {max_reserved:.2f} MB")

def trace_handler(prof: torch.profiler.profile):
   # 获取时间用于文件命名
   timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
   file_name = f"visual_mem_{timestamp}"

   # 导出tracing格式的profiling
   prof.export_chrome_trace(f"{file_name}.json")

   # 导出mem消耗可视化数据
   prof.export_memory_timeline(f"{file_name}.html")
   print("已经调用tracehandler")


class LearningShapeletsCL:
    """
    Parameters
    ----------
    shapelets_size_and_len : dict(int:int)
        The keys are the length of the shapelets and the values the number of shapelets of
        a given length, e.g. {40: 4, 80: 4} learns 4 shapelets of length 40 and 4 shapelets of
        length 80.
    loss_func : torch.nn
        the loss function
    in_channels : int
        the number of input channels of the dataset
    num_classes : int
        the number of output classes.
    dist_measure: `euclidean`, `cross-correlation`, or `cosine`
        the distance measure to use to compute the distances between the shapelets.
      and the time series.
    verbose : bool
        monitors training loss if set to true.
    to_cuda : bool
        if true loads everything to the GPU
    """

    def __init__(self, shapelets_size_and_len, loss_func, in_channels=1, num_classes=2,
                 dist_measure='euclidean', verbose=0, to_cuda=True, l3=0.0, l4=0.0, T=0.1, alpha=0.0, is_ddp=False,
                 checkpoint=False, seed=None, dynamic_checkpoint=False,args=None):
        self.args = args
        self.memory_buffer = 0
        self.memory_threshold = 4
        self.min_input_size = 21
        self.max_input_size = 512
        self.static_checkpoint = None
        self.warmup_iters = 3
        self.dynamic_checkpoint = dynamic_checkpoint
        # memory_threshold = self.memory_threshold
        # if memory_threshold > 3:
        #     torch.cuda.set_per_process_memory_fraction(
        #         memory_threshold * (1024 ** 3) / torch.cuda.get_device_properties(0).total_memory)

        self.is_ddp = is_ddp
        self.checkpoint = checkpoint
        self.seed = seed
        if dist_measure == 'mix':
            self.model = LearningShapeletsModelMixDistances(shapelets_size_and_len=shapelets_size_and_len,
                                                            in_channels=in_channels, num_classes=num_classes,
                                                            dist_measure=dist_measure,
                                                            to_cuda=to_cuda, checkpoint=self.checkpoint)
        else:
            self.model = LearningShapeletsModel(shapelets_size_and_len=shapelets_size_and_len,
                                                in_channels=in_channels, num_classes=num_classes,
                                                dist_measure=dist_measure,
                                                to_cuda=to_cuda, checkpoint=checkpoint)
        self.to_cuda = to_cuda
        if self.to_cuda:
            self.model.cuda()

        self.shapelets_size_and_len = shapelets_size_and_len
        self.num_shapelets = sum(shapelets_size_and_len.values())
        self.loss_func = loss_func
        self.verbose = verbose
        self.optimizer = None
        self.scheduler = None

        self.current_epoch = None  # 用于 update_CL 中判断是否记录反向时间

        self.l3 = l3
        self.l4 = l4
        self.alpha = alpha
        self.use_regularizer = False

        if self.dynamic_checkpoint:
            warmup_iters = self.warmup_iters
            self.dc_manager = Manager(warmup_iters=warmup_iters)
            self.dc_manager.set_max_memory_GB(memory_threshold=self.memory_threshold - self.memory_buffer)
            self.dc_manager.static_strategy = self.static_checkpoint
            self.dc_manager.max_input = self.max_input_size
            self.dc_manager.min_input = self.min_input_size
            self.dc_manager.shapelets_size_and_len = self.shapelets_size_and_len
            cast_forward(self.model, "0", self.dc_manager, self.shapelets_size_and_len)

        self.batch_size = 0
        self.column = 0
        self.length = 0
        # self.mask = MaskBlock(p=0.5)

        # self.bn = nn.BatchNorm1d(num_features=self.model.num_shapelets)
        # self.relu = nn.ReLU()

        # if self.to_cuda:
        #    self.mask.cuda()
        #    self.bn.cuda()
        #    self.relu.cuda()

        self.T = T

        # self.r = 64

        # self.num_clusters = [0.01, 0.02, 0.04]
        # 确保日志目录存在
        log_dir = args.logdir
        os.makedirs(log_dir, exist_ok=True)

        # 构造日志文件路径
        log_file = os.path.join(log_dir, f"{args.dataset}{args.de}.log")

        # 全局 logger 配置
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def update(self, x, y):
        y_hat = self.model(x)
        loss = self.loss_func(y_hat, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def update_CL(self, x, C_accu_q, c_normalising_factor_q, C_accu_k, c_normalising_factor_k):

        augmentation_list = ['AddNoise(seed=np.random.randint(2 ** 32 - 1, dtype=np.int64))',
                             'Crop(int(0.9 * ts_l), seed=np.random.randint(2 ** 32 - 1, dtype=np.int64))',
                             'Pool(seed=np.random.randint(2 ** 32 - 1, dtype=np.int64))',
                             'Quantize(seed=np.random.randint(2 ** 32 - 1, dtype=np.int64))',
                             'TimeWarp(seed=np.random.randint(2 ** 32 - 1, dtype=np.int64))'
                             ]
        # augmentation_list = ['AddNoise()', 'Pool()', 'Quantize()', 'TimeWarp()']

        ts_l = x.size(2)

        aug1 = np.random.choice(augmentation_list, 1, replace=False)

        x_q = x.transpose(1, 2).cpu().numpy()
        for aug in aug1:
            x_q = eval('tsaug.' + aug + '.augment(x_q)')
        x_q = torch.from_numpy(x_q).float()
        x_q = x_q.transpose(1, 2)

        if self.to_cuda:
            x_q = x_q.cuda()

        aug2 = np.random.choice(augmentation_list, 1, replace=False)
        while (aug2 == aug1).all():
            aug2 = np.random.choice(augmentation_list, 1, replace=False)

        x_k = x.transpose(1, 2).cpu().numpy()
        for aug in aug2:
            x_k = eval('tsaug.' + aug + '.augment(x_k)')
        x_k = torch.from_numpy(x_k).float()
        x_k = x_k.transpose(1, 2)
        if self.to_cuda:
            x_k = x_k.cuda()

        num_shapelet_lengths = len(self.shapelets_size_and_len)

        num_shapelet_per_length = self.num_shapelets // num_shapelet_lengths

        with torch.autograd.set_detect_anomaly(True):


            q = self.model(x_q, optimize=None, masking=False)
            # print("-----------------------------------------------")
            k = self.model(x_k, optimize=None, masking=False)
            # print("-----------------------------------------------")

            torch.cuda.synchronize()
            start = time.time()
            q = nn.functional.normalize(q, dim=1)
            k = nn.functional.normalize(k, dim=1)

            logits = torch.einsum('nc,ck->nk', [q, k.t()])
            logits /= self.T
            labels = torch.arange(q.shape[0], dtype=torch.long)

            if self.to_cuda:
                labels = labels.cuda()

            loss = self.loss_func(logits, labels)

            q_sum = None
            q_square_sum = None

            k_sum = None
            k_square_sum = None

            loss_sdl = 0
            c_normalising_factor_q = self.alpha * c_normalising_factor_q + 1

            c_normalising_factor_k = self.alpha * c_normalising_factor_k + 1

            for length_i in range(num_shapelet_lengths):
                qi = q[:, length_i * num_shapelet_per_length: (length_i + 1) * num_shapelet_per_length]
                ki = k[:, length_i * num_shapelet_per_length: (length_i + 1) * num_shapelet_per_length]

                logits = torch.einsum('nc,ck->nk',
                                      [nn.functional.normalize(qi, dim=1), nn.functional.normalize(ki, dim=1).t()])
                logits /= self.T
                loss += self.loss_func(logits, labels)

                if q_sum == None:
                    q_sum = qi
                    q_square_sum = qi * qi
                else:
                    q_sum = q_sum + qi
                    q_square_sum = q_square_sum + qi * qi

                C_mini_q = torch.matmul(qi.t(), qi) / (qi.shape[0] - 1)
                C_accu_t_q = self.alpha * C_accu_q[length_i] + C_mini_q
                C_appx_q = C_accu_t_q / c_normalising_factor_q
                loss_sdl += torch.norm(
                    C_appx_q.flatten()[:-1].view(C_appx_q.shape[0] - 1, C_appx_q.shape[0] + 1)[:, 1:], 1).sum()
                # print(length_i)
                C_accu_q[length_i] = C_accu_t_q.detach()

                if k_sum == None:
                    k_sum = ki
                    k_square_sum = ki * ki
                else:
                    k_sum = k_sum + ki
                    k_square_sum = k_square_sum + ki * ki

                C_mini_k = torch.matmul(ki.t(), ki) / (ki.shape[0] - 1)
                C_accu_t_k = self.alpha * C_accu_k[length_i] + C_mini_k
                C_appx_k = C_accu_t_k / c_normalising_factor_k
                loss_sdl += torch.norm(
                    C_appx_k.flatten()[:-1].view(C_appx_k.shape[0] - 1, C_appx_k.shape[0] + 1)[:, 1:], 1).sum()
                # print(length_i)
                C_accu_k[length_i] = C_accu_t_k.detach()

            loss_cca = 0.5 * torch.sum(q_square_sum - q_sum * q_sum / num_shapelet_lengths) + 0.5 * torch.sum(
                k_square_sum - k_sum * k_sum / num_shapelet_lengths)

            loss += self.l3 * (loss_cca + self.l4 * loss_sdl)

            self.optimizer.zero_grad()
            #
            # if self.current_epoch == 0:  # 假设你设置了 self.current_epoch
            #     torch.cuda.synchronize()
            #     backward_start = time.time()
            #     loss.backward()
            #     torch.cuda.synchronize()
            #     backward_end = time.time()
            #     global global_backward_total_time
            #     global_backward_total_time += (backward_end - backward_start)
            # else:
            #     loss.backward()

            loss.backward()  # 0.2s



            self.optimizer.step()



        return [loss.item(), 0, loss_cca.item(), loss_sdl.item(),
                0], C_accu_q, c_normalising_factor_q, C_accu_k, c_normalising_factor_k

    def fine_tune(self, X, Y, epochs=1, batch_size=256, epoch_idx=-1):
        if self.optimizer is None:
            raise ValueError("No optimizer set. Please initialize an optimizer via set_optimizer(optim)")

        # convert to pytorch tensors and data set / loader for training
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float).contiguous()

        if not isinstance(Y, torch.Tensor):
            Y = torch.tensor(Y, dtype=torch.long).contiguous()

        train_ds = torch.utils.data.TensorDataset(X, Y)
        sampler = torch.utils.data.distributed.DistributedSampler(train_ds, shuffle=True) if self.is_ddp else None

        train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=(sampler is None),
                                               sampler=sampler, drop_last=True)

        # set model in train mode
        self.model.train()

        losses_ce = []
        progress_bar = tqdm(range(epochs), disable=False if self.verbose > 0 else True)

        for epoch in progress_bar:
            if self.is_ddp:
                sampler.set_epoch(epoch + epoch_idx * epochs)

            for (x, y) in train_dl:

                # check if training should be done with regularizer
                if self.to_cuda:
                    x = x.cuda()
                    y = y.cuda()
                    # print("Training data", idx, " on cuda ", torch.cuda.current_device())
                loss_ce = self.update(x, y)
                losses_ce.append(loss_ce)
        return losses_ce

    def train(self, X, epochs=1, batch_size=256, epoch_idx=-1):
        # pre = torch.cuda.memory_allocated()

        if self.optimizer is None:
            raise ValueError("No optimizer set. Please initialize an optimizer via set_optimizer(optim)")

        # convert to pytorch tensors and data set / loader for training
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float).contiguous()

        train_ds = torch.utils.data.TensorDataset(X, torch.arange(X.shape[0]))
        sampler = torch.utils.data.distributed.DistributedSampler(train_ds, shuffle=True) if self.is_ddp else None

        train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=(sampler is None),
                                               sampler=sampler, drop_last=True)
        # set model in train mode

        self.model.train()
        seq_length = X.shape[-1]
        # if self.dynamic_checkpoint:
        #     self.dc_manager.set_input_size(seq_length)
        # print(self.dc_manager.checkpoint_module)
        # print(self.dc_manager.non_checkpoint)

        losses_ce = []
        losses_dist = []
        losses_sim = []
        progress_bar = tqdm(range(epochs), disable=False if self.verbose > 0 else True)
        current_loss_ce = 0
        current_loss_dist = 0
        current_loss_sim = 0

        if epoch_idx == 0 and not logs.euclidean_checkpoint_shapelet_lengths:
            shapelet_lengths = list(self.shapelets_size_and_len.keys())
            logs.euclidean_checkpoint_shapelet_lengths = shapelet_lengths.copy()
            logs.cosine_checkpoint_shapelet_lengths = shapelet_lengths.copy()
            #print("📌 第一个 epoch：默认所有模块使用 checkpoint")
            self.logger.info("📌 第一个 epoch：默认所有模块使用 checkpoint")

        for epoch in progress_bar:

            self.current_epoch = epoch_idx
            if self.is_ddp:
                sampler.set_epoch(epoch + epoch_idx * epochs)

            if self.to_cuda:
                c_normalising_factor_q = torch.tensor([0], dtype=torch.float).cuda()
                C_accu_q = [torch.tensor([0], dtype=torch.float).cuda() for _ in
                            range(len(self.shapelets_size_and_len))]
                c_normalising_factor_k = torch.tensor([0], dtype=torch.float).cuda()
                C_accu_k = [torch.tensor([0], dtype=torch.float).cuda() for _ in
                            range(len(self.shapelets_size_and_len))]
            else:
                c_normalising_factor_q = torch.tensor([0], dtype=torch.float).cuda()
                C_accu_q = [torch.tensor([0], dtype=torch.float).cuda() for _ in
                            range(len(self.shapelets_size_and_len))]
                c_normalising_factor_k = torch.tensor([0], dtype=torch.float).cuda()
                C_accu_k = [torch.tensor([0], dtype=torch.float).cuda() for _ in
                            range(len(self.shapelets_size_and_len))]

            # 监控结果文件存放位置
            hander_path = './log/' + self.args.dataset + self.args.de
            #性能监控
            # with torch.profiler.profile(
            #         activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            #         schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
            #         #on_trace_ready=torch.profiler.tensorboard_trace_handler(hander_path),
            #         on_trace_ready=torch.profiler.tensorboard_trace_handler(hander_path, worker_name="epoch"+str(self.current_epoch)),
            #         record_shapes=True,
            #         profile_memory=True,
            #         with_stack=True
            # ) as prof:
            self.model.train()
            if self.current_epoch == 0:
                logs.global_iter_count = len(train_dl) * 2

            for (x, idx) in train_dl:
                self.batch_size = x.shape[0]
                self.column = x.shape[1]
                self.length = x.shape[2]
                #prof.step()
                # check if training should be done with regularizer
                if self.to_cuda:
                    x = x.cuda()


                if not self.use_regularizer:
                    torch.cuda.synchronize()
                    start_time = time.time()
                    current_loss_ce, C_accu_q, c_normalising_factor_q, C_accu_k, c_normalising_factor_k = self.update_CL(
                        x, C_accu_q, c_normalising_factor_q, C_accu_k, c_normalising_factor_k)
                    torch.cuda.synchronize()
                    if(self.current_epoch == 1):

                        logs.global_backward_total_time += time.time() - start_time
                    losses_ce.append(current_loss_ce)
                else:
                    pass

            if self.current_epoch== 1:
                self.estimate_backward_time_bias()
                # 估算时间
                start = time.time()
                if self.args.algo == "mimose":
                    self.plan_checkpoint_schedule_bucket()
                else:
                    self.plan_checkpoint_schedule(algo = self.args.algo)
                elapsed = time.time() - start
                self.logger.info(f"🕒 显存计划规划时间: {elapsed:.2f} 秒")



            if not self.use_regularizer:
                progress_bar.set_description(f"Loss: {current_loss_ce}")
            else:
                if self.l1 > 0.0 and self.l2 > 0.0:
                    progress_bar.set_description(f"Loss CE: {current_loss_ce}, Loss dist: {current_loss_dist}, "
                                                f"Loss sim: {current_loss_sim}")
                else:
                    progress_bar.set_description(f"Loss CE: {current_loss_ce}, Loss dist: {current_loss_dist}")
            if self.scheduler != None:
                self.scheduler.step()

            if self.dynamic_checkpoint:
                self.dc_manager.after_update()


            return losses_ce if not self.use_regularizer else (losses_ce, losses_dist, losses_sim) if self.l2 > 0.0 else (
                losses_ce, losses_dist)




    def plan_checkpoint_schedule(self,algo="monet"):
        from scipy.optimize import differential_evolution

        from logs import x, global_backward_b, global_pre_forward_mem, global_backward_peak_mem, cdist_euclidean_mem
        shapelet_lengths = list(self.shapelets_size_and_len.keys())
        s = 13

        # 记录运行时间和前向峰值显存
        T_euclidean = {}
        T_cosine = {}
        T_cross = {}
        peak_memory_e = {}
        peak_memory_c = {}

        for length in shapelet_lengths:
            stat = logs.block_forward_stats_by_type["euclidean"].get(length)
            if stat:
                T_euclidean[length] = stat["forward_total_time"] / (stat["forward_calls"] - 1)
                peak_memory_e[length] = stat["peak_mem"]

            stat = logs.block_forward_stats_by_type["cosine"].get(length)
            if stat:
                T_cosine[length] = stat["forward_total_time"] / (stat["forward_calls"] - 1)
                peak_memory_c[length] = stat["peak_mem"]

            stat = logs.block_forward_stats_by_type["cross"].get(length)
            if stat:
                T_cross[length] = stat["forward_total_time"] / (stat["forward_calls"] - 1)

        forward_peek = max(list(peak_memory_e.values()) + list(peak_memory_c.values()))
        # 显存公式
        # get_M_e(length): 计算长度为 `length` 的欧氏模块前向后需保留的最终内存。
        def get_M_e(length):
            return 4 * self.batch_size * self.column * (self.length- length + 1) * length + cdist_euclidean_mem

        def get_M_c(length):
            return 4 * (2* s * self.column+ self.column*s*length+self.batch_size*self.column * (self.length - length + 1) *length + self.batch_size* s+ self.batch_size * (self.length - length + 1) * s)

        # get_S_e(length): 获取长度为 `length` 的欧氏模块前向峰值内存 (统计数据)。
        def get_S_e(length):
            return peak_memory_e[length]

        def get_S_c(length):
            return peak_memory_c[length]

        # is_E(i): 判断索引 `i` 是否为欧氏模块 (1-8, 17-24)。
        def is_E(i):
            return 1 <= i <= 8 or 17 <= i <= 24

        def is_C(i):
            return 9 <= i <= 16 or 25 <= i <= 32

        # get_length(i): 根据索引 `i` (1-32) 获取对应的 Shapelet 长度
        def get_length(i):
            return shapelet_lengths[(i - 1) % 8]

        # get_peak_mem(i): 获取第 `i` 个阶段的峰值内存
        def get_peak_mem(i):
            return get_S_e(get_length(i)) if is_E(i) else get_S_c(get_length(i))

        # get_final_mem(i): 计算第 `i` 个前向阶段结束后保留的内存 (由策略 x[i] 决定)
        def get_final_mem(i):
            return x[i] * (get_M_e(get_length(i)) if is_E(i) else get_M_c(get_length(i)))

        # get_release(i): 计算第 `i` 个反向阶段释放的内存 (由策略 x[i] 决定)。
        def get_release(i):
            if x[i] == 0: return 0
            return get_M_e(get_length(i)) if is_E(i) else get_M_c(get_length(i))

        # get_backward_peak(i): 估算第 `i` 个模块反向时的峰值内存。
        def get_backward_peak(i):
            return global_backward_peak_mem / forward_peek * get_S_e(get_length(i)) if is_E(i) else get_S_c(get_length(i))

        # 计算前向和后向共65 个stage的显存峰值,这部分有点问题，应该是33
        def compute_K():
            K = [0] * 33
            cum_final = 0
            for t in range(1, 17):
                K[t] = global_pre_forward_mem  + get_peak_mem(t) + cum_final
                cum_final += get_final_mem(t)
            total_final = sum(get_final_mem(i) for i in range(1, 33))
            cum_release = 0
            for t in range(17, 33):
                i = 33 - t
                K[t] = get_backward_peak(i) + total_final - cum_release
                cum_release += get_release(i)
            return K

        # 根据给定的检查点策略 (x 数组，由 z_bin 决定)，计算并返回在整个模拟的前向和反向传播过程中预测会出现的最高显存峰值 。
        def compute_overall_peak():
            return max(compute_K()[1:])

        b = global_backward_b
        memory_limit = self.args.lim * 1024 ** 3 #byte

        # 目标函数 zbin 的值来自于遗传算法的遍历,BE CAUTIOUS 之前这里是67
        def objective(z_bin):
            # z_bin = [int(z >= 0.5) for z in z_float]
            for i in range(1, 17):
                x[i] = z_bin[i - 1]
                x[i + 16] = z_bin[i - 1]

            K_max = compute_overall_peak()

            x_euc, x_cos = z_bin[:8], z_bin[8:]
            T_ckp = T_nockp = 0
            for i in range(8):
                length = shapelet_lengths[i]
                if z_bin[i] == 0:
                    T_ckp += T_euclidean[length]
                else:
                    T_nockp += T_euclidean[length]
                if z_bin[i + 8] == 0:
                    T_ckp += T_cosine[length]
                else:
                    T_nockp += T_cosine[length]

            T_cross_total = sum(T_cross)
            total_time = 48 * (3 * T_ckp + 2 * T_nockp + 2 * T_cross_total) + b
            penalty = 1e10 * max(0, K_max - memory_limit)
            return total_time + penalty

        # monet algorithem
        if algo == 'monet' or algo == "checkmate":
            """
                cp.Minimize(total_time)
                constraints.append(K_max < memory_limit)

            """
            mem = {}
            for i in range(33):
                if i<16:
                    mem[i] = get_peak_mem(i)
                else:
                    mem[i] = get_backward_peak(i)
            if algo == "checkmate":
                result =checkmate(T_euclidean,T_cross,mem,memory_limit)  # checkmate 里面会改变z_best的值
            else:
                result = monet(T_euclidean,T_cross,mem,memory_limit)
            print(result)
            # 验证内存限制是否到达
            mem_list = list(mem.values())
            real_mem =sum(a * b for a, b in zip(result, mem_list))
            self.logger.info(f"规划前的显存GB：{float(memory_limit)/float(1024**3)}")
            self.logger.info(f"规划后的显存GB：{float(real_mem)/float(1024**3)}")
            # 写入z_best
            z_best = np.array(result[:16])


        # 差分进化
        if algo == "diff":
            result = differential_evolution(objective, bounds=[(0, 1)] * 16, strategy='best1bin', maxiter=300, disp=True)
            z_best = (np.array(result.x) >= 0.5).astype(int) #? 干啥的母鸡好像是差分

        #遗传算法
        if algo == "ga":
            ga = GA(
                func=objective,
                n_dim=16,
                size_pop=50,
                max_iter=100,
                prob_mut=0.1,
                lb=[0] * 16,
                ub=[1] * 16,
                precision=1,
            )
            z_best, best_y = ga.run()




        self.logger.info(f"✅ 最优策略：{z_best.tolist()}")
        all_lengths = list(self.shapelets_size_and_len.keys())
        logs.euclidean_checkpoint_shapelet_lengths.clear()
        logs.euclidean_checkpoint_shapelet_lengths.extend(
            [shapelet_lengths[i] for i in range(8) if z_best[i] == 0]
        )

        logs.cosine_checkpoint_shapelet_lengths.clear()
        logs.cosine_checkpoint_shapelet_lengths.extend(
            [shapelet_lengths[i] for i in range(8) if z_best[i + 8] == 0]
        )

        self.logger.info(f"📌 checkpoint 的欧氏长度:, {logs.euclidean_checkpoint_shapelet_lengths}")
        self.logger.info(f"📌 checkpoint 的余弦长度:, {logs.cosine_checkpoint_shapelet_lengths}")

        self.logger.info(f"💾 最终显存峰值：%.2f MB % {(compute_overall_peak() / 1024 ** 2)}")

    def estimate_backward_time_bias(self):
        """
        估算反向传播的基础时间开销 (bias)。

        该函数在第一个训练 Epoch (epoch == 1) 结束后调用。
        它利用在该 Epoch 中收集到的各模块前向传播时间统计数据，
        以及测量到的总反向传播时间，来拟合一个时间模型：
        总反向时间 ≈ 2 * (A + B) + b

        其中：
        - A: 所有欧氏距离和余弦距离模块估算的反向计算时间之和。
        - B: 所有交叉距离模块估算的反向计算时间之和。
        - b: 与前向计算时间无关的基础反向时间开销 (bias)。

        这个估算出的 `b` (存储为 `logs.global_backward_b`) 会被 `plan_checkpoint_schedule`
        中的目标函数用于更精确地预测不同检查点策略下的总运行时间。

        输入:
            无显式输入参数。该函数依赖于以下全局或类级别的状态：
            - `logs.block_forward_stats_by_type`: 包含各模块前向时间统计的字典。
            - `logs.global_backward_total_time`: 第一个 Epoch 测量到的总反向时间。
            - `self.current_epoch`: 用于判断是否在第一个 Epoch 后调用此函数。

        输出:
            无显式返回值。该函数将计算出的基础时间 `b` 存储到全局变量 `logs.global_backward_b` 中。
            同时会打印出拟合过程和结果信息。
        """

        A = 0.0
        B = 0.0

        for dist_type in ["euclidean", "cosine"]:
            for length, stats in logs.block_forward_stats_by_type[dist_type].items():
                t = stats["forward_total_time"]
                n = stats["forward_calls"]
                A += 3 * t / (n-1) * n

        for length, stats in logs.block_forward_stats_by_type["cross"].items():
            t = stats["forward_total_time"]
            n = stats["forward_calls"]
            B += 2 * t / (n-1) * n

        b = logs.global_backward_total_time - (A + B)
        logs.global_backward_b = b  # ✅ 存起来
        # self.backward_time_bias = b

        self.logger.info("\n[🔁 拟合反向传播时间模型]")
        self.logger.info(f"总反向传播时间: {logs.global_backward_total_time:.6f}s")
        self.logger.info(f"拟合模型: backward_total ≈ 2 × (A + B) + b")
        self.logger.info(f"A = {A:.6f}, B = {B:.6f}, b = {b:.6f}s")

    def transform(self, X, *, batch_size=512, result_type='tensor', normalize=False):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float)

        self.model.eval()
        dataset = torch.utils.data.TensorDataset(X)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        shapelet_transform = []
        for (x,) in dataloader:
            if self.to_cuda:
                x = x.cuda()
            with torch.no_grad():
                # shapelet_transform = self.model.transform(X)
                shapelet_transform.append(self.model(x, optimize=None).cpu())
        shapelet_transform = torch.cat(shapelet_transform, 0)
        if normalize:
            shapelet_transform = nn.functional.normalize(shapelet_transform, dim=1)
        if result_type == 'tensor':
            return shapelet_transform
        return shapelet_transform.detach().numpy()

    def predict(self, X, *, batch_size=512):

        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float)

        self.model.eval()
        dataset = torch.utils.data.TensorDataset(X)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        preds = []
        for (x,) in dataloader:
            if self.to_cuda:
                x = x.cuda()
            with torch.no_grad():
                # shapelet_transform = self.model.transform(X)
                preds.append(self.model(x).cpu())
        preds = torch.cat(preds, 0)

        return preds.detach().numpy()


    def plan_checkpoint_schedule_bucket(self):
        # 保留您原有的变量定义
        from logs import x, global_backward_b, global_pre_forward_mem, global_backward_peak_mem, cdist_euclidean_mem
        shapelet_lengths = list(self.shapelets_size_and_len.keys())
        s = 13

        # 保留您原有的日志数据收集逻辑
        T_euclidean = {}
        T_cosine = {}
        T_cross = {}
        peak_memory_e = {}
        peak_memory_c = {}

        for length in shapelet_lengths:
            stat = logs.block_forward_stats_by_type["euclidean"].get(length)
            if stat:
                T_euclidean[length] = stat["forward_total_time"] / (stat["forward_calls"] - 1)
                peak_memory_e[length] = stat["peak_mem"]

            stat = logs.block_forward_stats_by_type["cosine"].get(length)
            if stat:
                T_cosine[length] = stat["forward_total_time"] / (stat["forward_calls"] - 1)
                peak_memory_c[length] = stat["peak_mem"]

            stat = logs.block_forward_stats_by_type["cross"].get(length)
            if stat:
                T_cross[length] = stat["forward_total_time"] / (stat["forward_calls"] - 1)

        forward_peek = max(list(peak_memory_e.values()) + list(peak_memory_c.values()))

        # 保留您原有的所有内存计算函数 (get_M_e, get_M_c, get_S_e, get_S_c, is_E, is_C, get_length)
        def get_M_e(length):
            return 4 * self.batch_size * self.column * (self.length- length + 1) * length + cdist_euclidean_mem

        def get_M_c(length):
            return 4 * (2* s * self.column+ self.column*s*length+self.batch_size*self.column * (self.length - length + 1) *length + self.batch_size* s+ self.batch_size * (self.length - length + 1) * s)

        def get_S_e(length):
            return peak_memory_e[length]

        def get_S_c(length):
            return peak_memory_c[length]

        def is_E(i):
            return 1 <= i <= 8 or 17 <= i <= 24

        def is_C(i):
            return 9 <= i <= 16 or 25 <= i <= 32

        def get_length(i):
            return shapelet_lengths[(i - 1) % 8]

        # 以下部分是 Algorithm 1 贪心决策的实现
        # 1. 收集并获取所有模块的内存消耗 (使用您已计算好的 peak_memory_e/c)
        layer_memory = {}
        for i in range(1, 17):
            layer_memory[i] = get_S_e(get_length(i)) if is_E(i) else get_S_c(get_length(i))

        # 2. 按预测内存大小降序排序 (Algorithm 1, Line 3)
        sorted_layers = sorted(layer_memory.keys(), key=lambda i: layer_memory[i], reverse=True)

        # 3. 分桶 (Buckets) (Algorithm 1, Lines 4-12)
        buckets = []
        while sorted_layers:
            l = sorted_layers.pop(0)  # 取出内存最大的层 (Line 5)
            bucket = [l]
            # 将内存大小在 l 的 90% 以上的层归入同一桶 (Line 8)
            remaining = []
            for layer in sorted_layers:
                if layer_memory[layer] > layer_memory[l] * 0.9:
                    bucket.append(layer)
                else:
                    remaining.append(layer)
            # 对桶内层按索引升序排序 (即前向执行顺序) (Line 11)
            bucket.sort()
            buckets.append(bucket)
            sorted_layers = remaining

        # 4. 计算总内存和超支量 (Algorithm 1, Line 13)
        total_estimated_mem = sum(layer_memory.values())
        memory_limit = self.args.lim * 1024 ** 3
        excess_mem = total_estimated_mem - memory_limit

        # 5. 贪心决策：选择需要丢弃（检查点）的层 (Algorithm 1, Lines 14-21)
        # 初始化 x 数组，1 表示保留
        for i in range(1, 17):
            x[i] = 1

        while excess_mem > 0 and buckets:
            # 寻找候选桶：桶内最大内存 > excess_mem (Line 15)
            bucket_candidates = [b for b in buckets if layer_memory[b[0]] > excess_mem]

            if not bucket_candidates:
                # 如果没有候选桶，选择所有桶中内存最大的层 (Line 17)
                chosen_layer = buckets[0][0] # buckets[0][0] 是全局内存最大的层
            else:
                # 选择候选桶中内存最大的桶，并取其第一个元素 (Line 19)
                chosen_bucket = max(bucket_candidates, key=lambda b: layer_memory[b[0]])
                chosen_layer = chosen_bucket[0]
                # 从候选桶列表中移除该桶
                buckets.remove(chosen_bucket)

            # 将选中的层标记为需要丢弃（检查点）(Line 20)
            x[chosen_layer] = 0
            # 更新超支量 (Line 21)
            excess_mem -= layer_memory[chosen_layer]

            # 从所有桶中移除该层
            for bucket in buckets:
                if chosen_layer in bucket:
                    bucket.remove(chosen_layer)
                    if not bucket: # 如果桶为空，则移除该桶
                        buckets.remove(bucket)
                    break

        # 6. 保留您原有的日志记录
        self.logger.info(f"✅ 最优策略：{x[1:]}")
        all_lengths = list(self.shapelets_size_and_len.keys())
        logs.euclidean_checkpoint_shapelet_lengths.clear()
        logs.euclidean_checkpoint_shapelet_lengths.extend(
            [shapelet_lengths[i] for i in range(8) if x[i+1] == 0] # 注意索引从1开始
        )

        logs.cosine_checkpoint_shapelet_lengths.clear()
        logs.cosine_checkpoint_shapelet_lengths.extend(
            [shapelet_lengths[i] for i in range(8) if x[i + 9] == 0] # 注意索引从1开始
        )

        self.logger.info(f"📌 checkpoint 的欧氏长度:, {logs.euclidean_checkpoint_shapelet_lengths}")
        self.logger.info(f"📌 checkpoint 的余弦长度:, {logs.cosine_checkpoint_shapelet_lengths}")

        self.logger.info(f"💾 最终显存峰值：%.2f MB % {(excess_mem + memory_limit) / 1024 ** 2}")

        # 注意：这里不再返回任何值，也不再调用 differential_evolution 或 GA。
        # 检查点策略已经通过修改全局变量 `x` 来设置。





