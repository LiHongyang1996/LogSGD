import datetime
import os
import time
from contextlib import contextmanager
from typing import List

import numpy as np
import torch

try:
    import bit2byte
except ImportError:
    pass

def quantize_tensor_uni(tensor, num_bits=2):
    """
    将张量值量化为 num_bits 所表示的离散值。
    tensor: 输入的张量。
    num_bits: 量化位数，支持 2, 4, 8 或 16 位。
    返回量化后的张量，以及缩放因子和离散值集合。
    """
    # 设置量化范围，计算量化后的值数量
    qmin = 0
    qmax = 2**num_bits - 1  # 例如：2位时qmax=3，共4个离散值

    # 假设输入值在 [-1, 1] 的范围内
    min_val = -1
    max_val = 1

    # 计算 scale（步长），使得输入区间 [-1, 1] 均匀划分成 qmin 到 qmax 共 (qmax-qmin+1) 个值
    scale = (max_val - min_val) / (qmax - qmin)

    # 生成离散的量化值列表
    quantized_values = torch.linspace(min_val, max_val, qmax - qmin + 1)


    # 对每个输入值找到最接近的离散量化值
    quantized = torch.zeros_like(tensor)
    for i, value in enumerate(tensor):
        # 计算输入值与所有量化值的距离，找到最接近的值
        diff = torch.abs(quantized_values.to('cuda:0') - value)
        quantized[i] = quantized_values[torch.argmin(diff)]  # 找到最小的距离对应的量化值


    return quantized, scale, quantized_values


def quantize_tensor(tensor, num_bits=8, x=1.0):
    """
    Quantizes tensor values using a logarithmic distribution.
    tensor: Input tensor of any shape.
    num_bits: Number of quantization bits, supports 2, 4, 8, or 16 bits.
    x: Parameter to adjust the strength of the logarithmic distribution; the larger x is, the more concentrated the distribution near zero.

    Returns the quantized tensor, scaling factor, and the set of discrete quantized values.
    """
    # Set quantization range and calculate the number of quantized values
    qmin = 0
    qmax = 2 ** num_bits - 1  # e.g., for 2 bits, qmax=3, resulting in 4 discrete values

    # Assume input values are in the range [-1, 1]
    min_val = -1.0
    max_val = 1.0

    device = tensor.device  # Get the device (CPU or GPU) of the tensor

    # Create the logarithmic quantization levels
    linspace = torch.linspace(0, 1, qmax - qmin + 1, device=device)
    logspace = torch.sign(linspace - 0.5) * torch.log1p(torch.abs(linspace - 0.5) * x) / torch.log1p(
        torch.tensor(x, device=device))

    # Adjust logspace to the range [-1, 1]
    quantized_values = logspace * (max_val - min_val)

    # Compute scale (step size), even though it's a log distribution, we keep scale for subsequent calculations
    scale = (max_val - min_val) / (qmax - qmin)

    # Vectorized operation to find the closest quantized value for each input value
    # Expand quantized_values to match the tensor's shape for broadcasting
    diff = torch.abs(tensor.unsqueeze(-1) - quantized_values)

    # Find the index of the nearest quantized value
    indices = torch.argmin(diff, dim=-1)

    # Map indices to quantized values
    quantized = quantized_values[indices]

    return quantized, scale, quantized_values



class Reducer:
    def __init__(self, random_seed, device, timer):
        self.rng = np.random.RandomState(random_seed)
        M = 1024 * 1024
        self.precalc_numbers = (
            torch.from_numpy(self.rng.randn(128 * M)).to(device).type(torch.float32)
        )
        if torch.distributed.is_available():
            self.n_workers = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()
        else:
            self.n_workers = 1
            self.rank = 0
        self.device = device
        self.timer = timer

    def reduce(self, grad_in, grad_out, memory_out,quantized=0):
        """Return communicated bits"""
        raise NotImplementedError()


class SignAndNormReducer(Reducer):
    """
    Optimizations:
    pack all weights in one big vector
    turn that to bits
    """

    def reduce(self, grad_in, grad_out, memory_out, quantized = None):
        """
        Reduce gradients between the workers in place
        :param grad_in: dictionary
        :param grad_out: dictionary
        :param memory_out: dictionary
        """
        bits_communicated = 0
        sign_compressor = SignCompressor()

        with self.timer("reduce.flatpack"):
            flatgrad = TensorBuffer(grad_in)

        # Compute norms
        with self.timer("reduce.norms", verbosity=2):
            my_norms = torch.empty(len(grad_in), device=self.device)
            for i, tensor in enumerate(grad_in):
                my_norms[i] = tensor.norm(p=1)

        with self.timer("reduce.compress", verbosity=2):
            my_bits, sign_size = sign_compressor.compress(flatgrad.buffer)

        with self.timer("reduce.gather", verbosity=2):
            if self.n_workers > 1:
                bits = [torch.empty_like(my_bits) for i in range(self.n_workers)]
                norms = [torch.empty_like(my_norms) for i in range(self.n_workers)]
                h1 = all_gather(bits, my_bits, async_op=True)
                h2 = all_gather(norms, my_norms, async_op=True)
                h1.wait()
                h2.wait()
            else:
                bits = [my_bits]
                norms = [my_norms]

        bits_communicated += n_bits(my_bits)  # for the norm vector, being optimistic here
        bits_communicated += n_bits(my_norms)  # for the norm

        with self.timer("reduce.decompress", verbosity=2):
            flatsigns = []
            for their_bits in bits:
                uncompressed = sign_compressor.uncompress(their_bits, sign_size)
                flatsigns.append(uncompressed)

        with self.timer("reduce.average", verbosity=2):
            for out in grad_out:
                out.data[:] = 0.0

            for their_flatsigns, their_norms in zip(flatsigns, norms):
                flatgrad.buffer = their_flatsigns
                for sign, out, norm in zip(
                        flatgrad, grad_out, their_norms
                ):
                    out.data.add_(
                        norm / sign.nelement() / self.n_workers,
                        sign,
                    )

        with self.timer("reduce.memory", verbosity=2):
            for tensor, mem, norm in zip(grad_in, memory_out, my_norms):
                mem.data[:] = tensor
                mem.data.add_(-norm / tensor.nelement(), tensor.sign())


        return bits_communicated


class TopKReducer(Reducer):
    """
    Use same amount as rank-based
    """

    def __init__(self, random_seed, device, timer, compression=1 / 244):
        super().__init__(random_seed, device, timer)
        self.compression = compression

    def reduce(self, grad_in, grad_out, memory_out,quantized = 0):
        """
        Reduce gradients between the workers in place
        :param grad_in: dictionary
        :param grad_out: dictionary
        :param memory_out: dictionary
        """
        bits_communicated = 0

        with self.timer("reduce.flatpack", verbosity=2):
            # Find the size of a flatpacked gradient
            flatgrad_size = 0
            tensor_idx = [0]
            for tensor in grad_in:
                top_size = max(1, int(0.5 * self.compression * tensor.nelement()))
                flatgrad_size += top_size
                tensor_idx.append(tensor_idx[-1] + top_size)
            flatgrad_start_idx = tensor_idx[:-1]
            flatgrad_end_idx = tensor_idx[1:]
            flat_values = torch.empty(flatgrad_size, device=self.device)
            flat_positions = torch.empty(flatgrad_size, device=self.device, dtype=torch.int)

        with self.timer("reduce.topk", verbosity=2):
            for tensor, start, end in zip(grad_in, flatgrad_start_idx, flatgrad_end_idx):
                top_size = max(1, int(0.5 * self.compression * tensor.nelement()))
                _, positions = torch.topk(tensor.view(-1).abs(), top_size, sorted=False)
                values = tensor.view(-1)[positions].contiguous()
                flat_values[start:end] = values
                flat_positions[start:end] = positions

        num_bits = 32
        with self.timer("quantize.topk", verbosity=2):
            if quantized:
                num_bits = 8
                flat_values,_,_ = quantize_tensor(flat_values, num_bits)


        with self.timer("reduce.memory", verbosity=2):
            for tensor, mem, start, end in zip(
                    grad_in, memory_out, flatgrad_start_idx, flatgrad_end_idx
            ):
                positions = flat_positions[start:end]
                mem.data[:] = tensor
                mem.view(-1)[positions.long()] = 0.0

        with self.timer("reduce.gather", verbosity=2):
            if self.n_workers > 1:
                worker_values = [torch.empty_like(flat_values) for i in range(self.n_workers)]
                worker_positions = [torch.empty_like(flat_positions) for i in range(self.n_workers)]
                h1 = all_gather(worker_values, flat_values, async_op=True)
                h2 = all_gather(worker_positions, flat_positions, async_op=True)
                h1.wait()
                h2.wait()
            else:
                worker_values = [flat_values]
                worker_positions = [flat_positions]
            bits_communicated += n_bits(flat_values)//(32/num_bits) + n_bits(flat_positions)


        with self.timer("reduce.combine", verbosity=2):
            for tensor, out, start, end in zip(
                    grad_in, grad_out, flatgrad_start_idx, flatgrad_end_idx
            ):
                out.data[:] = 0
                for pos, val in zip(worker_positions, worker_values):
                    positions = pos[start:end]
                    values = val[start:end]
                    # out.view(-1)[pos].add_(1.0 / self.n_workers, val)
                    out.view(-1)[positions.long()] += values / self.n_workers

        return bits_communicated

class RankKReducer_original(Reducer):
    def __init__(self, random_seed, device, timer, n_power_iterations=0, reuse_query=False, rank=1):
        super().__init__(random_seed, device, timer)
        assert n_power_iterations == 0
        self.rank = rank
        self.p_memory = None
        self.q_memory = None
        self.reuse_query = reuse_query

    def set_random(self, vector):
        # torch.manual_seed(self.rng.randint(1_000_000_000))
        print(vector.shape)
        torch.manual_seed(42)
        vector.data[:] = torch.randn(*vector.shape, device=self.device)
        # orthogonalize(vector)

    def reduce(self, grad_in, grad_out, memory_out, round=-1, batch = 1):
        """
        Reduce gradients between the workers in place
        :param grad_in: dictionary
        :param grad_out: dictionary
        :param memory_out: dictionary
        """
        bits_communicated = 0

        # Split the tensors into rank1-ones that will be reduced un-compressed
        # and rank > 1 tensors that are compressed
        rank1_tensors = [
            (tensor, out, mem)
            for tensor, out, mem in zip(grad_in, grad_out, memory_out)
            if tensor.ndimension() <= 1
        ]
        high_rank_tensors = [
            (tensor, out, mem)
            for tensor, out, mem in zip(grad_in, grad_out, memory_out)
            if tensor.ndimension() > 1
        ]

        # We are building a rank-1 approximation of every tensor
        # that can be interpreted as a matrix. Let the approximation be
        # M = p q^T
        # We are allocating consequtive memory for the p's and q's

        memory_is_uninitialized = self.p_memory is None

        with self.timer("reduce.allocate_memory", verbosity=2):
            p_total_size = 0
            q_total_size = 0
            for tensor, _, _ in high_rank_tensors:
                matrix = tensor.view(tensor.shape[0], -1)
                n, m = matrix.shape
                # print(matrix.shape)
                rank = min(n, m, self.rank)
                p_total_size += n * rank
                q_total_size += m * rank
            if self.p_memory is None:
                self.p_memory = torch.empty(p_total_size, device=self.device)
                self.q_memory = torch.empty(q_total_size, device=self.device)
            # Find them again and make lists of pointers
            ps = []
            qs = []
            p_idx = 0
            q_idx = 0
            for tensor, _, _ in high_rank_tensors:
                matrix = tensor.view(tensor.shape[0], -1)
                n, m = matrix.shape
                rank = min(n, m, self.rank)
                ps.append(self.p_memory[p_idx: p_idx + n * rank].view(n, rank))
                qs.append(self.q_memory[q_idx: q_idx + m * rank].view(m, rank))
                p_idx += n * rank
                q_idx += m * rank

        with self.timer("reduce.prepare.q", verbosity=2):
            for (tensor, _, _), q, p in zip(high_rank_tensors, qs, ps):
                matrix = tensor.view(tensor.shape[0], -1)
                n, m = matrix.shape

                if self.reuse_query and not memory_is_uninitialized:
                    # orthogonalize(q)
                    pass
                else:
                    # Sample a query vector q
                    self.set_random(q)

        # print(q[0][0])

        with self.timer("reduce.compute.p", verbosity=2):
            for (tensor, _, _), q, p in zip(high_rank_tensors, qs, ps):
                matrix = tensor.view(tensor.shape[0], -1)
                torch.matmul(matrix, q, out=p)

        with self.timer("reduce.p", verbosity=2):
            all_reduce(self.p_memory)
            bits_communicated += n_bits(self.p_memory)

        # Start communicating rank 1 tensors
        with self.timer("reduce.rank1.pack", verbosity=2):
            rank1_tensor_list = TensorBuffer([tensor for (tensor, _, _) in rank1_tensors])
        with self.timer("reduce.rank1.all_reduce", verbosity=2):
            rank1_handle = rank1_tensor_list.all_reduce(async_op=True)
            bits_communicated += rank1_tensor_list.bits()

        with self.timer("reduce.normalize.p", verbosity=2):
            for p in ps:
                orthogonalize(p)

        with self.timer("reduce.compute.q", verbosity=2):
            for p, q, (tensor, _, _) in zip(ps, qs, high_rank_tensors):
                matrix = tensor.view(tensor.shape[0], -1)
                torch.matmul(matrix.t(), p, out=q)

        with self.timer("reduce.q", verbosity=2):
            all_reduce(self.q_memory)
            bits_communicated += n_bits(self.q_memory)
            self.q_memory.data[:] /= self.n_workers

        with self.timer("reduce.outerprod", verbosity=2):
            for p, q, (tensor, out, mem) in zip(ps, qs, high_rank_tensors):
                # Set the output gradient
                torch.matmul(p, q.t(), out=out.data[:])
                mem.data[:] = tensor - out

        with self.timer("reduce.rank1.unpack", verbosity=2):
            rank1_handle.wait()
            rank1_tensor_list.buffer /= self.n_workers
            rank1_tensor_list.unpack([out for (_, out, _) in rank1_tensors])

        # num_zeros_p = (self.p_memory == 0).sum().item()
        # num_zeros_q = (self.q_memory == 0).sum().item()
        # print(f"Number of zeros in p: {num_zeros_p}")
        # print(f"Number of zeros in q: {num_zeros_q}")

        # low_rank_elements = p_total_size + q_total_size
        # print(f"Low-rank p and q matrices total elements: {low_rank_elements}")
        #
        # original_grad_elements = sum(tensor.numel() for tensor, _, _ in high_rank_tensors)
        # print(f"Original gradients total elements (high_rank_tensors): {original_grad_elements}")
        #
        # rank1_elements = sum(tensor.numel() for tensor, _, _ in rank1_tensors)
        # print(rank1_elements)

        # min_p = self.p_memory.min().item()
        # max_p = self.p_memory.max().item()
        # min_q = self.q_memory.min().item()
        # max_q = self.q_memory.max().item()
        #
        # too = (min_p,max_p ,min_q,max_q )
        #
        # print(too)
        # 统计 p 和 q 矩阵中的零值数量


        return bits_communicated


class RankKReducer(Reducer):
    def __init__(self, random_seed, device, timer, n_power_iterations=0, reuse_query=False, rank=1):
        super().__init__(random_seed, device, timer)
        assert n_power_iterations == 0
        self.rank = rank
        self.p_memory = None
        self.q_memory = None
        self.reuse_query = reuse_query
    #
    def set_random(self, vector):
        torch.manual_seed(self.rng.randint(1_000_000_000))

        # torch.manual_seed(42)
        vector.data[:] = torch.randn(*vector.shape, device=self.device)
        # orthogonalize(vector)

    def reduce_original(self, grad_in, grad_out, memory_out):
        """
        Reduce gradients between the workers in place
        :param grad_in: dictionary
        :param grad_out: dictionary
        :param memory_out: dictionary
        """
        bits_communicated = 0

        # Split the tensors into rank1-ones that will be reduced un-compressed
        # and rank > 1 tensors that are compressed
        rank1_tensors = [
            (tensor, out, mem)
            for tensor, out, mem in zip(grad_in, grad_out, memory_out)
            if tensor.ndimension() <= 1
        ]
        high_rank_tensors = [
            (tensor, out, mem)
            for tensor, out, mem in zip(grad_in, grad_out, memory_out)
            if tensor.ndimension() > 1
        ]

        # We are building a rank-1 approximation of every tensor
        # that can be interpreted as a matrix. Let the approximation be
        # M = p q^T
        # We are allocating consequtive memory for the p's and q's

        memory_is_uninitialized = self.p_memory is None

        with self.timer("reduce.allocate_memory", verbosity=2):
            p_total_size = 0
            q_total_size = 0
            for tensor, _, _ in high_rank_tensors:
                matrix = tensor.view(tensor.shape[0], -1)
                n, m = matrix.shape
                rank = min(n, m, self.rank)
                p_total_size += n * rank
                q_total_size += m * rank
            if self.p_memory is None:
                self.p_memory = torch.empty(p_total_size, device=self.device)
                self.q_memory = torch.empty(q_total_size, device=self.device)

            # Find them again and make lists of pointers
            ps = []
            qs = []
            p_idx = 0
            q_idx = 0
            for tensor, _, _ in high_rank_tensors:
                matrix = tensor.view(tensor.shape[0], -1)
                n, m = matrix.shape
                rank = min(n, m, self.rank)
                ps.append(self.p_memory[p_idx: p_idx + n * rank].view(n, rank))
                qs.append(self.q_memory[q_idx: q_idx + m * rank].view(m, rank))
                p_idx += n * rank
                q_idx += m * rank

        with self.timer("reduce.prepare.q", verbosity=2):
            for (tensor, _, _), q, p in zip(high_rank_tensors, qs, ps):
                matrix = tensor.view(tensor.shape[0], -1)
                n, m = matrix.shape

                if self.reuse_query and not memory_is_uninitialized:
                    # orthogonalize(q)
                    pass
                else:
                    # Sample a query vector q
                    self.set_random(q)

        with self.timer("reduce.compute.p", verbosity=2):
            for (tensor, _, _), q, p in zip(high_rank_tensors, qs, ps):
                matrix = tensor.view(tensor.shape[0], -1)
                torch.matmul(matrix, q, out=p)

        with self.timer("reduce.p", verbosity=2):
            all_reduce(self.p_memory)
            bits_communicated += n_bits(self.p_memory)

        # Start communicating rank 1 tensors
        with self.timer("reduce.rank1.pack", verbosity=2):
            rank1_tensor_list = TensorBuffer([tensor for (tensor, _, _) in rank1_tensors])
        with self.timer("reduce.rank1.all_reduce", verbosity=2):
            rank1_handle = rank1_tensor_list.all_reduce(async_op=True)
            bits_communicated += rank1_tensor_list.bits()

        with self.timer("reduce.normalize.p", verbosity=2):
            for p in ps:
                orthogonalize(p)

        with self.timer("reduce.compute.q", verbosity=2):
            for p, q, (tensor, _, _) in zip(ps, qs, high_rank_tensors):
                matrix = tensor.view(tensor.shape[0], -1)
                torch.matmul(matrix.t(), p, out=q)

        with self.timer("reduce.q", verbosity=2):
            all_reduce(self.q_memory)
            bits_communicated += n_bits(self.q_memory)
            self.q_memory.data[:] /= self.n_workers

        with self.timer("reduce.outerprod", verbosity=2):
            for p, q, (tensor, out, mem) in zip(ps, qs, high_rank_tensors):
                # Set the output gradient
                torch.matmul(p, q.t(), out=out.data[:])
                mem.data[:] = tensor - out

        with self.timer("reduce.rank1.unpack", verbosity=2):
            rank1_handle.wait()
            rank1_tensor_list.buffer /= self.n_workers
            rank1_tensor_list.unpack([out for (_, out, _) in rank1_tensors])

        return bits_communicated

    def reduce(self, grad_in, grad_out, memory_out, quantized = False):
        """
        Reduce gradients between the workers in place
        :param grad_in: dictionary
        :param grad_out: dictionary
        :param memory_out: dictionary
        """
        if quantized == False:
            bits_communicated = self.reduce_original(grad_in, grad_out, memory_out)
            return bits_communicated
        bits_communicated = 0

        # Split the tensors into rank1-ones that will be reduced un-compressed
        # and rank > 1 tensors that are compressed
        rank1_tensors = [
            (tensor, out, mem)
            for tensor, out, mem in zip(grad_in, grad_out, memory_out)
            if tensor.ndimension() <= 1
        ]
        high_rank_tensors = [
            (tensor, out, mem)
            for tensor, out, mem in zip(grad_in, grad_out, memory_out)
            if tensor.ndimension() > 1
        ]

        # We are building a rank-1 approximation of every tensor
        # that can be interpreted as a matrix. Let the approximation be
        # M = p q^T
        # We are allocating consequtive memory for the p's and q's

        memory_is_uninitialized = self.p_memory is None


        with self.timer("reduce.allocate_memory", verbosity=2):
            p_total_size = 0
            q_total_size = 0
            for tensor, _, _ in high_rank_tensors:
                matrix = tensor.view(tensor.shape[0], -1)
                n, m = matrix.shape
                # print(matrix.shape)
                rank = min(n, m, self.rank)
                p_total_size += n * rank
                q_total_size += m * rank
            if self.p_memory is None:
                self.p_memory = torch.empty(p_total_size, device=self.device)
                self.q_memory = torch.empty(q_total_size, device=self.device)
            # Find them again and make lists of pointers
            ps = []
            qs = []
            p_idx = 0
            q_idx = 0
            for tensor, _, _ in high_rank_tensors:
                matrix = tensor.view(tensor.shape[0], -1)
                n, m = matrix.shape
                rank = min(n, m, self.rank)
                ps.append(self.p_memory[p_idx: p_idx + n * rank].view(n, rank))
                qs.append(self.q_memory[q_idx: q_idx + m * rank].view(m, rank))
                p_idx += n * rank
                q_idx += m * rank

        with self.timer("reduce.prepare.q", verbosity=2):
            for (tensor, _, _), q, p in zip(high_rank_tensors, qs, ps):
                matrix = tensor.view(tensor.shape[0], -1)
                n, m = matrix.shape
                if self.reuse_query and not memory_is_uninitialized:
                    # orthogonalize(q)
                    pass
                else:
                    # Sample a query vector q
                    self.set_random(q)

        # print(q[0][0])

        with self.timer("reduce.compute.p", verbosity=2):
            for (tensor, _, _), q, p in zip(high_rank_tensors, qs, ps):
                matrix = tensor.view(tensor.shape[0], -1)
                torch.matmul(matrix, q, out=p)

        num_bits_p = 8
        #量化后发送

        with self.timer("reduce.p_quantize", verbosity=2):
            # 对 p_memory 进行量化
            if quantized:
                self.p_memory, _,_ = quantize_tensor(self.p_memory,num_bits_p)

        # print(self.p_memory[0].cpu().element_size())


        #求和
        with self.timer("reduce.p_quantize_all_reduce", verbosity=2):
            # 量化后的 p_memory 进行 all_reduce
            all_reduce(self.p_memory)
            if quantized:
                bits_communicated += n_bits(self.p_memory)
                bits_communicated = bits_communicated // 4
            else:
                bits_communicated += n_bits(self.p_memory)


        with self.timer("reduce.normalize.p", verbosity=2):
            for p in ps:
                orthogonalize(p)
                if quantized:
                    p, _, _ = quantize_tensor(p, num_bits_p)

        with self.timer("reduce.compute.q", verbosity=2):
            for p, q, (tensor, _, _) in zip(ps, qs, high_rank_tensors):
                matrix = tensor.view(tensor.shape[0], -1)
                torch.matmul(matrix.t(), p, out=q)

        num_bits_q = 8
            # 量化后发送
        with self.timer("reduce.p_quantize", verbosity=2):
            # 对 p_memory 进行量化
            if quantized:
                self.q_memory, _, _ = quantize_tensor(self.q_memory, num_bits_p)

        with self.timer("reduce.q", verbosity=2):
            all_reduce(self.q_memory)
            self.q_memory, _, _ = quantize_tensor(self.q_memory, num_bits_p)
            if quantized:
                bits_communicated += n_bits(self.q_memory)//(32/num_bits_p)
            else:
                bits_communicated += n_bits(self.q_memory)
            self.q_memory.data[:] /= self.n_workers

        with self.timer("reduce.outerprod", verbosity=2):
            for p, q, (tensor, out, mem) in zip(ps, qs, high_rank_tensors):
                # Set the output gradient
                torch.matmul(p, q.t(), out=out.data[:])
                mem.data[:] = tensor - out

        num_bits_t1 = 16
        # Start communicating rank 1 tensors
        with self.timer("reduce.rank1.pack", verbosity=2):
            rank1_tensor_list = TensorBuffer([tensor for (tensor, _, _) in rank1_tensors])

        with self.timer("reduce.rank1.pack.quantized", verbosity=2):
            if quantized:
                rank1_tensor_list.quantize(num_bits_t1)

        with self.timer("reduce.rank1.all_reduce", verbosity=2):
            rank1_handle = rank1_tensor_list.all_reduce(async_op=True)
            if quantized:
                bits_communicated += rank1_tensor_list.bits()//(32/num_bits_t1)
            else:
                bits_communicated += rank1_tensor_list.bits()

        with self.timer("reduce.rank1.unpack", verbosity=2):
            rank1_handle.wait()
            if quantized:
                rank1_tensor_list.quantize(num_bits_t1)
            rank1_tensor_list.buffer /= self.n_workers
            rank1_tensor_list.unpack([out for (_, out, _) in rank1_tensors])

        return bits_communicated


@torch.jit.script
def orthogonalize(matrix, eps=torch.tensor(1e-8)):
    n, m = matrix.shape
    for i in range(m):
        # Normalize the i'th column
        col = matrix[:, i: i + 1]
        col /= torch.sqrt(torch.sum(col ** 2)) + eps
        # Project it on the rest and remove it
        if i + 1 < m:
            rest = matrix[:, i + 1:]
            # rest -= torch.matmul(col.t(), rest) * col
            rest -= torch.sum(col * rest, dim=0) * col


class ExactReducer(Reducer):
    def reduce(self, grad_in, grad_out, memory_out,a=0):
        """
        Reduce gradients between the workers in place
        :param grad_in: dictionary
        :param grad_out: dictionary
        :param memory_out: dictionary
        """
        with self.timer("reduce.zero_mem", verbosity=2):
            for mem in memory_out:
                mem.zero_()

        with self.timer("reduce.build_lists", verbosity=2):
            list_in = grad_in
            list_out = grad_out

        with self.timer("reduce.reduce", verbosity=2):
            bits_communicated = reduce_mean_list(self.device, list_in, list_out, self.timer)

        return bits_communicated


def reduce_mean_list(
        device: torch.device, list_in: List[torch.Tensor], list_out: List[torch.Tensor], timer
):
    if torch.distributed.is_available():
        n_workers = torch.distributed.get_world_size()
    else:
        n_workers = 1

    if n_workers == 1:
        for t_in, t_out in zip(list_in, list_out):
            t_out[:] = t_in
        return 0

    with timer("reduce.mean.pack"):
        buffer = TensorBuffer(list_in)

    with timer("reduce.mean.allreduce"):
        buffer.all_reduce()
        buffer.buffer /= n_workers
        bits_communicated = buffer.bits()

    with timer("reduce.mean.unpack", verbosity=2):
        buffer.unpack(list_out)

    return bits_communicated


def n_bits(tensor):
    return 8 * tensor.nelement() * tensor.element_size()


class TensorBuffer():
    """
    Packs multiple tensors into one flat buffer for efficient
    intra-worker communication.
    """

    def __init__(self, tensors):
        indices = [0]
        for tensor in tensors:
            new_end = indices[-1] + tensor.nelement()
            indices.append(new_end)

        self._start_idx = indices[:-1]
        self._end_idx = indices[1:]
        self._tensors = tensors

        self.buffer = torch.cat([t.view(-1) for t in tensors])  # copies

    def __getitem__(self, index):
        return self.buffer[self._start_idx[index]: self._end_idx[index]].view(*self._tensors[index].shape)

    def __len__(self):
        return len(self._tensors)

    def pack(self, tensors=None):
        # Optional. init already does this.
        if tensors is None:
            tensors = self._tensors
        for tensor, entry in zip(tensors, self):
            entry[:] = tensor

    def unpack(self, tensors):
        for tensor, entry in zip(tensors, self):
            tensor[:] = entry

    def nelement(self):
        return self.buffer.nelement()

    def element_size(self):
        return self.buffer.element_size()

    def bits(self):
        return 8 * self.nelement() * self.element_size()

    def all_reduce(self, async_op=False):
        return torch.distributed.all_reduce(self.buffer, async_op=async_op)

    def all_gather(self, async_op=False):
        n_workers = torch.distributed.get_world_size() if torch.distributed.is_available() else 1
        buffers = [torch.empty_like(self.buffer) for i in range(n_workers)]
        handle = all_gather(buffers, self.buffer, async_op=async_op)
        if async_op:
            return buffers, handle
        else:
            return buffers

    def quantize(self, num_bits=8, x=1.0):
        """
        对 buffer 进行对数量化，以减少张量的大小。
        num_bits: 量化位数。
        x: 控制对数分布强度的参数，x 越大，分布越集中在靠近 0 的区域。
        """

        # 设置量化范围
        qmin = 0
        qmax = 2 ** num_bits - 1  # 例如，8 位量化时 qmax=255

        # 假设输入值在 [-1, 1] 的范围内
        min_val = -1.0
        max_val = 1.0

        device = self.buffer.device  # 获取张量所在的设备（CPU 或 GPU）

        # 生成对数分布的量化值
        linspace = torch.linspace(0, 1, qmax - qmin + 1, device=device)
        logspace = torch.sign(linspace - 0.5) * torch.log1p(torch.abs(linspace - 0.5) * x) / torch.log1p(
            torch.tensor(x, device=device)
        )
        quantized_values = logspace * (max_val - min_val)

        # 确保量化值是升序排列
        quantized_values, _ = torch.sort(quantized_values)

        # 将 buffer 展平成一维张量
        flat_buffer = self.buffer.flatten()

        # 使用 searchsorted 找到每个元素对应的量化值索引
        indices = torch.searchsorted(quantized_values, flat_buffer, right=True)

        # 防止索引超出范围
        indices = torch.clamp(indices, 0, len(quantized_values) - 1)

        # 获取量化后的值
        quantized_buffer = quantized_values[indices].view_as(self.buffer)

        # 将量化结果保存回 buffer
        self.buffer.copy_(quantized_buffer)


def all_reduce(*args, **kwargs):
    if torch.distributed.is_available() and torch.distributed.get_world_size() > 1:
        return torch.distributed.all_reduce(*args, **kwargs)


def all_gather(out_list, in_tensor, **kwargs):
    if torch.distributed.is_available() and torch.distributed.get_world_size() > 1:
        return torch.distributed.all_gather(out_list, in_tensor, **kwargs)
    else:
        assert len(out_list) == 1
        out_list[0].data = in_tensor


@torch.jit.script
def l2norm(x):
    return torch.sqrt(torch.sum(x ** 2))


def normalize_(tensor):
    """Divide by L2 norm. In place"""
    tensor /= l2norm(tensor)


class SignCompressor:
    """Taken from https://github.com/PermiJW/signSGD-with-Majority-Vote"""

    def packing(self, src_tensor):
        src_tensor = torch.sign(src_tensor)
        src_tensor_size = src_tensor.size()
        src_tensor = src_tensor.view(-1)
        src_len = len(src_tensor)
        add_elm = 32 - (src_len % 32)
        if src_len % 32 == 0:
            add_elm = 0
        new_tensor = torch.zeros([add_elm], dtype=torch.float32, device=src_tensor.device)
        src_tensor = torch.cat((src_tensor, new_tensor), 0)
        src_tensor = src_tensor.view(32, -1)
        src_tensor = src_tensor.to(dtype=torch.int32)
        dst_tensor = bit2byte.packing(src_tensor)
        dst_tensor = dst_tensor.to(dtype=torch.int32)
        return dst_tensor, src_tensor_size

    def unpacking(self, src_tensor, src_tensor_size):
        src_element_num = self.element_num(src_tensor_size)
        add_elm = 32 - (src_element_num % 32)
        if src_element_num % 32 == 0:
            add_elm = 0
        src_tensor = src_tensor.int()
        new_tensor = torch.ones(
            src_element_num + add_elm, device=src_tensor.device, dtype=torch.int32
        )
        new_tensor = new_tensor.view(32, -1)
        new_tensor = bit2byte.unpacking(src_tensor, new_tensor)
        new_tensor = new_tensor.view(-1)
        new_tensor = new_tensor[:src_element_num]
        new_tensor = new_tensor.view(src_tensor_size)
        new_tensor = -new_tensor.add_(-1)
        new_tensor = new_tensor.float()
        return new_tensor

    def majority_vote(self, src_tensor_list):
        voter_num = len(src_tensor_list)
        src_tensor = torch.stack(src_tensor_list)
        src_tensor = src_tensor.view(-1)
        full_size = 32 * len(src_tensor)
        new_tensor = torch.ones(full_size, device=src_tensor.device, dtype=torch.int32)
        new_tensor = new_tensor.view(32, -1)
        new_tensor = bit2byte.unpacking(src_tensor, new_tensor)
        new_tensor = -new_tensor.add_(-1)
        # sum
        new_tensor = new_tensor.permute(1, 0).contiguous().view(voter_num, -1)
        new_tensor = torch.sum(new_tensor, 0)
        new_tensor = new_tensor.view(-1, 32).permute(1, 0)
        new_tensor = torch.sign(new_tensor)
        new_tensor = bit2byte.packing(new_tensor)
        new_tensor = new_tensor.to(dtype=torch.int32)
        return new_tensor

    def element_num(self, size):
        num = 1
        for i in range(len(size)):
            num *= size[i]
        return num

    def compress(self, src_tensor):
        return self.packing(src_tensor)

    def uncompress(self, src_tensor, src_tensor_size):
        dst_tensor = self.unpacking(src_tensor, src_tensor_size)
        return dst_tensor