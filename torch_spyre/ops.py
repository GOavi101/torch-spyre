# Copyright 2025 The Torch-Spyre Authors.
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

import torch
import torch_spyre.fallbacks  # noqa: F401
from typing import List, Optional, Union


def maybe_wrap_dim(dim, ndims):
    if dim < 0:
        return dim + ndims
    return dim


@torch.library.register_kernel("aten::mm", ["spyre"])
def spyre__mm(self: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
    compiled_mm = torch.compile(torch.mm, dynamic=False)
    return compiled_mm(self, mat2)


@torch.library.register_kernel("aten::mm.out", ["spyre"])
def spyre__mm_out(
    self: torch.Tensor, mat2: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    compiled_mm = torch.compile(torch.mm, dynamic=False)
    return compiled_mm(self, mat2, out=out)


@torch.library.register_kernel("aten::fill_.Scalar", ["spyre"])
def spyre__fill_scalar(
    self: torch.Tensor, other: Union[int, float, bool, complex]
) -> torch.Tensor:
    tmp = torch.ones(self.size(), dtype=self.dtype) * other
    self.copy_(tmp)
    return self


@torch.library.register_kernel("aten::transpose.int", ["spyre"])
def spyre__transpose_int(self: torch.Tensor, dim0: int, dim1: int) -> torch.Tensor:
    ndims = self.dim()
    dim0 = maybe_wrap_dim(dim0, ndims)
    dim1 = maybe_wrap_dim(dim1, ndims)

    # Transpose of a tensor is a view operation.
    if dim0 == dim1:
        return torch.ops.aten.alias(self)

    sizes = list(self.shape)
    sizes[dim0], sizes[dim1] = sizes[dim1], sizes[dim0]
    strides = list(self.stride())
    strides[dim0], strides[dim1] = strides[dim1], strides[dim0]
    prev_stl = self.device_tensor_layout()
    dim_map = prev_stl.dim_map
    for idx, dim in enumerate(dim_map):
        if dim == dim0:
            dim_map[idx] = dim1
        elif dim == dim1:
            dim_map[idx] = dim0
    new_stl = torch_spyre._C.SpyreTensorLayout(
        prev_stl.device_size, dim_map, prev_stl.device_dtype
    )

    result = torch_spyre._C.as_strided_with_layout(
        self, sizes, strides, self.storage_offset(), new_stl
    )
    return result


# derived from https://github.com/pytorch/pytorch/blob/f91f262275e12bd6249a2bcd2c3c06e0c78e20ee/aten/src/ATen/native/TensorShape.cpp#L3897
# with changes specific for spyre
def infer_squeeze_geometry(
    tensor: torch.Tensor, dims: Optional[int | list[int]] = None
):
    sizes = []
    strides = []
    current_stl = tensor.device_tensor_layout()
    stick_dim = current_stl.host_stick_dim()
    dim_map = current_stl.dim_map

    for idx in range(tensor.dim()):
        dim_check = False

        # Handle the cases where dims is set
        if isinstance(dims, int):
            dim_check = idx != dims
        elif isinstance(dims, list):
            dim_check = idx not in dims

        # Keep any dim > 1 that fulfills the dim_check
        if dim_check or tensor.size(idx) != 1:
            sizes.append(tensor.size(idx))
            strides.append(tensor.stride(idx))
        elif idx == stick_dim:
            # We cannot squeeze the stick dimension!
            raise ValueError("The stick dimension cannot be squeezed")
        else:
            # For the squeezed dimensions, correct the dim_map by
            # lowering the dimensions after the squeezed one
            for dim_idx in range(len(dim_map)):
                if dim_map[dim_idx] >= idx:
                    dim_map[dim_idx] -= 1

    new_stl = torch_spyre._C.SpyreTensorLayout(
        current_stl.device_size, dim_map, current_stl.device_dtype
    )

    return sizes, strides, new_stl


@torch.library.register_kernel("aten::squeeze", ["spyre"])
def spyre__squeeze(self: torch.Tensor) -> torch.Tensor:
    sizes, strides, new_stl = infer_squeeze_geometry(self)

    result = torch_spyre._C.as_strided_with_layout(
        self, sizes, strides, self.storage_offset(), new_stl
    )
    return result


@torch.library.register_kernel("aten::squeeze.dim", ["spyre"])
def spyre__squeeze_dim(self: torch.Tensor, dim: int) -> torch.Tensor:
    sizes, strides, new_stl = infer_squeeze_geometry(self, dim)

    result = torch_spyre._C.as_strided_with_layout(
        self, sizes, strides, self.storage_offset(), new_stl
    )
    return result


@torch.library.register_kernel("aten::squeeze.dims", ["spyre"])
def spyre__squeeze_dims(self: torch.Tensor, dim: list[int]) -> torch.Tensor:
    sizes, strides, new_stl = infer_squeeze_geometry(self, dim)

    result = torch_spyre._C.as_strided_with_layout(
        self, sizes, strides, self.storage_offset(), new_stl
    )
    return result


# derived from https://github.com/pytorch/pytorch/blob/f91f262275e12bd6249a2bcd2c3c06e0c78e20ee/aten/src/ATen/native/TensorShape.cpp#L3943
# with changes specific to Spyre
def infer_unsqueeze_geometry(tensor: torch.Tensor, dim: int):
    sizes = list(tensor.size())
    strides = list(tensor.stride())

    new_stride = 1
    if dim < tensor.dim():
        new_stride = sizes[dim] * strides[dim]

    sizes.insert(dim, 1)
    strides.insert(dim, new_stride)

    current_stl = tensor.device_tensor_layout()
    dim_map = current_stl.dim_map

    for dim_idx in range(len(dim_map)):
        if dim_map[dim_idx] >= dim:
            dim_map[dim_idx] += 1

    new_stl = torch_spyre._C.SpyreTensorLayout(
        current_stl.device_size, dim_map, current_stl.device_dtype
    )

    return sizes, strides, new_stl


@torch.library.register_kernel("aten::unsqueeze", ["spyre"])
def spyre__unsqueeze(self: torch.Tensor, dim: int) -> torch.Tensor:
    sizes, strides, new_stl = infer_unsqueeze_geometry(self, dim)

    result = torch_spyre._C.as_strided_with_layout(
        self, sizes, strides, self.storage_offset(), new_stl
    )
    return result


def infer_slice_geometry(
    tensor: torch.Tensor,
    dim: int,
    start: Optional[int],
    end: Optional[int],
    step: int,
):
    """
    Compute new sizes, strides, and storage offset for slice along one dimension.
    Matches PyTorch slice.Tensor semantics (view, no copy).
    """
    ndim = tensor.dim()
    if ndim == 0:
        raise RuntimeError("slice() cannot be applied to a 0-dim tensor")
    dim = maybe_wrap_dim(dim, ndim)
    dim_size = tensor.size(dim)
    if step == 0:
        raise RuntimeError("slice step cannot be zero")

    if start is None:
        start_val = 0 if step > 0 else dim_size - 1
    else:
        start_val = start + dim_size if start < 0 else start
        start_val = max(0, min(dim_size, start_val))

    if end is None:
        end_val = dim_size if step > 0 else -1
    else:
        end_val = end + dim_size if end < 0 else end
        end_val = max(0, min(dim_size, end_val))

    if step > 0:
        length = max(0, (end_val - start_val + step - 1) // step)
    else:
        length = max(0, (start_val - end_val - step - 1) // (-step))

    new_sizes = list(tensor.size())
    new_sizes[dim] = length
    new_strides = list(tensor.stride())
    new_strides[dim] = tensor.stride(dim) * step
    new_storage_offset = tensor.storage_offset() + start_val * tensor.stride(dim)

    current_stl = tensor.device_tensor_layout()
    # Preserve the original layout and only change the size of the sliced dimension.
    # This keeps device/host dimension mapping so copy produces correct host shape.
    new_device_size = list(current_stl.device_size)
    new_dim_map = list(current_stl.dim_map)
    old_host_size = tensor.size(dim)
    for i, host_dim in enumerate(new_dim_map):
        if host_dim == dim and new_device_size[i] == old_host_size:
            # This device dimension is the logical size for the sliced host dim (not stick).
            new_device_size[i] = length
    new_stl = torch_spyre._C.SpyreTensorLayout(
        new_device_size, new_dim_map, current_stl.device_dtype
    )
    return new_sizes, new_strides, new_storage_offset, new_stl


@torch.library.register_kernel("aten::slice.Tensor", ["spyre"])
def spyre__slice_Tensor(
    self: torch.Tensor,
    dim: int = 0,
    start: Optional[int] = None,
    end: Optional[int] = None,
    step: int = 1,
) -> torch.Tensor:
    """Slice along one dimension (view, no copy). Implemented in Python in ops.py."""
    new_sizes, new_strides, new_storage_offset, new_stl = infer_slice_geometry(
        self, dim, start, end, step
    )
    return torch_spyre._C.as_strided_with_layout(
        self, new_sizes, new_strides, new_storage_offset, new_stl
    )


def _to_int(x: Union[int, torch.SymInt]) -> int:
    """Coerce SymInt or int to int for use in pad/slice logic."""
    if hasattr(x, "guard_int"):
        return x.guard_int("pad", 0)
    return int(x)


def _pad_parse_and_check(
    tensor: torch.Tensor,
    pad: Union[List[int], tuple],
    mode: str,
    value: Optional[float],
) -> tuple:
    """
    Parse pad list and check mode. Pad format: (last_dim_left, last_dim_right,
    second_last_left, second_last_right, ...). Returns (ndim_padded, pad_per_dim)
    where pad_per_dim entries are (d, pad_before, pad_after) for each padded dim d.
    """
    pad = [_to_int(p) for p in pad]
    ndim = tensor.dim()
    if len(pad) % 2 != 0:
        raise RuntimeError("Padding length must be even")
    k = len(pad) // 2
    if k > ndim:
        raise RuntimeError(
            f"Padding length {len(pad)} implies {k} dimensions but tensor has {ndim}"
        )
    if mode != "constant":
        raise RuntimeError(
            f"pad on spyre only supports mode='constant', got mode='{mode}'"
        )
    pad_per_dim = []
    for i in range(k):
        d = ndim - 1 - i  # dimension index (last dim first)
        pad_before = pad[2 * i]
        pad_after = pad[2 * i + 1]
        pad_per_dim.append((d, pad_before, pad_after))
    return k, pad_per_dim


@torch.library.register_kernel("aten::pad", ["spyre"])
def spyre__pad(
    self: torch.Tensor,
    pad: List[int],
    mode: str = "constant",
    value: Optional[float] = None,
) -> torch.Tensor:
    """
    Pad (or crop when pad values are negative) on spyre.
    - Negative pad: crop (view, no copy), implemented via slice.
    - Positive pad: allocate, fill with value, copy input into center (copy).
    Only mode='constant' is supported on device.
    """
    fill_value = value if value is not None else 0.0
    k, pad_per_dim = _pad_parse_and_check(self, pad, mode, value)

    ndim = self.dim()
    sizes = list(self.size())

    all_non_positive = all(pb <= 0 and pa <= 0 for _, pb, pa in pad_per_dim)

    if all_non_positive:
        # Crop only: implement as a view via successive slices (no copy)
        slice_op = torch.ops.aten.slice.Tensor
        result = self
        for d, pad_before, pad_after in pad_per_dim:
            start_d = max(0, -pad_before)
            end_d = sizes[d] + pad_after  # pad_after <= 0 so this crops from end
            end_d = max(start_d, min(sizes[d], end_d))
            result = slice_op(result, d, start_d, end_d, 1)
            sizes[d] = end_d - start_d
        return result

    # At least one positive pad: allocate output, fill, then copy input into place
    new_sizes = list(self.size())
    for d, pad_before, pad_after in pad_per_dim:
        new_sizes[d] = sizes[d] + pad_before + pad_after
        if new_sizes[d] < 0:
            raise RuntimeError(
                f"pad would make dimension {d} negative: "
                f"size {sizes[d]} + {pad_before} + {pad_after}"
            )

    out = torch.empty(new_sizes, dtype=self.dtype, device=self.device)
    out.fill_(fill_value)

    # Destination view: out[pad_before_d : pad_before_d+size_d, ...] for each dim
    slice_op = torch.ops.aten.slice.Tensor
    dest = out
    for d, pad_before, _ in pad_per_dim:
        dest = slice_op(dest, d, pad_before, pad_before + self.size(d), 1)
    dest.copy_(self)
    return out


# INSERT_CODEGEN_HERE
