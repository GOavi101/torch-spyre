## Spyre Inductor Operation Cookbook

This document describe the common patterns used to define operations
in the front-end compiler.

### View operations (no data copy)

View operations on Spyre use `as_strided_with_layout` (C++) so they share
storage with the base tensor and do not copy data:

- **aten::slice.Tensor** — Implemented in [ops.py](../torch_spyre/ops.py) as a view: `infer_slice_geometry` computes new sizes, strides, storage offset, and a layout derived from the original (only the sliced dimension’s size changes); result is `as_strided_with_layout(...)`.
- **aten::pad** (negative pad only) — Negative pad values mean “crop”; implemented as a view by applying `aten::slice.Tensor` per dimension (no allocation).
- **aten::transpose.int**, **aten::squeeze**, **aten::unsqueeze** — Implemented as views in ops.py via `as_strided_with_layout` with an updated layout.
- **aten::view**, **aten::_unsafe_view** — Implemented in C++ ([spyre_views.cpp](../torch_spyre/csrc/spyre_views.cpp)) via `compute_view_layout` and `spyre_alias_with_sizes_and_strides`.

Copying a view back to CPU (e.g. `.cpu()`) uses a row-by-row path when `storage_offset != 0` so the runtime’s single-dcsi behaviour still produces the correct result.

### Slice and pad: decomposition and lowering

- **aten::slice.Tensor** and **aten::pad** have eager kernels in [ops.py](../torch_spyre/ops.py) and **lowerings** in [lowering.py](../torch_spyre/_inductor/lowering.py) that delegate to inductor’s `make_fallback`, so they run as eager ops at runtime inside compiled graphs. There are **no** decompositions for them; lowering produces a `FallbackKernel` that calls the registered spyre kernel. Stickify and core_division handle `FallbackKernel` (see [stickify.py](../torch_spyre/_inductor/stickify.py), [core_division.py](../torch_spyre/_inductor/core_division.py)).

### Direct mapping from ATen to OpFunc

If a pointwise ATen operation can be implemented with a single Spyre OpFunc,
then enabling it in our backend only requires
adding a method to `SpyreOpFuncs` in [spyre_kernel.py](../torch_spyre/_inductor/spyre_kernel.py).
Canonical examples are `add` and `softplus` (see `softplus`for an example of using `op_info` for non-tensor arguments).

Note that some pointwise ATen operations that can be be implemented with a single Spyre OpFunc
have default decompositions defined by Inductor. Adding a method to
`SpyreOpFuncs` in [spyre_kernel.py](../torch_spyre/_inductor/spyre_kernel.py)
overrides the default decomposition and thus enables the desired direct mapping.
Canonical examples are `reciprocal` and `sigmoid`.

### Spyre-specific decompositions

We define Spyre-specific decompositions in [decompositions.py](../torch_spyre/_inductor/decompositions.py)
using the `@register_decomposition` decorator.  Decompositions are graph transformations
that are performed before the graph is lowered to loop level IR.

### Spyre-specific lowerings

We define Spyre-specific lowerings from ATen operations to Inductor's
loop level IR in [lowering.py](../torch_spyre/_inductor/lowering.py) using the `@register_spyre_lowering` decorator.

### Spyre-specific OpFuncs

For Spyre OpFuncs that do not have corresponding ATen operations, we use
the `@torch.library.custom_op` decorator to define a new operation in
[customops.py](../torch_spyre/_inductor/customops.py). This has two pieces:
+ defining the signature of the operation (using `@custom_op`)
+ defining its fake function (using the `@opname.register_fake` that is defined as part of the `@custom_op`)

In addition, when defining a custom op, you will also need to do one of:
+ register a lowering for the custom op in [lowering.py](../torch_spyre/_inductor/lowering.py) and
  add a method to `SpyreOpFuncs` in [spyre_kernel.py](../torch_spyre/_inductor/spyre_kernel.py).
  A canonical example is `spyre.clamp`.
+ register a decomposition for the custom op in [decompositions.py](../torch_spyre/_inductor/decompositions.py)
  that removes the custom op from the graph before lowering. A canonical example is `spyre.compact`.
+ define a `CustomPrePass` or `CustomPostPass` that implements a more general graph
  rewrite that removes the custom op from the graph before lowering. We currently do not have any custom ops that use this option.
