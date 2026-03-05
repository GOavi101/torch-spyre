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

from torch_spyre._inductor.codegen.compute_ops import (
    num_bytes,
    DimInfos,
    get_device_size,
    create_tensor_specific_layouts,
    gen_coord_info_value,
)
from torch_spyre._inductor.constants import (
    INPUT_DIM_LABELS,
    OUTPUT_DIM_LABELS,
)
import math


def generate_transpose(pointers, *, op, dimensions, inputs, outputs, **kwargs):
    return {
        "reshape": {
            "numCoresUsed_": 1,
            "dscs_": [],
            "datadscs_": [
                {
                    "reshape": {
                        "coreIdsUsed_": [0],
                        "dimPool_": ["mb", "out"],
                        "primaryDs_": [{"name_": "pds0", "dimNames": ["mb", "out"]}],
                        "labeledDs_": [
                            {
                                "pdsName_": "pds0",
                                "wordLength": 2,
                                "dataformat": "SEN169_FP16",
                                "layoutDimOrder_": ["mb", "out"],
                                "stickDimOrder_": ["out"],
                                "dimToLayoutSize_": {
                                    "mb": dimensions[0],
                                    "out": dimensions[1],
                                },
                                "dimToStickSize_": {"out": 64},
                                "validGap_": {
                                    "mb": [[64, 0]],
                                    "out": [[64, 0]],
                                },
                                "PieceInfo": [
                                    {
                                        "key_": "p0",
                                        "dimToSize_": {"mb": 64, "out": 64},
                                        "PlacementInfo": [
                                            {
                                                "type": "hbm",
                                                "memId": [-1],
                                                "startAddr": [
                                                    pointers[inputs[0]["name"]] // 128
                                                ],
                                            },
                                            {
                                                "type": "lx",
                                                "memId": [0],
                                                "startAddr": [0],
                                            },
                                        ],
                                    },
                                    {
                                        "key_": "p1",
                                        "dimToSize_": {"mb": 64, "out": 64},
                                        "PlacementInfo": [
                                            {
                                                "type": "hbm",
                                                "memId": [-1],
                                                "startAddr": [0],
                                            },
                                            {
                                                "type": "lx",
                                                "memId": [0],
                                                "startAddr": [8192],
                                            },
                                        ],
                                    },
                                ],
                                "hbmStartAddress_": pointers[inputs[0]["name"]] // 128,
                            },
                            {
                                "pdsName_": "pds0",
                                "wordLength": 2,
                                "dataformat": "SEN169_FP16",
                                "layoutDimOrder_": ["out", "mb"],
                                "stickDimOrder_": ["mb"],
                                "dimToLayoutSize_": {
                                    "mb": dimensions[0],
                                    "out": dimensions[1],
                                },
                                "dimToStickSize_": {"mb": 64},
                                "validGap_": {
                                    "mb": [[64, 0]],
                                    "out": [[64, 0]],
                                },
                                "PieceInfo": [
                                    {
                                        "key_": "p0",
                                        "dimToSize_": {"mb": 64, "out": 64},
                                        "validGap_": {
                                            "out": [[64, 0]],
                                            "mb": [[64, 0]],
                                        },
                                        "PlacementInfo": [
                                            {
                                                "type": "hbm",
                                                "memId": [-1],
                                                "startAddr": [
                                                    pointers[outputs[0]["name"]] // 128
                                                ],
                                            },
                                            {
                                                "type": "lx",
                                                "memId": [0],
                                                "startAddr": [16384],
                                            },
                                        ],
                                    },
                                    {
                                        "key_": "p1",
                                        "dimToSize_": {"mb": 64, "out": 64},
                                        "PlacementInfo": [
                                            {
                                                "type": "hbm",
                                                "memId": [-1],
                                                "startAddr": [0],
                                            },
                                            {
                                                "type": "lx",
                                                "memId": [0],
                                                "startAddr": [24576],
                                            },
                                        ],
                                    },
                                ],
                                "hbmStartAddress_": pointers[outputs[0]["name"]] // 128,
                            },
                        ],
                        "op": {
                            "name": "ReStickifyOpWithPTHBM",
                            "coreIDtoANInfo": {
                                "0": {
                                    "loopCount": {
                                        "out": dimensions[0] // 64,
                                        "mb": dimensions[1] // 64,
                                    },
                                    "loopCountL3SU": {
                                        "out": dimensions[0] // 64,
                                        "mb": dimensions[1] // 64,
                                    },
                                    "addr_info_": {
                                        "l3lu": {
                                            "type_": "stride",
                                            "offset_": {
                                                "mb": dimensions[0],
                                                "out": 64,
                                            },
                                        },
                                        "l3su": {
                                            "type_": "stride",
                                            "offset_": {
                                                "mb": 64,
                                                "out": dimensions[1],
                                            },
                                        },
                                    },
                                    "inpPieceOrder": [
                                        f"p{i % 2}"
                                        for i in range(
                                            dimensions[0] * dimensions[1] // 4096
                                        )
                                    ],
                                    "outPieceOrder": [
                                        f"p{i % 2}"
                                        for i in range(
                                            dimensions[0] * dimensions[1] // 4096
                                        )
                                    ],
                                }
                            },
                            "numClToUse": 1,
                            "cl0ToLxOffsetLU": 0,
                            "cl0ToLxOffsetSU": 0,
                        },
                    }
                }
            ],
        }
    }


# 3D transpose restickify
def generate_transpose_3d_stick(
    pointers, *, op, dimensions, inputs, outputs, transposed_dims, **kwargs
):
    transpose_0_2 = 0 in transposed_dims and 2 in transposed_dims
    return {
        "reshape": {
            "numCoresUsed_": 1,
            "dscs_": [],
            "datadscs_": [
                {
                    "reshape": {
                        "coreIdsUsed_": [0],
                        "dimPool_": ["mb", "out", "x"],
                        "primaryDs_": [
                            {"name_": "pds0", "dimNames": ["mb", "out", "x"]}
                        ],
                        "labeledDs_": [
                            {
                                "pdsName_": "pds0",
                                "wordLength": 2,
                                "dataformat": "SEN169_FP16",
                                "layoutDimOrder_": ["mb", "out", "x"],
                                "stickDimOrder_": ["out"],
                                "dimToLayoutSize_": {
                                    "mb": dimensions[0],
                                    "out": dimensions[-1],
                                    "x": dimensions[1],
                                },
                                "dimToStickSize_": {"out": 64},
                                "validGap_": {
                                    "mb": [[64, 0]],
                                    "out": [[64, 0]],
                                    "x": [[64, 0]],
                                },
                                "PieceInfo": [
                                    {
                                        "key_": "p0",
                                        "dimToSize_": {
                                            "mb": 64 if transpose_0_2 else 1,
                                            "out": 64,
                                            "x": 1 if transpose_0_2 else 64,
                                        },
                                        "validGap_": {
                                            "out": [[64, 0]],
                                            "mb": [
                                                [
                                                    64 if transpose_0_2 else 1,
                                                    0,
                                                ]
                                            ],
                                            "x": [
                                                [
                                                    1 if transpose_0_2 else 64,
                                                    0,
                                                ]
                                            ],
                                        },
                                        "PlacementInfo": [
                                            {
                                                "type": "hbm",
                                                "memId": [-1],
                                                "startAddr": [
                                                    pointers[inputs[0]["name"]] // 128
                                                ],
                                            },
                                            {
                                                "type": "lx",
                                                "memId": [0],
                                                "startAddr": [0],
                                            },
                                        ],
                                    },
                                    {
                                        "key_": "p1",
                                        "dimToSize_": {
                                            "mb": 64 if transpose_0_2 else 1,
                                            "out": 64,
                                            "x": 1 if transpose_0_2 else 64,
                                        },
                                        "validGap_": {
                                            "out": [[64, 0]],
                                            "mb": [
                                                [
                                                    64 if transpose_0_2 else 1,
                                                    0,
                                                ]
                                            ],
                                            "x": [
                                                [
                                                    1 if transpose_0_2 else 64,
                                                    0,
                                                ]
                                            ],
                                        },
                                        "PlacementInfo": [
                                            {
                                                "type": "hbm",
                                                "memId": [-1],
                                                "startAddr": [0],
                                            },
                                            {
                                                "type": "lx",
                                                "memId": [0],
                                                "startAddr": [8192],
                                            },
                                        ],
                                    },
                                ],
                                "hbmStartAddress_": pointers[inputs[0]["name"]] // 128,
                            },
                            {
                                "pdsName_": "pds0",
                                "wordLength": 2,
                                "dataformat": "SEN169_FP16",
                                "layoutDimOrder_": ["out", "mb", "x"]
                                if transpose_0_2
                                else ["mb", "x", "out"],
                                "stickDimOrder_": ["mb" if transpose_0_2 else "x"],
                                "dimToLayoutSize_": {
                                    "mb": dimensions[0],
                                    "out": dimensions[-1],
                                    "x": dimensions[1],
                                },
                                "dimToStickSize_": {"mb": 64}
                                if transpose_0_2
                                else {"x": 64},
                                "validGap_": {
                                    "mb": [[64, 0]],
                                    "out": [[64, 0]],
                                    "x": [[64, 0]],
                                },
                                "PieceInfo": [
                                    {
                                        "key_": "p0",
                                        "dimToSize_": {
                                            "mb": 64 if transpose_0_2 else 1,
                                            "out": 64,
                                            "x": 1 if transpose_0_2 else 64,
                                        },
                                        "validGap_": {
                                            "out": [[64, 0]],
                                            "mb": [
                                                [
                                                    64 if transpose_0_2 else 1,
                                                    0,
                                                ]
                                            ],
                                            "x": [
                                                [
                                                    1 if transpose_0_2 else 64,
                                                    0,
                                                ]
                                            ],
                                        },
                                        "PlacementInfo": [
                                            {
                                                "type": "hbm",
                                                "memId": [-1],
                                                "startAddr": [
                                                    pointers[outputs[0]["name"]] // 128
                                                ],
                                            },
                                            {
                                                "type": "lx",
                                                "memId": [0],
                                                "startAddr": [16384],
                                            },
                                        ],
                                    },
                                    {
                                        "key_": "p1",
                                        "dimToSize_": {
                                            "mb": 64 if transpose_0_2 else 1,
                                            "out": 64,
                                            "x": 1 if transpose_0_2 else 64,
                                        },
                                        "validGap_": {
                                            "out": [[64, 0]],
                                            "mb": [
                                                [
                                                    64 if transpose_0_2 else 1,
                                                    0,
                                                ]
                                            ],
                                            "x": [
                                                [
                                                    1 if transpose_0_2 else 64,
                                                    0,
                                                ]
                                            ],
                                        },
                                        "PlacementInfo": [
                                            {
                                                "type": "hbm",
                                                "memId": [-1],
                                                "startAddr": [0],
                                            },
                                            {
                                                "type": "lx",
                                                "memId": [0],
                                                "startAddr": [24576],
                                            },
                                        ],
                                    },
                                ],
                                "hbmStartAddress_": pointers[outputs[0]["name"]] // 128,
                            },
                        ],
                        "op": {
                            "name": "ReStickifyOpWithPTHBM",
                            "coreIDtoANInfo": {
                                "0": {
                                    "loopCount": {
                                        "out": dimensions[0] // 64
                                        if transpose_0_2
                                        else dimensions[-1] // 64,
                                        "mb": dimensions[-1] // 64
                                        if transpose_0_2
                                        else dimensions[0],
                                        "x": dimensions[1]
                                        if transpose_0_2
                                        else dimensions[1] // 64,
                                    },
                                    "loopCountL3SU": {
                                        "out": dimensions[0] // 64
                                        if transpose_0_2
                                        else dimensions[-1] // 64,
                                        "mb": dimensions[-1] // 64
                                        if transpose_0_2
                                        else dimensions[0],
                                        "x": dimensions[1]
                                        if transpose_0_2
                                        else dimensions[1] // 64,
                                    },
                                    "addr_info_": {
                                        "l3lu": {
                                            "type_": "stride",
                                            "offset_": {
                                                "mb": dimensions[0]
                                                if transpose_0_2
                                                else 1,
                                                "out": 64
                                                if transpose_0_2
                                                else dimensions[0],
                                                "x": dimensions[-1]
                                                * dimensions[0]
                                                // 64
                                                if transpose_0_2
                                                else dimensions[0] * dimensions[-1],
                                            },
                                        },
                                        "l3su": {
                                            "type_": "stride",
                                            "offset_": {
                                                "mb": 64 if transpose_0_2 else 1,
                                                "out": dimensions[-1]
                                                if transpose_0_2
                                                else dimensions[0] * dimensions[1],
                                                "x": dimensions[-1]
                                                * dimensions[0]
                                                // 64
                                                if transpose_0_2
                                                else dimensions[0],
                                            },
                                        },
                                    },
                                    "inpPieceOrder": [
                                        f"p{i % 2}"
                                        for i in range(
                                            dimensions[0]
                                            * dimensions[1]
                                            * dimensions[-1]
                                            // (64 * 64)
                                        )
                                    ],
                                    "outPieceOrder": [
                                        f"p{i % 2}"
                                        for i in range(
                                            dimensions[0]
                                            * dimensions[1]
                                            * dimensions[-1]
                                            // (64 * 64)
                                        )
                                    ],
                                }
                            },
                            "numClToUse": 1,
                            "cl0ToLxOffsetLU": 0,
                            "cl0ToLxOffsetSU": 0,
                        },
                    }
                }
            ],
        }
    }


def generate_slice(pointers, *, op, dimensions, inputs, outputs, **kwargs):
    return {
        "reshape": {
            "numCoresUsed_": 1,
            "dscs_": [],
            "coreIdToDscSchedule": {"0": [[0, -1, 0, 0]]},
            "datadscs_": [
                {
                    "reshape": {
                        "coreIdsUsed_": [0],
                        "dimPool_": ["mb", "out"],
                        "primaryDs_": [{"name_": "pds0", "dimNames": ["mb", "out"]}],
                        "labeledDs_": [
                            {
                                "pdsName_": "pds0",
                                "wordLength": 2,
                                "dataformat": "SEN169_FP16",
                                "layoutDimOrder_": ["mb", "out"],
                                "stickDimOrder_": ["out"],
                                "dimToLayoutSize_": {
                                    "mb": 64,
                                    "out": dimensions[0],
                                },
                                "dimToStickSize_": {"out": 64},
                                "validGap_": {
                                    "mb": [[64, 0]],
                                    "out": [[dimensions[0], 0]],
                                },
                                "PieceInfo": [
                                    {
                                        "key_": f"p{i}",
                                        "dimToSize_": {"mb": 1, "out": 64},
                                        "validGap_": {
                                            "mb": [[1, 0]],
                                            "out": [[64, 0]],
                                        },
                                        "PlacementInfo": [
                                            {
                                                "type": "hbm",
                                                "memId": [-1],
                                                "startAddr": [
                                                    pointers[inputs[0]["name"]] // 128
                                                ],
                                            },
                                            {
                                                "type": "lx",
                                                "memId": [0],
                                                "startAddr": [0],
                                            },
                                        ],
                                    }
                                    for i in range(dimensions[0] // 64)
                                ],
                                "hbmStartAddress_": pointers[inputs[0]["name"]] // 128,
                            },
                            {
                                "pdsName_": "pds0",
                                "wordLength": 2,
                                "dataformat": "SEN169_FP16",
                                "layoutDimOrder_": ["mb", "out"],
                                "stickDimOrder_": ["out"],
                                "dimToLayoutSize_": {
                                    "mb": 1,
                                    "out": dimensions[0],
                                },
                                "dimToStickSize_": {"out": 64},
                                "validGap_": {
                                    "mb": [[1, 0]],
                                    "out": [[dimensions[0], 0]],
                                },
                                "PieceInfo": [
                                    {
                                        "key_": f"p{i}",
                                        "dimToSize_": {"mb": 1, "out": 64},
                                        "validGap_": {
                                            "mb": [[1, 0]],
                                            "out": [[64, 0]],
                                        },
                                        "PlacementInfo": [
                                            {
                                                "type": "hbm",
                                                "memId": [-1],
                                                "startAddr": [
                                                    pointers[outputs[0]["name"]] // 128
                                                ],
                                            },
                                            {
                                                "type": "lx",
                                                "memId": [0],
                                                "startAddr": [16384],
                                            },
                                        ],
                                    }
                                    for i in range(dimensions[0] // 64)
                                ],
                                "hbmStartAddress_": pointers[outputs[0]["name"]] // 128,
                            },
                        ],
                        "op": {
                            "name": "STCDPOpHBM",
                            "gtrIdsUsed": [],
                            "coreIDtoANInfo": {
                                "0": {
                                    "loopCount": {
                                        "out": dimensions[0] // 64,
                                        "mb": 1,
                                    },
                                    "loopCountL3SU": {},
                                    "addr_info_": {
                                        "l3lu": {
                                            "type_": "stride",
                                            "offset_": {
                                                "mb": 1,
                                                "out": 64,
                                            },
                                        },
                                        "l3su": {
                                            "type_": "stride",
                                            "offset_": {
                                                "mb": 1,
                                                "out": 1,
                                            },
                                        },
                                    },
                                    "inpPieceOrder": [
                                        f"p{i}" for i in range(dimensions[0] // 64)
                                    ],
                                    "outPieceOrder": [
                                        f"p{i}" for i in range(dimensions[0] // 64)
                                    ],
                                }
                            },
                            "numClToUse": 1,
                            "cl0ToLxOffsetLU": 0,
                            "cl0ToLxOffsetSU": 0,
                        },
                    }
                }
            ],
        }
    }


def generate_transpose_4d_stick(
    pointers, *, op, dimensions, inputs, outputs, transposed_dims, **kwargs
):
    transpose_0_3 = 0 in transposed_dims
    transpose_2_3 = 2 in transposed_dims
    input_dtype = inputs[0]["device_layout"].device_dtype
    word_length = num_bytes(input_dtype)
    data_format = input_dtype.name
    elems_per_stick = input_dtype.elems_per_stick()
    piece_count = (
        dimensions[0]
        * dimensions[1]
        * dimensions[-1]
        * dimensions[2]
        // (elems_per_stick * elems_per_stick)
    )
    valid_gaps = {
        "mb": [[dimensions[0], 0]],
        "out": [[dimensions[-1], 0]],
        "x": [[dimensions[1], 0]],
        "y": [[dimensions[2], 0]],
    }
    if transpose_0_3:
        input_layout = ["mb", "out", "x", "y"]
        output_layout = ["out", "mb", "x", "y"]
        output_stick = {"mb": elems_per_stick}
        dim_map = {
            "mb": dimensions[0],
            "out": dimensions[-1],
            "x": dimensions[1],
            "y": dimensions[2],
        }
        l3su_offsets = {
            "mb": elems_per_stick,
            "out": dimensions[-1],
            "y": dimensions[0] * dimensions[1] * dimensions[-1] // elems_per_stick,
            "x": dimensions[-1] * dimensions[0] // elems_per_stick,
        }
        l3lu_offsets = {
            "mb": dimensions[0],
            "out": elems_per_stick,
            "y": dimensions[0] * dimensions[1] * dimensions[-1] // elems_per_stick,
            "x": dimensions[-1] * dimensions[0] // elems_per_stick,
        }
        loop_counts = {
            "mb": dimensions[-1] // elems_per_stick,
            "out": dimensions[0] // elems_per_stick,
            "x": dimensions[1],
            "y": dimensions[2],
        }
        piece_sizes = {"mb": elems_per_stick, "out": elems_per_stick, "x": 1, "y": 1.0}
        piece_valid_gaps = {
            "mb": [[piece_sizes["mb"], 0]],
            "out": [[piece_sizes["out"], 0]],
            "x": [[piece_sizes["x"], 0]],
            "y": [[piece_sizes["y"], 0]],
        }
    elif transpose_2_3:
        input_layout = ["mb", "out", "y", "x"]
        output_layout = ["mb", "y", "out", "x"]
        output_stick = {"y": elems_per_stick}
        dim_map = {
            "mb": dimensions[0],
            "out": dimensions[-1],
            "x": dimensions[1],
            "y": dimensions[2],
        }
        l3su_offsets = {
            "mb": 1,
            "out": dimensions[0],
            "y": dimensions[2] * dimensions[0],
            "x": dimensions[0] * dimensions[-1] * dimensions[2] // elems_per_stick,
        }
        l3lu_offsets = {
            "mb": 1,
            "out": dimensions[-1] * dimensions[0],
            "y": dimensions[0],
            "x": dimensions[0] * dimensions[-1] * dimensions[2] // elems_per_stick,
        }
        loop_counts = {
            "mb": dimensions[0],
            "out": dimensions[2] // elems_per_stick,
            "x": dimensions[1],
            "y": dimensions[-1] // elems_per_stick,
        }
        piece_sizes = {
            "mb": 1,
            "out": elems_per_stick,
            "x": 1,
            "y": elems_per_stick,
        }
        piece_valid_gaps = {
            "mb": [[piece_sizes["mb"], 0]],
            "out": [[piece_sizes["out"], 0]],
            "x": [[piece_sizes["x"], 0]],
            "y": [[piece_sizes["y"], 0]],
        }
    else:  # transpose_1_3
        input_layout = ["mb", "out", "y", "x"]
        output_layout = ["mb", "x", "y", "out"]
        output_stick = {"x": elems_per_stick}
        dim_map = {
            "mb": dimensions[0],
            "out": dimensions[-1],
            "x": dimensions[1],
            "y": dimensions[2],
        }
        l3su_offsets = {
            "mb": 1,
            "out": dimensions[0],
            "x": dimensions[1] * dimensions[0] * dimensions[2],
            "y": dimensions[0] * dimensions[1] // elems_per_stick,
        }
        l3lu_offsets = {
            "mb": 1,
            "x": dimensions[0],
            "out": dimensions[-1] * dimensions[0] * dimensions[2],
            "y": dimensions[0] * dimensions[-1] // elems_per_stick,
        }
        loop_counts = {
            "mb": dimensions[0],
            "out": dimensions[1] // elems_per_stick,
            "y": dimensions[2],
            "x": dimensions[-1] // elems_per_stick,
        }
        piece_sizes = {
            "mb": 1,
            "out": elems_per_stick,
            "y": 1,
            "x": elems_per_stick,
        }
        piece_valid_gaps = {
            "mb": [[piece_sizes["mb"], 0]],
            "out": [[piece_sizes["out"], 0]],
            "x": [[piece_sizes["x"], 0]],
            "y": [[piece_sizes["y"], 0]],
        }

    return {
        "reshape": {
            "numCoresUsed_": 1,
            "dscs_": [],
            "datadscs_": [
                {
                    "reshape": {
                        "coreIdsUsed_": [0],
                        "dimPool_": ["mb", "out", "x", "y"],
                        "primaryDs_": [
                            {
                                "name_": "pds0",
                                "dimNames": ["mb", "out", "y", "x"],
                            }
                        ],
                        "labeledDs_": [
                            {
                                "pdsName_": "pds0",
                                "wordLength": word_length,
                                "dataformat": data_format,
                                "layoutDimOrder_": input_layout,
                                "stickDimOrder_": ["out"],
                                "dimToLayoutSize_": dim_map,
                                "dimToStickSize_": {"out": 64},
                                "validGap_": valid_gaps,
                                "PieceInfo": [
                                    {
                                        "key_": "p0",
                                        "dimToSize_": piece_sizes,
                                        "validGap_": piece_valid_gaps,
                                        "PlacementInfo": [
                                            {
                                                "type": "hbm",
                                                "memId": [-1],
                                                "startAddr": [
                                                    pointers[inputs[0]["name"]] // 128
                                                ],
                                            },
                                            {
                                                "type": "lx",
                                                "memId": [0],
                                                "startAddr": [0],
                                            },
                                        ],
                                    },
                                    {
                                        "key_": "p1",
                                        "dimToSize_": piece_sizes,
                                        "validGap_": piece_valid_gaps,
                                        "PlacementInfo": [
                                            {
                                                "type": "hbm",
                                                "memId": [-1],
                                                "startAddr": [0],
                                            },
                                            {
                                                "type": "lx",
                                                "memId": [0],
                                                "startAddr": [8192],
                                            },
                                        ],
                                    },
                                ],
                                "hbmStartAddress_": pointers[inputs[0]["name"]] // 128,
                            },
                            {
                                "pdsName_": "pds0",
                                "wordLength": word_length,
                                "dataformat": data_format,
                                "layoutDimOrder_": output_layout,
                                "stickDimOrder_": list(output_stick.keys()),
                                "dimToLayoutSize_": dim_map,
                                "dimToStickSize_": output_stick,
                                "validGap_": valid_gaps,
                                "PieceInfo": [
                                    {
                                        "key_": "p0",
                                        "dimToSize_": piece_sizes,
                                        "validGap_": piece_valid_gaps,
                                        "PlacementInfo": [
                                            {
                                                "type": "hbm",
                                                "memId": [-1],
                                                "startAddr": [
                                                    pointers[outputs[0]["name"]] // 128
                                                ],
                                            },
                                            {
                                                "type": "lx",
                                                "memId": [0],
                                                "startAddr": [16384],
                                            },
                                        ],
                                    },
                                    {
                                        "key_": "p1",
                                        "dimToSize_": piece_sizes,
                                        "validGap_": piece_valid_gaps,
                                        "PlacementInfo": [
                                            {
                                                "type": "hbm",
                                                "memId": [-1],
                                                "startAddr": [0],
                                            },
                                            {
                                                "type": "lx",
                                                "memId": [0],
                                                "startAddr": [24576],
                                            },
                                        ],
                                    },
                                ],
                                "hbmStartAddress_": pointers[outputs[0]["name"]] // 128,
                            },
                        ],
                        "op": {
                            "name": "ReStickifyOpWithPTHBM",
                            "coreIDtoANInfo": {
                                "0": {
                                    "loopCount": loop_counts,
                                    "loopCountL3SU": loop_counts,
                                    "addr_info_": {
                                        "l3lu": {
                                            "type_": "stride",
                                            "offset_": l3lu_offsets,
                                        },
                                        "l3su": {
                                            "type_": "stride",
                                            "offset_": l3su_offsets,
                                        },
                                    },
                                    "inpPieceOrder": [
                                        f"p{i % 2}" for i in range(piece_count)
                                    ],
                                    "outPieceOrder": [
                                        f"p{i % 2}" for i in range(piece_count)
                                    ],
                                }
                            },
                            "numClToUse": 1,
                            "cl0ToLxOffsetLU": 0,
                            "cl0ToLxOffsetSU": 0,
                        },
                    }
                }
            ],
        }
    }


def generate_identity(pointers, *, op, dimensions, inputs, outputs, **kwargs):
    tensors = inputs + outputs
    input_dtype = inputs[0]["device_layout"].device_dtype
    data_format = input_dtype

    ndim = len(dimensions)

    cores = 1

    # Get operation dim map from the tensor that represents the operation space
    op_dims_tensor = inputs[0]
    dl = op_dims_tensor["device_layout"]
    dim_map = dl.dim_map[::-1][1:]
    dim_labels = INPUT_DIM_LABELS[: ndim - 1] + OUTPUT_DIM_LABELS[:1]
    dim_splits = [1] * (ndim - 1) + [cores]

    # Obtain (padded) dimensions of the op from a spyre tensor layout
    padded_op_dimensions = [
        get_device_size(host_dim, op_dims_tensor) for host_dim in range(ndim)
    ]

    dim_infos = DimInfos(
        dim_map,
        dim_labels,
        dimensions,
        padded_op_dimensions,
        dim_splits,
    )

    layouts = create_tensor_specific_layouts(
        tensors, dim_infos, op, op_dims_tensor=op_dims_tensor
    )

    # Compute the stick label from the op tensor.
    op_stick_labels = dim_infos.get_tensor_stick_dim_labels(op_dims_tensor)

    core_id_to_wk_slice = {}
    for i in range(cores):
        core_id_to_wk_slice[str(i)] = {
            str(s): i if s in op_stick_labels else 0 for s in dim_labels
        }

    return {
        "identity": {
            "sdscFoldProps_": [{"factor_": 1, "label_": "time"}],
            "sdscFolds_": {
                "dim_prop_func": [{"Affine": {"alpha_": 1, "beta_": 0}}],
                "dim_prop_attr": [{"factor_": 1, "label_": "time"}],
                "data_": {"[0]": "0"},
            },
            "coreFoldProp_": {"factor_": cores, "label_": "core"},
            "coreletFoldProp_": {"factor_": 1, "label_": "corelet"},
            "numCoresUsed_": cores,
            "coreIdToDsc_": {str(c): 0 for c in range(cores)},
            "numWkSlicesPerDim_": {
                di.label: di.nsplits for di in dim_infos.get_op_infos()
            },
            "coreIdToWkSlice_": core_id_to_wk_slice,
            "coreIdToDscSchedule": {str(c): [[-1, 0, 0, 0]] for c in range(cores)},
            "dscs_": [
                {
                    op: {
                        "numCoresUsed_": cores,
                        "numCoreletsUsed_": 1,
                        "coreIdsUsed_": [c for c in range(cores)],
                        "N_": {
                            "name_": "n",
                            **{
                                di.label + "_": di.padded_size
                                for di in dim_infos.get_op_infos()
                            },  # dim sizes before split
                        },
                        "dataStageParam_": {
                            "0": {
                                "ss_": {
                                    "name_": "core",
                                    **{
                                        di.label + "_": di.split_size
                                        for di in dim_infos.get_op_infos()
                                    },
                                },
                                "el_": {
                                    "name_": "core",
                                    **{
                                        di.label + "_": di.split_size
                                        for di in dim_infos.get_op_infos()
                                    },
                                },
                            }
                        },
                        "primaryDsInfo_": {
                            name: {
                                "layoutDimOrder_": layout_info["layout_order"],
                                "stickDimOrder_": layout_info["stick_dim_order"],
                                "stickSize_": [data_format.elems_per_stick()],
                            }
                            for name, layout_info in layouts.items()
                        },
                        "scheduleTree_": [
                            {
                                "nodeType_": "allocate",
                                "name_": f"allocate-Tensor{i}_{'hbm' if tensor['lx_addr'] is None else 'lx'}",
                                "prev_": "",
                                "ldsIdx_": i,
                                "component_": "hbm"
                                if tensor["lx_addr"] is None
                                else "lx",
                                "layoutDimOrder_": dim_infos.get_tensor_layout_order(
                                    tensor
                                ),
                                "maxDimSizes_": [-1]
                                * len(dim_infos.get_tensor_layout_order(tensor)),
                                "startAddressCoreCorelet_": {
                                    "dim_prop_func": [
                                        {"Map": {}},
                                        {"Const": {}},
                                        {"Const": {}},
                                    ],
                                    "dim_prop_attr": [
                                        {"factor_": cores, "label_": "core"},
                                        {"factor_": 1, "label_": "corelet"},
                                        {"factor_": 1, "label_": "time"},
                                    ],
                                    "data_": {
                                        f"[{c}, 0, 0]": str(
                                            pointers[tensor["name"]]
                                            + c
                                            # calculate the prod of dim sizes
                                            # less significant than chosen split dim i.e. the stick
                                            * math.prod(
                                                dim_infos.get_padded_sizes()[:2]
                                            )
                                            * num_bytes(
                                                tensor["device_layout"].device_dtype
                                            )
                                            // cores
                                        )
                                        if tensor["lx_addr"] is None
                                        else tensor["lx_addr"]
                                        for c in range(cores)
                                    },
                                },
                                "coordinates_": {
                                    "coordInfo": {
                                        di.label: gen_coord_info_value(
                                            size=di.split_size
                                            if (di.scale == 1)
                                            else 1,
                                            nsplits=di.nsplits,
                                            elems_per_stick=tensor[
                                                "device_layout"
                                            ].device_dtype.elems_per_stick(),
                                            is_stick_dim=(di.label in op_stick_labels),
                                            is_stick_reduction=(
                                                di.label in op_stick_labels
                                                and di.scale == -1
                                            ),
                                        )
                                        for di in dim_infos.get_tensor_op_infos(
                                            tensor, op
                                        )
                                    },
                                    "coreIdToWkSlice_": {},
                                },
                            }
                            for i, tensor in enumerate(tensors)
                        ],
                        "labeledDs_": [
                            {
                                "ldsIdx_": i,
                                "dsName_": f"Tensor{i}",
                                "dsType_": tensor["ds_type"],
                                "scale_": [
                                    (
                                        di.scale
                                        # TODO: revisit whether this special case can be removed
                                        #       pending change in deeptools
                                        if not (
                                            di.label in op_stick_labels
                                            and di.scale == -1
                                        )
                                        else -2
                                    )
                                    for di in dim_infos.get_tensor_op_infos(tensor, op)
                                ],
                                "wordLength": num_bytes(
                                    tensor["device_layout"].device_dtype
                                ),
                                "dataFormat_": tensor[
                                    "device_layout"
                                ].device_dtype.name,
                                "memOrg_": {
                                    "hbm": {"isPresent": 1},
                                    "lx": {"isPresent": 1},
                                }
                                if tensor["lx_addr"] is None
                                else {"lx": {"isPresent": 1}},
                            }
                            for i, tensor in enumerate(tensors)
                        ],
                        "constantInfo_": {},
                        "computeOp_": [
                            {
                                "opFuncName": "identity",
                                "exUnit": "sfp",
                                "attributes_": {
                                    "dataFormat_": data_format.name,
                                    "fidelity_": "regular",
                                },
                                "location": "Inner",
                                "inputLabeledDs": [
                                    f"Tensor{i}-idx{i}" for i in range(len(inputs))
                                ],
                                "outputLabeledDs": [
                                    f"Tensor{i}-idx{i}"
                                    for i in range(len(inputs), len(tensors))
                                ],
                            }
                        ],
                    }
                }
            ],
        }
    }


def generate_pad(pointers, *, op, dimensions, inputs, outputs, **kwargs):
    """Emit pad as identity-style op: copy input (small) to output (padded) buffer.

    ``dimensions`` are the *output* (padded) dimensions.  Input dimensions are
    taken from ``inputs[0]["host_size"]``.

    The *operation space* is the input tensor: the DSC iterates over ``N =
    input_dimensions`` elements, reading from the small input buffer and
    writing to the beginning of the larger output buffer.  The backend
    interprets the size mismatch as implicit padding — only ``input_size``
    elements are written; the remaining gap at the end of each dimension is
    left as-is (the backend is expected to zero the output buffer before the
    copy, giving zero-valued padding).

    NOTE: For non-zero fill values, the gap will contain zeros (not the fill
    value) when run through the DSC.  Use the eager CPU path for non-zero fill.
    """
    tensors = inputs + outputs
    input_dtype = inputs[0]["device_layout"].device_dtype
    data_format = input_dtype

    ndim = len(dimensions)
    input_dimensions = list(inputs[0]["host_size"])
    assert ndim == len(input_dimensions), "pad input and output must have same rank"

    cores = 1

    # The *operation space* is the input tensor — we iterate over input
    # elements and copy them to the beginning of the (larger) output buffer.
    op_dims_tensor = inputs[0]
    dl_in = op_dims_tensor["device_layout"]
    dim_map = dl_in.dim_map[::-1][1:]
    dim_labels = INPUT_DIM_LABELS[: ndim - 1] + OUTPUT_DIM_LABELS[:1]
    dim_splits = [1] * (ndim - 1) + [cores]

    # Input (small) dim_infos — defines the iteration / operation space
    padded_input_dimensions = [
        get_device_size(host_dim, inputs[0]) for host_dim in range(ndim)
    ]
    dim_infos_input = DimInfos(
        dim_map,
        dim_labels,
        input_dimensions,
        padded_input_dimensions,
        dim_splits,
    )

    # Output (padded) dim_infos — used for per-tensor allocate / coordinates
    padded_output_dimensions = [
        get_device_size(host_dim, outputs[0]) for host_dim in range(ndim)
    ]
    dim_infos_output = DimInfos(
        dim_map,
        dim_labels,
        dimensions,
        padded_output_dimensions,
        dim_splits,
    )

    # Layouts are computed over the *input* op space
    layouts = create_tensor_specific_layouts(
        tensors, dim_infos_input, op, op_dims_tensor=op_dims_tensor
    )

    # Stick-dimension labels from the input tensor (operation space)
    op_stick_labels = dim_infos_input.get_tensor_stick_dim_labels(op_dims_tensor)
    core_id_to_wk_slice = {}
    for i in range(cores):
        core_id_to_wk_slice[str(i)] = {
            str(s): i if s in op_stick_labels else 0 for s in dim_labels
        }

    def dim_infos_for_tensor(tensor_index):
        """Input tensors use input dim_infos; output tensors use output dim_infos."""
        return dim_infos_input if tensor_index < len(inputs) else dim_infos_output

    # The top-level key is "clone" intentionally: pad reuses the clone / identity
    # SDSC structure but with mismatched input / output buffer sizes.  The
    # backend copies ``N = input_size`` elements from the input buffer to the
    # beginning of the larger output buffer, leaving the gap at the end of each
    # dimension untouched (zero-filled by the backend's output-buffer zeroing).
    # Setting N_ from dim_infos_input ensures the DSC iterates over the correct
    # number of elements — the padded *output* dimensions would be too large.
    return {
        "clone": {
            "sdscFoldProps_": [{"factor_": 1, "label_": "time"}],
            "sdscFolds_": {
                "dim_prop_func": [{"Affine": {"alpha_": 1, "beta_": 0}}],
                "dim_prop_attr": [{"factor_": 1, "label_": "time"}],
                "data_": {"[0]": "0"},
            },
            "coreFoldProp_": {"factor_": cores, "label_": "core"},
            "coreletFoldProp_": {"factor_": 1, "label_": "corelet"},
            "numCoresUsed_": cores,
            "coreIdToDsc_": {str(c): 0 for c in range(cores)},
            # numWkSlicesPerDim_ / N_ / dataStageParam_ all reflect the INPUT
            # (operation) space, not the padded output space.
            "numWkSlicesPerDim_": {
                di.label: di.nsplits for di in dim_infos_input.get_op_infos()
            },
            "coreIdToWkSlice_": core_id_to_wk_slice,
            "coreIdToDscSchedule": {str(c): [[-1, 0, 0, 0]] for c in range(cores)},
            "dscs_": [
                {
                    op: {
                        "numCoresUsed_": cores,
                        "numCoreletsUsed_": 1,
                        "coreIdsUsed_": [c for c in range(cores)],
                        # N_ = number of elements to copy = INPUT dimensions
                        "N_": {
                            "name_": "n",
                            **{
                                di.label + "_": di.padded_size
                                for di in dim_infos_input.get_op_infos()
                            },
                        },
                        "dataStageParam_": {
                            "0": {
                                "ss_": {
                                    "name_": "core",
                                    **{
                                        di.label + "_": di.split_size
                                        for di in dim_infos_input.get_op_infos()
                                    },
                                },
                                "el_": {
                                    "name_": "core",
                                    **{
                                        di.label + "_": di.split_size
                                        for di in dim_infos_input.get_op_infos()
                                    },
                                },
                            }
                        },
                        "primaryDsInfo_": {
                            name: {
                                "layoutDimOrder_": layout_info["layout_order"],
                                "stickDimOrder_": layout_info["stick_dim_order"],
                                "stickSize_": [data_format.elems_per_stick()],
                            }
                            for name, layout_info in layouts.items()
                        },
                        "scheduleTree_": [
                            {
                                "nodeType_": "allocate",
                                "name_": f"allocate-Tensor{i}_{'hbm' if tensor['lx_addr'] is None else 'lx'}",
                                "prev_": "",
                                "ldsIdx_": i,
                                "component_": "hbm"
                                if tensor["lx_addr"] is None
                                else "lx",
                                "layoutDimOrder_": dim_infos_for_tensor(
                                    i
                                ).get_tensor_op_layout_order(tensor, op),
                                "maxDimSizes_": [-1]
                                * len(
                                    dim_infos_for_tensor(
                                        i
                                    ).get_tensor_op_layout_order(tensor, op)
                                ),
                                "startAddressCoreCorelet_": {
                                    "dim_prop_func": [
                                        {"Map": {}},
                                        {"Const": {}},
                                        {"Const": {}},
                                    ],
                                    "dim_prop_attr": [
                                        {"factor_": cores, "label_": "core"},
                                        {"factor_": 1, "label_": "corelet"},
                                        {"factor_": 1, "label_": "time"},
                                    ],
                                    "data_": {
                                        f"[{c}, 0, 0]": str(
                                            pointers[tensor["name"]]
                                            + c
                                            * math.prod(
                                                dim_infos_for_tensor(
                                                    i
                                                ).get_padded_sizes()[:2]
                                            )
                                            * num_bytes(
                                                tensor[
                                                    "device_layout"
                                                ].device_dtype
                                            )
                                            // cores
                                        )
                                        if tensor["lx_addr"] is None
                                        else tensor["lx_addr"]
                                        for c in range(cores)
                                    },
                                },
                                "coordinates_": {
                                    "coordInfo": {
                                        di.label: gen_coord_info_value(
                                            size=di.split_size
                                            if (di.scale == 1)
                                            else 1,
                                            nsplits=di.nsplits,
                                            elems_per_stick=tensor[
                                                "device_layout"
                                            ].device_dtype.elems_per_stick(),
                                            is_stick_dim=(
                                                di.label
                                                in dim_infos_for_tensor(
                                                    i
                                                ).get_tensor_stick_dim_labels(
                                                    tensor
                                                )
                                            ),
                                            is_stick_reduction=(
                                                di.label
                                                in dim_infos_for_tensor(
                                                    i
                                                ).get_tensor_stick_dim_labels(
                                                    tensor
                                                )
                                                and di.scale == -1
                                            ),
                                        )
                                        for di in dim_infos_for_tensor(
                                            i
                                        ).get_tensor_op_infos(tensor, op)
                                    },
                                    "coreIdToWkSlice_": {},
                                },
                            }
                            for i, tensor in enumerate(tensors)
                        ],
                        "labeledDs_": [
                            {
                                "ldsIdx_": i,
                                "dsName_": f"Tensor{i}",
                                "dsType_": tensor["ds_type"],
                                "scale_": [
                                    (
                                        di.scale
                                        if not (
                                            di.label
                                            in dim_infos_for_tensor(
                                                i
                                            ).get_tensor_stick_dim_labels(
                                                tensor
                                            )
                                            and di.scale == -1
                                        )
                                        else -2
                                    )
                                    for di in dim_infos_for_tensor(
                                        i
                                    ).get_tensor_op_infos(tensor, op)
                                ],
                                "wordLength": num_bytes(
                                    tensor["device_layout"].device_dtype
                                ),
                                "dataFormat_": tensor[
                                    "device_layout"
                                ].device_dtype.name,
                                "memOrg_": {
                                    "hbm": {"isPresent": 1},
                                    "lx": {"isPresent": 1},
                                }
                                if tensor["lx_addr"] is None
                                else {"lx": {"isPresent": 1}},
                            }
                            for i, tensor in enumerate(tensors)
                        ],
                        "constantInfo_": {},
                        "computeOp_": [
                            {
                                "opFuncName": "identity",
                                "exUnit": "sfp",
                                "attributes_": {
                                    "dataFormat_": data_format.name,
                                    "fidelity_": "regular",
                                },
                                "location": "Inner",
                                "inputLabeledDs": [
                                    f"Tensor{i}-idx{i}" for i in range(len(inputs))
                                ],
                                "outputLabeledDs": [
                                    f"Tensor{i}-idx{i}"
                                    for i in range(len(inputs), len(tensors))
                                ],
                            }
                        ],
                    }
                }
            ],
        }
    }
