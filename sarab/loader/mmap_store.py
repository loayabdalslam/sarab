"""Zero-copy weight access over safetensors files via mmap.

This is the single biggest lever for running a model larger than RAM: we never read the
whole file. We parse the safetensors header (which records every tensor's dtype, shape
and byte range), mmap the file, and hand out NumPy views that point *directly* into the
mapping. The OS pages in only the bytes a view actually touches, and reclaims them under
memory pressure. The "model" is fully present on disk yet barely resident in RAM.

safetensors layout (see https://github.com/huggingface/safetensors):
    [ 8 bytes little-endian u64 = header_len ]
    [ header_len bytes of UTF-8 JSON ]
    [ raw tensor bytes ... ]
The JSON maps  name -> {"dtype", "shape", "data_offsets": [begin, end]}  where the
offsets are relative to the start of the raw-bytes region (right after the header).
"""

from __future__ import annotations

import json
import mmap
import struct
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Union

import numpy as np

# safetensors dtype string -> (numpy dtype, needs_bf16_upcast)
# numpy has no native bfloat16, so we read it as uint16 and upcast on demand.
_ST_DTYPES: Dict[str, Tuple[np.dtype, bool]] = {
    "F64": (np.dtype("<f8"), False),
    "F32": (np.dtype("<f4"), False),
    "F16": (np.dtype("<f2"), False),
    "BF16": (np.dtype("<u2"), True),
    "I64": (np.dtype("<i8"), False),
    "I32": (np.dtype("<i4"), False),
    "I16": (np.dtype("<i2"), False),
    "I8": (np.dtype("<i1"), False),
    "U8": (np.dtype("<u1"), False),
    "BOOL": (np.dtype("?"), False),
}


def bf16_to_f32(raw_u16: np.ndarray) -> np.ndarray:
    """Upcast bfloat16 (carried as uint16) to float32 losslessly.

    bfloat16 is simply the top 16 bits of a float32, so we widen to uint32, shift left
    16, and reinterpret. No precision is lost going bf16 -> f32.
    """
    u32 = raw_u16.astype(np.uint32) << 16
    return u32.view(np.float32)


class TensorView:
    """A lazily-materialized handle to one tensor inside an mmap'd safetensors file.

    Holds no decoded data until `.array()` is called, so building the full tensor index
    for a 70B model costs almost nothing.
    """

    __slots__ = ("name", "shape", "_np_dtype", "_bf16", "_buf", "_begin", "_end")

    def __init__(
        self,
        name: str,
        shape: Tuple[int, ...],
        np_dtype: np.dtype,
        bf16: bool,
        buf: mmap.mmap,
        begin: int,
        end: int,
    ) -> None:
        self.name = name
        self.shape = shape
        self._np_dtype = np_dtype
        self._bf16 = bf16
        self._buf = buf
        self._begin = begin
        self._end = end

    @property
    def nbytes(self) -> int:
        return self._end - self._begin

    def raw(self) -> np.ndarray:
        """A view straight into the mmap — no copy. Touching it pages in those bytes."""
        flat = np.frombuffer(self._buf, dtype=self._np_dtype, count=self._n_elems(),
                             offset=self._begin)
        return flat.reshape(self.shape)

    def array(self, dtype: Union[str, np.dtype, None] = "float32") -> np.ndarray:
        """Materialize as a real array, upcasting bf16 and casting to `dtype`.

        `dtype=None` returns the native view (zero-copy for non-bf16). Any concrete
        dtype produces an owned copy in that dtype — the caller can mutate freely and the
        mmap pages backing the source can be reclaimed afterwards.
        """
        view = self.raw()
        if self._bf16:
            out = bf16_to_f32(view)
            return out if dtype in (None, "float32", np.float32) else out.astype(dtype)
        if dtype is None:
            return view
        return view.astype(dtype, copy=True)

    def rows(self, indices, dtype: Union[str, np.dtype, None] = "float32") -> np.ndarray:
        """Gather specific rows of a 2-D tensor, paging in ONLY those rows.

        Critical for huge embedding tables: instead of materializing the whole
        [vocab, hidden] matrix to fetch a handful of token rows, we index the mmap view so
        only the touched rows fault into memory. Used for token embeddings every step.
        """
        raw = self.raw()                       # zero-copy view over the mmap
        gathered = raw[indices]                # copies only the selected rows
        if self._bf16:
            out = bf16_to_f32(gathered)
            return out if dtype in (None, "float32", np.float32) else out.astype(dtype)
        if dtype is None:
            return gathered
        return gathered.astype(dtype, copy=True)

    def _n_elems(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        kind = "bf16" if self._bf16 else str(self._np_dtype.name)
        return f"TensorView({self.name!r}, shape={self.shape}, dtype={kind})"


class MmapStore:
    """Indexes one or more safetensors shards and serves zero-copy `TensorView`s.

    Handles sharded checkpoints (model-00001-of-00003.safetensors + index json) and
    single-file checkpoints transparently. Open it once; look tensors up by name.
    """

    def __init__(self, shard_paths: Iterable[Union[str, Path]]) -> None:
        self._mmaps: List[mmap.mmap] = []
        self._files = []
        self._index: Dict[str, TensorView] = {}
        self.metadata: Dict[str, str] = {}
        for p in shard_paths:
            self._open_shard(Path(p))

    # -- construction helpers ------------------------------------------------------
    @classmethod
    def from_dir(cls, model_dir: Union[str, Path]) -> "MmapStore":
        """Open a checkpoint directory, resolving the shard index if present."""
        model_dir = Path(model_dir)
        index_json = model_dir / "model.safetensors.index.json"
        if index_json.is_file():
            with open(index_json, "r", encoding="utf-8") as f:
                weight_map = json.load(f)["weight_map"]
            shards = sorted({model_dir / fname for fname in weight_map.values()})
            return cls(shards)
        single = model_dir / "model.safetensors"
        if single.is_file():
            return cls([single])
        # fall back: any *.safetensors in the dir
        shards = sorted(model_dir.glob("*.safetensors"))
        if not shards:
            raise FileNotFoundError(f"no .safetensors found in {model_dir}")
        return cls(shards)

    def _open_shard(self, path: Path) -> None:
        f = open(path, "rb")
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        self._files.append(f)
        self._mmaps.append(mm)

        (header_len,) = struct.unpack_from("<Q", mm, 0)
        header_json = bytes(mm[8 : 8 + header_len]).decode("utf-8")
        header = json.loads(header_json)
        data_start = 8 + header_len

        for name, info in header.items():
            if name == "__metadata__":
                self.metadata.update(info)
                continue
            np_dtype, is_bf16 = _ST_DTYPES[info["dtype"]]
            begin, end = info["data_offsets"]
            self._index[name] = TensorView(
                name=name,
                shape=tuple(info["shape"]),
                np_dtype=np_dtype,
                bf16=is_bf16,
                buf=mm,
                begin=data_start + begin,
                end=data_start + end,
            )

    # -- access --------------------------------------------------------------------
    def __contains__(self, name: str) -> bool:
        return name in self._index

    def __getitem__(self, name: str) -> TensorView:
        try:
            return self._index[name]
        except KeyError as e:
            raise KeyError(f"tensor {name!r} not in checkpoint") from e

    def get(self, name: str, default=None):
        return self._index.get(name, default)

    def keys(self) -> Iterable[str]:
        return self._index.keys()

    def total_bytes(self) -> int:
        """On-disk size of all tensors (i.e. the full model footprint)."""
        return sum(v.nbytes for v in self._index.values())

    def close(self) -> None:
        for mm in self._mmaps:
            try:
                mm.close()
            except (BufferError, ValueError):
                pass
        for f in self._files:
            f.close()
        self._mmaps.clear()
        self._files.clear()
        self._index.clear()

    def __enter__(self) -> "MmapStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
