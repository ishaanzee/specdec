#!/usr/bin/env python3
"""Splice a Qwen3.5/3.8 MTP sidecar into a main GGUF that was converted with --no-mtp.

  splice_mtp_gguf.py MAIN.gguf mtp-SIDECAR.gguf OUT.gguf

The trunk tensors and metadata are copied byte-for-byte from MAIN. From the sidecar we
take only the MTP block (blk.<n_layer>.*; its own token_embd/output/output_norm copies are
dropped, the main model's are shared) plus the metadata that describes it:
<arch>.block_count (n_layer + 1), <arch>.nextn_predict_layers and the per-layer
<arch>.attention.recurrent_layers array (65 entries, MTP layer = not recurrent).

Result: one GGUF that `llama-server -m OUT.gguf --spec-type draft-mtp` can use without -md,
and whose no-spec output is identical to MAIN (MTP tensors are skipped at load time
unless draft-mtp is requested).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "vendor" / "llama.cpp" / "gguf-py"))
import gguf  # noqa: E402


def field_value(field):
    val_type = field.types[0]
    sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
    return field.contents(), val_type, sub_type


def main():
    main_p, side_p, out_p = map(Path, sys.argv[1:4])
    rm = gguf.GGUFReader(main_p)
    rs = gguf.GGUFReader(side_p)
    arch = rm.fields["general.architecture"].contents()
    assert rs.fields["general.architecture"].contents() == arch
    n_layer = int(rm.fields[f"{arch}.block_count"].contents())
    n_side = int(rs.fields[f"{arch}.block_count"].contents())
    n_next = int(rs.fields[f"{arch}.nextn_predict_layers"].contents())
    assert f"{arch}.nextn_predict_layers" not in rm.fields, "main GGUF already has MTP layers"
    assert n_side == n_layer + n_next, (n_side, n_layer, n_next)

    override = {f"{arch}.block_count", f"{arch}.nextn_predict_layers", f"{arch}.attention.recurrent_layers"}
    w = gguf.GGUFWriter(out_p, arch=arch, endianess=rm.endianess)
    for field in rm.fields.values():
        if field.name == "general.architecture" or field.name.startswith("GGUF.") or field.name in override:
            continue
        v, t, st = field_value(field)
        w.add_key_value(field.name, v, t, sub_type=st)
    for name in sorted(override):
        if name in rs.fields:
            v, t, st = field_value(rs.fields[name])
            w.add_key_value(name, v, t, sub_type=st)

    mtp_prefixes = tuple(f"blk.{i}." for i in range(n_layer, n_side))
    side_tensors = [t for t in rs.tensors if t.name.startswith(mtp_prefixes)]
    tensors = list(rm.tensors) + side_tensors
    for t in tensors:
        w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    for i, t in enumerate(tensors):
        w.write_tensor_data(t.data, tensor_endianess=rm.endianess)
        if i % 100 == 0:
            print(f"{i}/{len(tensors)} {t.name}", flush=True)
    w.close()
    print(f"wrote {out_p}: {len(rm.tensors)} trunk + {len(side_tensors)} MTP tensors = {len(tensors)}")


if __name__ == "__main__":
    main()
