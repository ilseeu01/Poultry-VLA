"""LoRA 어댑터 스냅샷 -> 병합 체크포인트 (평가용, merge-on-demand).

사용: python merge_adapter.py <adapter_dir> [out_dir]
- adapter_dir 의 adapter_config.json 에서 base 경로를 읽음.
- adapter_dir 에 dataset_statistics.json 이 있으면 병합 dir로 복사(평가 unnorm_key용).
- out_dir 미지정 시 <adapter_dir>_merged.
평가 후 out_dir 은 삭제해 디스크 회수 권장(15GB).
"""
import json
import os
import shutil
import sys

import torch
from peft import PeftModel
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

adapter_dir = sys.argv[1]
out_dir = sys.argv[2] if len(sys.argv) > 2 else adapter_dir.rstrip("/") + "_merged"

with open(os.path.join(adapter_dir, "adapter_config.json")) as f:
    base = json.load(f)["base_model_name_or_path"]

AutoConfig.register("openvla", OpenVLAConfig)
AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

print(f"[*] base   : {base}")
print(f"[*] adapter: {adapter_dir}")
print(f"[*] out    : {out_dir}")

m = AutoModelForVision2Seq.from_pretrained(
    base, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
)
m = PeftModel.from_pretrained(m, adapter_dir).merge_and_unload()
os.makedirs(out_dir, exist_ok=True)
m.save_pretrained(out_dir)
AutoProcessor.from_pretrained(base, trust_remote_code=True).save_pretrained(out_dir)

src_stats = os.path.join(adapter_dir, "dataset_statistics.json")
if os.path.isfile(src_stats):
    shutil.copy(src_stats, os.path.join(out_dir, "dataset_statistics.json"))
    print("[*] copied dataset_statistics.json")
else:
    print("[!] WARNING: adapter_dir에 dataset_statistics.json 없음 — 평가 unnorm_key 수동 필요")

print("[*] DONE ->", out_dir)
