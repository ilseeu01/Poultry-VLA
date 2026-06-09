"""진단용: 베이스 openvla-7b-finetuned-libero-object 에 v2 LoRA 어댑터를 merge 하고,
blue_chick_thermal 로 키잉된 dataset_statistics.json 을 병합 디렉토리에 작성한다.

남은 자산(484MB LoRA 어댑터)만으로 평가 가능한 병합 체크포인트를 복원하는 용도.
"""
import json
import os
import shutil

import torch
from peft import PeftModel
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

BASE = "/home/capstone/openvla_ckpts/openvla-7b-finetuned-libero-object"
ADAPTER = (
    "/home/capstone/openvla_ckpts/adapter-tmp/"
    "openvla-7b-finetuned-libero-object+blue_chick_thermal+b16+lr-0.0005+lora-r32+dropout-0.0--v2--image_aug"
)
OUT = "/home/capstone/openvla_ckpts/merged-v2-diag"
RLDS_STATS = (
    "/home/capstone/tensorflow_datasets/blue_chick_thermal/1.0.0/"
    "dataset_statistics_b27092559b8286f6201e9690b61aa996e29875d51876924b284f657cbc2fcebb.json"
)
UNNORM_KEY = "blue_chick_thermal"

# Register OpenVLA with HF Auto classes
AutoConfig.register("openvla", OpenVLAConfig)
AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

print(f"[*] Loading base: {BASE}")
base = AutoModelForVision2Seq.from_pretrained(
    BASE, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
)
print(f"[*] Applying LoRA adapter: {ADAPTER}")
merged = PeftModel.from_pretrained(base, ADAPTER)
merged = merged.merge_and_unload()

os.makedirs(OUT, exist_ok=True)
print(f"[*] Saving merged model -> {OUT}")
merged.save_pretrained(OUT)

print("[*] Saving processor")
processor = AutoProcessor.from_pretrained(BASE, trust_remote_code=True)
processor.save_pretrained(OUT)

# dataset_statistics.json: RLDS builder 가 만든 평면 통계를 unnorm_key 로 감싼다
print("[*] Writing keyed dataset_statistics.json")
with open(RLDS_STATS) as f:
    flat = json.load(f)
# flat 이 이미 {action, proprio, ...} 평면이면 감싸고, 이미 키잉돼 있으면 그대로
if "action" in flat:
    keyed = {UNNORM_KEY: flat}
else:
    keyed = flat
with open(os.path.join(OUT, "dataset_statistics.json"), "w") as f:
    json.dump(keyed, f, indent=2)

print("[*] DONE. merged checkpoint at:", OUT)
print("    unnorm_key:", list(keyed.keys()))
