#!/usr/bin/env python
"""robosuite 시각 버퍼 패치: MjvScene maxgeom 1000 -> 5000.

양계장 씬은 1000개의 짚(straw) 비주얼 geom을 렌더하므로 robosuite 기본
maxgeom=1000 으로는 버퍼가 부족하다. 이 스크립트는 설치된 robosuite 의
utils/binding_utils.py 를 찾아 maxgeom 값을 5000 으로 올린다 (idempotent).

사용: python patches/apply_robosuite_patch.py
"""
import os
import re
import sys

try:
    import robosuite
except ImportError:
    sys.exit("robosuite 가 설치되어 있지 않습니다. 먼저 requirements 를 설치하세요.")

target = os.path.join(os.path.dirname(robosuite.__file__), "utils", "binding_utils.py")
if not os.path.isfile(target):
    sys.exit(f"binding_utils.py 를 찾을 수 없음: {target}")

src = open(target).read()

if "maxgeom=5000" in src:
    print(f"[skip] 이미 패치됨: {target}")
    sys.exit(0)

new = re.sub(r"maxgeom\s*=\s*1000", "maxgeom=5000", src)
if new == src:
    print(f"[warn] 'maxgeom=1000' 패턴을 찾지 못함. 수동 확인 필요: {target}")
    sys.exit(1)

# 백업 후 기록
open(target + ".pvla.bak", "w").write(src)
open(target, "w").write(new)
print(f"[ok] maxgeom 1000 -> 5000 패치 완료: {target}")
print(f"     백업: {target}.pvla.bak")
