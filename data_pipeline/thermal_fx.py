"""적외선(IR) 열화상 카메라 룩 후처리.

실제 열화상 카메라처럼: 렌더 프레임의 밝기(=온도 proxy)를 열화상 컬러맵에
통째로 매핑한다. 3D 셰이딩이 만든 개체 표면의 밝기 변화가 자연스럽게
색 그라데이션(뜨거운 코어 → 차가운 가장자리)으로 변환된다.

컬러맵 ramp (cold→hot):
  검정 → 파랑 → 인디고 → 보라 → 자홍 → 빨강 → 주황 → 호박 → 노랑 → 흰색

상단에 온도 컬러바(눈금)를 합성한다. 밝기 0~1 → 온도 TEMP_MIN~TEMP_MAX 선형 매핑.
"""
import numpy as np

try:
    from scipy.ndimage import gaussian_filter
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# 열화상 컬러맵 control points (x=정규화 밝기 0~1, rgb 0~1)
# cold(검정/파랑) → hot(노랑/흰색). dead chick은 ~0.22 부근 → 파랑으로 매핑.
_CMAP_X = np.array([0.00, 0.12, 0.22, 0.34, 0.46, 0.58, 0.68, 0.78, 0.88, 0.95, 1.00])
_CMAP_R = np.array([0.00, 0.10, 0.15, 0.35, 0.55, 0.78, 0.92, 1.00, 1.00, 1.00, 1.00])
_CMAP_G = np.array([0.00, 0.10, 0.20, 0.12, 0.10, 0.12, 0.20, 0.48, 0.72, 0.92, 1.00])
_CMAP_B = np.array([0.00, 0.55, 0.80, 0.65, 0.55, 0.38, 0.15, 0.05, 0.05, 0.35, 0.95])

# 컬러바 온도 스케일 (밝기 0 → TEMP_MIN, 밝기 1 → TEMP_MAX)
TEMP_MIN = 15.0   # °C — 차가운 바닥/그늘
TEMP_MAX = 42.0   # °C — 살아있는 닭 체온 수준


def _blur(arr, sigma):
    """gaussian blur. 2D 또는 3D(채널 마지막) 지원. scipy 없으면 박스 fallback."""
    if sigma <= 0:
        return arr
    if _HAS_SCIPY:
        s = (sigma, sigma) if arr.ndim == 2 else (sigma, sigma, 0)
        return gaussian_filter(arr, sigma=s)
    out = arr.copy()
    k = max(1, int(sigma))
    for _ in range(3):
        out = (np.roll(out, k, 0) + np.roll(out, -k, 0)
               + np.roll(out, k, 1) + np.roll(out, -k, 1) + out) / 5.0
    return out


def apply_colormap(intensity):
    """intensity HxW [0,1] → HxWx3 [0,1] 열화상 색."""
    r = np.interp(intensity, _CMAP_X, _CMAP_R)
    g = np.interp(intensity, _CMAP_X, _CMAP_G)
    b = np.interp(intensity, _CMAP_X, _CMAP_B)
    return np.stack([r, g, b], axis=-1)


def _load_font(size):
    """가용한 TTF 폰트 로드, 실패 시 PIL 기본 폰트."""
    from PIL import ImageFont
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def add_colorbar(img, temp_min=TEMP_MIN, temp_max=TEMP_MAX, n_ticks=6):
    """프레임 상단에 온도 컬러바 + 눈금 라벨을 합성. HxWx3 uint8 in/out."""
    from PIL import Image, ImageDraw

    H, W = img.shape[:2]
    pil = Image.fromarray(np.ascontiguousarray(img)).convert("RGB")
    draw = ImageDraw.Draw(pil, "RGBA")

    margin = max(10, W // 36)
    bar_x0, bar_x1 = margin, W - margin
    bar_y0 = max(8, H // 50)
    bar_h = max(10, H // 36)
    bar_w = bar_x1 - bar_x0
    font = _load_font(max(11, H // 38))

    # 가독성용 반투명 어두운 패널
    panel_h = bar_y0 + bar_h + max(20, H // 22)
    draw.rectangle([0, 0, W, panel_h], fill=(0, 0, 0, 140))

    # 그라데이션 바
    grad = np.linspace(0.0, 1.0, bar_w, dtype=np.float32)
    bar_rgb = (apply_colormap(grad) * 255).astype(np.uint8)         # bar_w x 3
    bar_strip = np.tile(bar_rgb[None, :, :], (bar_h, 1, 1))         # bar_h x bar_w x 3
    pil.paste(Image.fromarray(bar_strip), (bar_x0, bar_y0))
    draw.rectangle([bar_x0, bar_y0, bar_x1, bar_y0 + bar_h],
                   outline=(255, 255, 255, 200), width=1)

    # 눈금 + 온도 라벨
    for i in range(n_ticks):
        frac = i / (n_ticks - 1)
        x = bar_x0 + int(round(frac * bar_w))
        temp = temp_min + frac * (temp_max - temp_min)
        draw.line([x, bar_y0 + bar_h, x, bar_y0 + bar_h + 4],
                  fill=(255, 255, 255, 220), width=1)
        label = f"{temp:.0f}"
        tb = draw.textbbox((0, 0), label, font=font)
        tw = tb[2] - tb[0]
        tx = min(max(x - tw // 2, 2), W - tw - 2)
        draw.text((tx, bar_y0 + bar_h + 5), label, font=font, fill=(255, 255, 255, 235))

    # °C 단위 표기 (바 오른쪽 위)
    draw.text((bar_x1 + 2 - 22, max(0, bar_y0 - 2)), "°C", font=font,
              fill=(255, 255, 255, 235))
    # 좌상단 라벨
    draw.text((margin, bar_y0 + bar_h + 5 + 0), "", font=font, fill=(255, 255, 255, 0))

    return np.asarray(pil, dtype=np.uint8)


def thermal_fx(img, blur_sigma=1.1, gain=1.15, gamma=0.85,
               bloom_threshold=0.58, bloom_sigma=8.0, bloom_strength=0.95,
               colorbar=True):
    """HxWx3 uint8 프레임 → 열화상 카메라 룩 uint8 프레임.

    blur_sigma:      카메라 softness (밝기 맵 약한 blur)
    gain/gamma:      온도 매핑 전 밝기 보정 (대비 조절)
    bloom_*:         가장 뜨거운 영역의 발광 번짐
    colorbar:        True면 상단에 온도 컬러바 합성
    """
    f = img.astype(np.float32) / 255.0

    # 1) 밝기 = 온도 proxy (luminance). 파란 개체는 자연히 낮은 온도로 매핑됨.
    lum = 0.21 * f[..., 0] + 0.72 * f[..., 1] + 0.07 * f[..., 2]

    # 2) 카메라 softness
    lum = _blur(lum, blur_sigma)

    # 3) 밝기 보정 후 [0,1] clamp
    lum = np.clip(lum * gain, 0.0, 1.0) ** gamma

    # 4) 열화상 컬러맵 적용 — 셰이딩 밝기 변화가 색 그라데이션으로
    colored = apply_colormap(lum)

    # 5) bloom — 가장 뜨거운(밝은) 영역만 크게 번지게
    hot = np.clip(lum - bloom_threshold, 0.0, 1.0)[..., None]
    bloom = _blur(colored * hot, bloom_sigma)
    out = np.clip(colored + bloom * bloom_strength, 0.0, 1.0)

    frame = (out * 255.0).astype(np.uint8)

    # 6) 온도 컬러바
    if colorbar:
        frame = add_colorbar(frame)

    return frame


if __name__ == "__main__":
    import argparse
    import imageio.v2 as imageio

    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-colorbar", action="store_true")
    args = ap.parse_args()

    src = imageio.imread(args.inp)
    dst = thermal_fx(src[..., :3], colorbar=not args.no_colorbar)
    imageio.imwrite(args.out, dst)
    print(f"saved {args.out}")
