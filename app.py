import os, time
from flask import Flask, request, jsonify
from pyzbar.pyzbar import decode, ZBarSymbol
from PIL import Image, ImageOps, ImageFilter

# ---- Optional OpenCV (ROI + rectification rescue tier) ----
try:
    import cv2, numpy as np
    HAS_CV = True
except Exception:
    HAS_CV = False

# ----------------- ENV TOGGLES (with defaults) -----------------
ENABLE_YELLOW_ROI = os.getenv("ENABLE_YELLOW_ROI", "1") == "1"  # rescue tier
ROT_FINE_SWEEP    = int(os.getenv("ROT_FINE_SWEEP", "20"))      # ±degrees in rescue tier
CLAHE_CLIP        = float(os.getenv("CLAHE_CLIP", "2.0"))
CLAHE_TILE        = int(os.getenv("CLAHE_TILE", "8"))
UPSCALE_TARGET    = int(os.getenv("UPSCALE_TARGET", "1400"))    # min max-dimension for tiers 1–2
FAST_MAX_ROT      = int(os.getenv("FAST_MAX_ROT", "270"))       # 0,90,180,270
SYMBOLS_ENV       = os.getenv("BARCODE_SYMBOLS", "CODE128,CODE39,EAN13,EAN8,UPCA,I25")
EARLY_EXIT        = os.getenv("EARLY_EXIT", "1") == "1"         # stop on first hit
MAX_MS_RESUE      = int(os.getenv("RESCUE_TIMEOUT_MS", "2500")) # safety cap for rescue tier
LOG_METRICS       = os.getenv("LOG_METRICS", "1") == "1"        # print() to stdout

# Map symbol names -> ZBarSymbol
_SYMBOL_MAP = {
    "CODE128": ZBarSymbol.CODE128, "CODE39": ZBarSymbol.CODE39,
    "EAN13": ZBarSymbol.EAN13, "EAN8": ZBarSymbol.EAN8,
    "UPCA": ZBarSymbol.UPCA, "I25": ZBarSymbol.I25,
}
SYMBOLS = [_SYMBOL_MAP[s.strip()] for s in SYMBOLS_ENV.split(",") if s.strip() in _SYMBOL_MAP]

app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    return "ok", 200

# ----------------- Common helpers -----------------
def _scale_up(img: Image.Image, target=UPSCALE_TARGET):
    w, h = img.size
    m = max(w, h)
    if m < target:
        s = target / m
        img = img.resize((int(w*s), int(h*s)), Image.LANCZOS)
    return img

def _dedupe_append(results, seen, found):
    for b in found or []:
        key = (b.type, b.data)
        if key not in seen:
            seen.add(key)
            results.append({"type": b.type, "data": b.data.decode("utf-8", "ignore")})

# ----------------- Tier 1: Fast (generic) -----------------
def tier1_variants(img: Image.Image):
    base = _scale_up(img)
    for rot in (0, 90, 180, FAST_MAX_ROT):
        g = ImageOps.grayscale(base.rotate(rot, expand=True))
        yield g                                  # raw gray
        yield ImageOps.equalize(g)               # mild equalize
        yield ImageOps.autocontrast(g, cutoff=0.5)

# ----------------- Tier 2: Gentle (no hard thresholds) -----------------
def tier2_variants(img: Image.Image):
    base = _scale_up(img, int(UPSCALE_TARGET * 1.2))
    for rot in (0, 90, 180, FAST_MAX_ROT):
        g = ImageOps.grayscale(base.rotate(rot, expand=True))
        yield g.filter(ImageFilter.MedianFilter(3)).filter(
            ImageFilter.UnsharpMask(radius=1, percent=60, threshold=2)
        )
        yield ImageOps.equalize(g.filter(
            ImageFilter.UnsharpMask(radius=1, percent=80, threshold=3)
        ))
        yield g.filter(ImageFilter.GaussianBlur(0.7))
        yield ImageOps.invert(g)

# ----------------- Tier 3 (rescue): ROI + rectification + fine sweep -----------------
def _order_box(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1); rect[0] = pts[np.argmin(s)]; rect[2] = pts[np.argmax(s)]
    d = np.diff(pts, axis=1); rect[1] = pts[np.argmin(d)]; rect[3] = pts[np.argmax(d)]
    return rect

def _warp_rect(image, box, pad=6):
    box = np.array(box, dtype="float32")
    rect = _order_box(box); tl,tr,br,bl = rect
    wA = np.linalg.norm(br-bl); wB = np.linalg.norm(tr-tl); W = int(max(wA,wB))
    hA = np.linalg.norm(tr-br); hB = np.linalg.norm(tl-bl); H = int(max(hA,hB))
    dst = np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (W,H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return cv2.copyMakeBorder(warped, pad,pad,pad,pad, cv2.BORDER_REPLICATE)

def _find_yellow_rois(bgr):
    rois = []
    h,w = bgr.shape[:2]
    scale = 1.0
    if max(h,w) > 2000:
        scale = 2000.0 / max(h,w)
        bgr_s = cv2.resize(bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    else:
        bgr_s = bgr
    hsv = cv2.cvtColor(bgr_s, cv2.COLOR_BGR2HSV)
    lower = np.array([15, 60, 80]); upper = np.array([40, 255, 255])  # broad yellow
    mask = cv2.inRange(hsv, lower, upper)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, 2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, 1)
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        if cv2.contourArea(c) < 2000: continue
        rect = cv2.minAreaRect(c)
        box  = cv2.boxPoints(rect) / scale
        rois.append(_warp_rect(bgr, box))
    return rois

def tier3_rescue_np(rgb):
    """Returns decoded results from rescue tier (or [])."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    targets = []
    if ENABLE_YELLOW_ROI:
        targets = _find_yellow_rois(bgr)
    if not targets:
        targets = [bgr]  # fallback: whole image

    results, seen = [], set()
    t0 = time.time()
    for tgt in targets:
        gray = cv2.cvtColor(tgt, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(CLAHE_TILE, CLAHE_TILE))
        base = clahe.apply(gray)

        for ang in range(-ROT_FINE_SWEEP, ROT_FINE_SWEEP+1, 2):
            if (time.time() - t0) * 1000 > MAX_MS_RESUE:
                return results  # safety cutoff
            M = cv2.getRotationMatrix2D((base.shape[1]/2, base.shape[0]/2), ang, 1.0)
            rot = cv2.warpAffine(base, M, (base.shape[1], base.shape[0]), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            sharp = cv2.addWeighted(rot, 1.5, cv2.GaussianBlur(rot, (0,0), 1.2), -0.5, 0)

            for arr in (rot, sharp):
                pil_img = Image.fromarray(arr)
                _dedupe_append(results, seen, decode(pil_img, symbols=SYMBOLS))
                if EARLY_EXIT and results: return results
    return results

# ----------------- API -----------------
@app.route("/decode", methods=["POST"])
def decode_api():
    debug = request.args.get("debug", "0") == "1"
    if "file" not in request.files:
        return jsonify({"error": "no file"}), 400

    try:
        img = Image.open(request.files["file"].stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"cannot open image: {e}"}), 400

    meta = {"tiers": []}
    t_start = time.time()

    # Tier 1
    t0 = time.time()
    results, seen = [], set()
    for v in tier1_variants(img):
        _dedupe_append(results, seen, decode(v, symbols=SYMBOLS))
        if EARLY_EXIT and results: break
    meta["tiers"].append({"name": "tier1_fast", "ms": int((time.time()-t0)*1000), "hits": len(results)})

    # Tier 2
    if not results:
        t0 = time.time()
        for v in tier2_variants(img):
            _dedupe_append(results, seen, decode(v, symbols=SYMBOLS))
            if EARLY_EXIT and results: break
        meta["tiers"].append({"name": "tier2_gentle", "ms": int((time.time()-t0)*1000), "hits": max(0, len(results))})
    else:
        meta["tiers"].append({"name": "tier2_gentle", "skipped": True})

    # Tier 3 (rescue)
    if not results and HAS_CV:
        t0 = time.time()
        results = tier3_rescue_np(np.array(img))
        meta["tiers"].append({"name": "tier3_rescue_cv", "ms": int((time.time()-t0)*1000), "hits": len(results), "yellow_roi": ENABLE_YELLOW_ROI})
    elif not HAS_CV:
        meta["tiers"].append({"name": "tier3_rescue_cv", "skipped": "opencv_not_available"})

    meta["total_ms"] = int((time.time()-t_start)*1000)

    if LOG_METRICS:
        print(f"[decode] total={meta['total_ms']}ms hits={len(results)} tiers={meta['tiers']}")

    payload = {"barcodes": results}
    if debug:
        payload["meta"] = meta
        payload["config"] = {
            "symbols": SYMBOLS_ENV, "early_exit": EARLY_EXIT, "enable_yellow_roi": ENABLE_YELLOW_ROI,
            "rot_fine_sweep": ROT_FINE_SWEEP, "clahe": {"clip": CLAHE_CLIP, "tile": CLAHE_TILE},
            "upscale_target": UPSCALE_TARGET
        }
    return jsonify(payload)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
