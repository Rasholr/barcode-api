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
ENABLE_YELLOW_ROI = os.getenv("ENABLE_YELLOW_ROI", "1") == "1"  # now a *hint*, not a requirement
ROT_FINE_SWEEP    = int(os.getenv("ROT_FINE_SWEEP", "20"))      # ±degrees in rescue tier
CLAHE_CLIP        = float(os.getenv("CLAHE_CLIP", "2.0"))
CLAHE_TILE        = int(os.getenv("CLAHE_TILE", "8"))
UPSCALE_TARGET    = int(os.getenv("UPSCALE_TARGET", "1400"))    # min max-dimension for tiers 1–2
FAST_MAX_ROT      = int(os.getenv("FAST_MAX_ROT", "270"))       # 0,90,180,270
SYMBOLS_ENV       = os.getenv("BARCODE_SYMBOLS", "CODE128,EAN13,EAN8,UPCA,CODE39")
EARLY_EXIT        = os.getenv("EARLY_EXIT", "1") == "1"         # stop on first hit
MAX_MS_RESUE      = int(os.getenv("RESCUE_TIMEOUT_MS", "1200")) # safety cap for rescue tier
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
def _scale_cap(img: Image.Image, target=UPSCALE_TARGET):
    """Cap overly large images and mildly upscale tiny ones to ~1000–target px max side."""
    w, h = img.size
    m = max(w, h)
    if m > target:
        s = target / m
        return img.resize((int(w*s), int(h*s)), Image.LANCZOS)
    if m < 1000:
        s = 1000 / m
        return img.resize((int(w*s), int(h*s)), Image.LANCZOS)
    return img

def _dedupe_append(results, seen, found):
    for b in found or []:
        key = (b.type, b.data)
        if key not in seen:
            seen.add(key)
            results.append({"type": b.type, "data": b.data.decode("utf-8", "ignore")})

# ----------------- Tier 1: Fast (generic, minimal transforms) -----------------
def tier1_variants(img: Image.Image):
    base = _scale_cap(img).convert("RGB")
    for rot in (0, 90, 180, FAST_MAX_ROT):
        r = base.rotate(rot, expand=True)
        g = ImageOps.grayscale(r)
        yield g  # raw gray
        yield g.filter(ImageFilter.UnsharpMask(radius=1, percent=80, threshold=3))  # mild unsharp
        yield ImageOps.autocontrast(g, cutoff=1.0)  # gentle
        yield ImageOps.invert(g)  # white-on-black labels

# ----------------- Tier 2: Gentle (still global; no hard thresholds) -----------------
def tier2_variants(img: Image.Image):
    base = _scale_cap(img, int(UPSCALE_TARGET))
    for rot in (0, 90, 180, FAST_MAX_ROT):
        g = ImageOps.grayscale(base.rotate(rot, expand=True))
        # very light denoise + unsharp
        yield g.filter(ImageFilter.MedianFilter(3)).filter(
            ImageFilter.UnsharpMask(radius=1, percent=70, threshold=3)
        )

# ----------------- Tier 3 (rescue): Color-agnostic ROI + tiny angle sweep -----------------
def _find_rois_color_agnostic(bgr):
    """Return list of candidate ROIs regardless of label color; uses:
       1) OpenCV BarcodeDetector quads (fast, color-agnostic)
       2) (Optional) yellow hint for speed if present
       3) Stripe/gradient heuristic to catch white/grey stickers
    """
    rois = []

    # 1) Built-in BarcodeDetector (boxes)
    try:
        bd = cv2.barcode_BarcodeDetector()
        ok, vals, pts = bd.detectAndDecode(bgr)
        if pts is not None and len(pts) > 0:
            for quad in pts:
                quad = quad.reshape(-1, 2).astype(np.float32)
                W = int(np.linalg.norm(quad[1]-quad[0]))
                H = int(np.linalg.norm(quad[3]-quad[0]))
                if W*H >= 8000:
                    M = cv2.getPerspectiveTransform(quad, np.float32([[0,0],[W-1,0],[W-1,H-1],[0,H-1]]))
                    roi = cv2.warpPerspective(bgr, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                    rois.append(roi)
    except Exception:
        pass

    # 2) Optional yellow hint (kept as *hint*, not required)
    if ENABLE_YELLOW_ROI:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (15, 50, 70), (45, 255, 255))
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, 2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, 1)
        cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if cv2.contourArea(c) < 3000: 
                continue
            rect = cv2.minAreaRect(c)
            box  = cv2.boxPoints(rect).astype(np.float32)
            W = int(max(rect[1][0], 1)); H = int(max(rect[1][1], 1))
            if W*H < 10000:
                continue
            M = cv2.getPerspectiveTransform(box, np.float32([[0,0],[W-1,0],[W-1,H-1],[0,H-1]]))
            roi = cv2.warpPerspective(bgr, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            rois.append(roi)

    # 3) Stripe/gradient heuristic (works for white/grey)
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (3,3), 0.8)
    grad_x = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.absdiff(grad_x, grad_y)
    grad = cv2.convertScaleAbs(grad)
    grad = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(25,5)), 1)
    _, binm = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    binm = cv2.morphologyEx(binm, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(9,3)), 1)
    cnts,_ = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if w*h < 12000 or w < h:
            continue
        roi = bgr[max(y-6,0):y+h+6, max(x-6,0):x+w+6]
        if roi.size: rois.append(roi)

    if not rois:
        rois = [bgr]  # fallback
    return rois

def tier3_rescue_np(rgb):
    """Returns decoded results from rescue tier (or [])."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    targets = _find_rois_color_agnostic(bgr) if HAS_CV else [bgr]

    results, seen = [], set()
    t0 = time.time()
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(CLAHE_TILE, CLAHE_TILE))

    for tgt in targets:
        gray = cv2.cvtColor(tgt, cv2.COLOR_BGR2GRAY)
        base = clahe.apply(gray)
        sharp = cv2.addWeighted(base, 1.3, cv2.GaussianBlur(base, (0,0), 1.2), -0.3, 0)

        h, w = base.shape
        ctr = (w/2, h/2)
        for ang in range(-ROT_FINE_SWEEP, ROT_FINE_SWEEP+1, 5):
            if (time.time() - t0) * 1000 > MAX_MS_RESUE:
                return results  # safety cutoff
            M = cv2.getRotationMatrix2D(ctr, ang, 1.0)
            rot = cv2.warpAffine(sharp, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

            pil_img = Image.fromarray(rot)
            _dedupe_append(results, seen, decode(pil_img, symbols=SYMBOLS))
            if EARLY_EXIT and results: 
                return results
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
        meta["tiers"].append({"name": "tier3_rescue_cv", "ms": int((time.time()-t0)*1000), "hits": len(results), "yellow_roi_hint": ENABLE_YELLOW_ROI})
    elif not HAS_CV:
        meta["tiers"].append({"name": "tier3_rescue_cv", "skipped": "opencv_not_available"})

    meta["total_ms"] = int((time.time()-t_start)*1000)

    if LOG_METRICS:
        print(f"[decode] total={meta['total_ms']}ms hits={len(results)} tiers={meta['tiers']}")

    payload = {"barcodes": results}
    if debug:
        payload["meta"] = meta
        payload["config"] = {
            "symbols": SYMBOLS_ENV, "early_exit": EARLY_EXIT, "enable_yellow_roi_hint": ENABLE_YELLOW_ROI,
            "rot_fine_sweep": ROT_FINE_SWEEP, "clahe": {"clip": CLAHE_CLIP, "tile": CLAHE_TILE},
            "upscale_target": UPSCALE_TARGET
        }
    return jsonify(payload)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
