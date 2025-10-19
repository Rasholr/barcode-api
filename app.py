from flask import Flask, request, jsonify
from pyzbar.pyzbar import decode, ZBarSymbol
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import io

# try to use OpenCV if present (better for uneven lighting)
try:
    import cv2
    import numpy as np
    HAS_CV = True
except Exception:
    HAS_CV = False

app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    return "ok", 200

def pil_to_np(img: Image.Image):
    return np.array(img)[:, :, ::-1]  # RGB->BGR

def np_to_pil(arr):
    return Image.fromarray(arr[:, :, ::-1])  # BGR->RGB

def scale_up(img: Image.Image, target=1600):
    w, h = img.size
    m = max(w, h)
    if m < target:
        s = target / m
        img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
    return img

def gen_variants_pil(img: Image.Image):
    # base rotations and mild preprocessing without hard thresholding
    for rot in (0, 90, 180, 270):
        base = img.rotate(rot, expand=True)
        base = scale_up(base, 1400)

        # also try a bit larger
        for scale in (1.0, 1.4, 1.8):
            w, h = base.size
            b = base if scale == 1.0 else base.resize((int(w*scale), int(h*scale)), Image.LANCZOS)

            g = ImageOps.grayscale(b)

            # plain gray
            yield g

            # equalized histogram (gentle)
            yield ImageOps.equalize(g)

            # autocontrast with tiny cutoff (avoid overexposure)
            yield ImageOps.autocontrast(g, cutoff=0.5)

            # sharpen a little (preserve edges)
            yield ImageOps.equalize(g.filter(ImageFilter.UnsharpMask(radius=1, percent=80, threshold=3)))

            # slight denoise then sharpen
            yield g.filter(ImageFilter.MedianFilter(size=3)).filter(ImageFilter.UnsharpMask(radius=1, percent=60, threshold=2))

            # very light gaussian blur (helps broken bars)
            yield g.filter(ImageFilter.GaussianBlur(0.7))

            # try inverted (some labels are printed light-on-dark)
            yield ImageOps.invert(g)

def gen_variants_cv(img: Image.Image):
    """OpenCV branch for uneven lighting (optional)."""
    if not HAS_CV:
        return
    rgb = np.array(img)
    for rot in (0, 90, 180, 270):
        # rotate
        M = cv2.getRotationMatrix2D((rgb.shape[1]/2, rgb.shape[0]/2), rot, 1.0)
        r = cv2.warpAffine(rgb, M, (rgb.shape[1], rgb.shape[0]), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        # upscale if needed
        h, w = r.shape[:2]
        m = max(h, w)
        if m < 1600:
            s = 1600 / m
            r = cv2.resize(r, (int(w*s), int(h*s)), interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(r, cv2.COLOR_RGB2GRAY)

        # CLAHE (local contrast, не «пересвечивает»)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        g1 = clahe.apply(gray)
        yield Image.fromarray(g1)

        # adaptive threshold (мягкая бинаризация, устойчиво к неравномерному свету)
        for bs in (21, 31, 41):
            at = cv2.adaptiveThreshold(g1, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY, blockSize=bs, C=5)
            # немного «соединить» полосы вертикальным морф-ядром
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
            opened = cv2.morphologyEx(at, cv2.MORPH_OPEN, k, iterations=1)
            yield Image.fromarray(opened)

        # лёгкий bilateral (сохраняет края), затем Unsharp
        b = cv2.bilateralFilter(g1, 7, 40, 40)
        us = cv2.addWeighted(b, 1.5, cv2.GaussianBlur(b, (0, 0), 1.0), -0.5, 0)
        yield Image.fromarray(us)

@app.route("/decode", methods=["POST"])
def decode_barcode():
    if "file" not in request.files:
        return jsonify({"error": "no file"}), 400

    f = request.files["file"]
    try:
        img = Image.open(f.stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"cannot open image: {e}"}), 400

    # collect unique results across many variants
    seen = set()
    results = []

    # Order: quick PIL variants first, then heavier OpenCV ones
    for variant in gen_variants_pil(img):
        found = decode(variant, symbols=[ZBarSymbol.CODE128])
        for b in found or []:
            key = (b.type, b.data)
            if key not in seen:
                seen.add(key)
                results.append({"type": b.type, "data": b.data.decode("utf-8", "ignore")})
        if results:
            # if we already have a confident read, no need to try endless variants
            break

    if not results and HAS_CV:
        for variant in gen_variants_cv(img):
            found = decode(variant, symbols=[ZBarSymbol.CODE128])
            for b in found or []:
                key = (b.type, b.data)
                if key not in seen:
                    seen.add(key)
                    results.append({"type": b.type, "data": b.data.decode("utf-8", "ignore")})
            if results:
                break

    return jsonify({"barcodes": results})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
