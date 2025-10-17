from flask import Flask, request, jsonify
from pyzbar.pyzbar import decode, ZBarSymbol
from PIL import Image, ImageOps, ImageFilter

app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    return "ok", 200

def gen_variants(img: Image.Image):
    w, h = img.size
    max_side = max(w, h)
    if max_side < 1600:
        scale = 1600 / max_side
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    for rot in (0, 90, 180, 270):
        base = img.rotate(rot, expand=True)
        g = ImageOps.grayscale(base)
        yield g
        yield ImageOps.autocontrast(g, cutoff=2)
        yield g.point(lambda x: 255 if x > 140 else 0)
        yield g.point(lambda x: 255 if x > 110 else 0)
        yield ImageOps.autocontrast(g.filter(ImageFilter.SHARPEN), cutoff=2)

@app.route("/decode", methods=["POST"])
def decode_barcode():
    if "file" not in request.files:
        return jsonify({"error": "no file"}), 400
    f = request.files["file"]
    try:
        img = Image.open(f.stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"cannot open image: {e}"}), 400
    results = []
    for variant in gen_variants(img):
        found = decode(variant, symbols=[ZBarSymbol.CODE128])
        if found:
            for b in found:
                results.append({"type": b.type, "data": b.data.decode("utf-8", "ignore")})
            break
    return jsonify({"barcodes": results})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
