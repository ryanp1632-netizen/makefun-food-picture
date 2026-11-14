"""Turn a chosen candidate into entry.png under the gate's limits (whole PNG or JPEG, <= 1 MiB, <= 1024x1024).

    python render/finish.py <candidate.png> entry.png [--crop left,top,right,bottom]

--crop takes a square box in source pixels first (used to trim a stray fingertip at the frame edge, which the
judge counts as a broken hand). The result is resized to 1024x1024 with Lanczos. Keeps a lossless PNG when it
fits; otherwise writes the highest-quality baseline JPEG that fits (the gate tells PNG and JPEG apart by their
first bytes, so the name stays entry.png).
"""
import argparse
import io

from PIL import Image

CAP = 1048576


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--crop", help="left,top,right,bottom in source pixels (a square box)")
    args = ap.parse_args()

    img = Image.open(args.src).convert("RGB")
    if args.crop:
        box = tuple(int(v) for v in args.crop.split(","))
        assert box[2] - box[0] == box[3] - box[1], "crop box must be square"
        img = img.crop(box)
    if max(img.size) > 1024:
        img = img.resize((1024, 1024), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    if buf.tell() > CAP:
        for q in range(97, 80, -1):
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=q, subsampling=0, optimize=True)
            if buf.tell() <= CAP:
                break
    data = buf.getvalue()
    assert len(data) <= CAP, len(data)
    with open(args.dst, "wb") as f:
        f.write(data)
    print(args.dst, img.size, len(data), "bytes", "PNG" if data[:4] == b"\x89PNG" else "JPEG")


if __name__ == "__main__":
    main()
