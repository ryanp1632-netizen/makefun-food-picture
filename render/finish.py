"""Turn a chosen candidate into entry.png under the gate's limits (whole PNG or JPEG, <= 1 MiB, <= 1024x1024).

    python render/finish.py <candidate.png> entry.png

Keeps a lossless PNG when it fits; otherwise writes the highest-quality baseline JPEG that fits
(the gate tells PNG and JPEG apart by their first bytes, so the name stays entry.png).
"""
import io
import sys

from PIL import Image

CAP = 1048576


def main() -> None:
    src, dst = sys.argv[1], sys.argv[2]
    img = Image.open(src).convert("RGB")
    if max(img.size) > 1024:
        img.thumbnail((1024, 1024), Image.LANCZOS)

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
    with open(dst, "wb") as f:
        f.write(data)
    print(dst, img.size, len(data), "bytes", "PNG" if data[:4] == b"\x89PNG" else "JPEG")


if __name__ == "__main__":
    main()
