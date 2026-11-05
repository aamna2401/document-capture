"""Render the pipeline flowchart and the system architecture diagram with Graphviz (docs/img/)."""
import subprocess
from pathlib import Path

IMG = Path(__file__).resolve().parents[1] / "docs" / "img"
IMG.mkdir(parents=True, exist_ok=True)

STYLE = 'graph [fontname="Helvetica", rankdir=TB, nodesep=0.35, ranksep=0.45, pad=0.3]; node [fontname="Helvetica", fontsize=11, shape=box, style="rounded,filled", fillcolor="#eef3fb", color="#2f5d9e"]; edge [fontname="Helvetica", fontsize=9, color="#555555"];'

FLOW = f"""digraph flow {{
{STYLE}
photo [label="Phone photo\\n(any angle, any background)", shape=ellipse, fillcolor="#fff6dd", color="#b58105"];
subgraph cluster_detect {{ label="1. Detect the document"; style="rounded,dashed"; color="#999999"; fontname="Helvetica";
  c1 [label="Otsu bright / dark\\nthresholds"]; c2 [label="Adaptive\\nthreshold"]; c3 [label="Canny edges\\n+ closing"]; c4 [label="Background model\\n(Lab distance from border)"]; c5 [label="GrabCut"];
  quads [label="Quadrilateral candidates\\n(approxPolyDP, min-area rect)", fillcolor="#ffffff"];
  score [label="Score each candidate\\nedge support, area, contrast,\\ncorner angles, mask fit,\\nstrategy prior (tuned on validation)", fillcolor="#ffffff"];
  c1 -> quads; c2 -> quads; c3 -> quads; c4 -> quads; c5 -> quads; quads -> score;
}}
ok [label="Best score >= threshold?", shape=diamond, fillcolor="#fff6dd", color="#b58105"];
full [label="Fallback: full frame\\n(pipeline degrades to no change)", fillcolor="#fdeaea", color="#b04040"];
rect [label="2. Rectify\\nhomography to upright rectangle\\nwidth/height ratio from camera model\\n(Zhang-He, guarded by edge lengths)"];
ori [label="3. Orientation\\nOCR-confidence over 0/90/180/270 turns"];
skew [label="4. Residual deskew\\nprojection-profile search, +/-4 deg"];
ocr [label="OCR (Tesseract / RapidOCR)", fillcolor="#e9f6ee", color="#2e8b57"];
out [label="Flat, upright, OCR-ready scan\\n+ corners, score, method (JSON)", shape=ellipse, fillcolor="#e9f6ee", color="#2e8b57"];
photo -> c1; photo -> c2; photo -> c3; photo -> c4; photo -> c5;
score -> ok; ok -> rect [label="yes"]; ok -> full [label="no"]; full -> rect;
rect -> ori -> skew -> out; out -> ocr [style=dashed];
}}"""

ARCH = f"""digraph arch {{
{STYLE} rankdir=LR;
client [label="Client\\n(mobile app / script)", shape=ellipse, fillcolor="#fff6dd", color="#b58105"];
subgraph cluster_svc {{ label="docscan service (FastAPI)"; style="rounded,dashed"; color="#999999"; fontname="Helvetica";
  api [label="POST /scan\\nPOST /detect\\nGET /health"];
  pipe [label="pipeline.scan()"];
  det [label="detect.py\\nproposals + scoring"]; rec [label="rectify.py\\nhomography + aspect"]; ori [label="orient.py\\nrotation + deskew"];
  api -> pipe; pipe -> det; pipe -> rec; pipe -> ori;
}}
ocr [label="OCR engine\\n(Tesseract / RapidOCR)", fillcolor="#e9f6ee", color="#2e8b57"];
subgraph cluster_eval {{ label="Offline evaluation"; style="rounded,dashed"; color="#999999"; fontname="Helvetica";
  sim [label="simulate.py\\npinhole-camera capture\\n+ exact ground truth"]; data [label="data.py\\nCORD-v2 receipts (CC BY 4.0)"]; met [label="metrics.py\\nIoU, corner error, word F1,\\nnumeric recall"]; run [label="scripts/run_eval.py"]; tune [label="scripts/tune_detector.py\\n(validation split only)"];
  data -> sim -> run; run -> met; tune -> det [style=dashed, label="weights"];
}}
client -> api [label="image"]; api -> client [label="scan.png + JSON"]; pipe -> ocr [style=dashed, label="orientation score"]; run -> pipe;
}}"""

for name, src in (("pipeline_flow", FLOW), ("architecture", ARCH)):
    dot = IMG / f"{name}.dot"
    dot.write_text(src)
    for fmt in ("png", "svg"):
        subprocess.run(["dot", f"-T{fmt}", "-Gdpi=130" if fmt == "png" else "-Gdpi=96", str(dot), "-o", str(IMG / f"{name}.{fmt}")], check=True)
    print("wrote", name)
