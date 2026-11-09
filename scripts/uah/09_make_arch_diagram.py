import os
import matplotlib
matplotlib.use("Agg")
# No hatching: the two fills differ strongly in lightness, so the distinction
# survives greyscale printing on tone alone, and a pattern behind the labels
# made them hard to read.
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(6.8, 2.95))
ax.set_xlim(0, 11.8)
ax.set_ylim(0.85, 5.35)
ax.axis("off")

# Two fills carry the evaluated/proposed distinction the caption claims, matching
# the split in Table 3. Dark fill: measured in Section 6. Light fill: specified
# in Section 4 but not deployed or benchmarked.
EVALUATED = "#2E5A88"
PROPOSED = "#B8C9DC"


def box(x, y, w, h, text, color=EVALUATED, fontsize=8, textcolor="white",
        proposed=False):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.06",
                        linewidth=1.2, edgecolor="#1f3a5f",
                        facecolor=PROPOSED if proposed else color, zorder=2)
    ax.add_patch(b)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fontsize,
            color=textcolor, weight="bold", wrap=True, zorder=4)
    return (x, y, w, h)

def arrow(b1, b2, label=None, dy=0):
    x1 = b1[0] + b1[2]
    y1 = b1[1] + b1[3]/2 + dy
    x2 = b2[0]
    y2 = b2[1] + b2[3]/2 + dy
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                         linewidth=1.4, color="#333333")
    ax.add_patch(a)
    if label:
        ax.text((x1+x2)/2, y1+0.18, label, ha="center", fontsize=8, color="#333333")

# Row 1: Ingestion
edge = box(0.2, 4.30, 1.8, 0.78, "Vehicle Sensors\n(OBD-II, GPS,\naccelerometer)")
gw = box(2.5, 4.30, 1.8, 0.78, "Edge Buffer\n(multi-rate sync)")
broker = box(4.8, 4.30, 1.8, 0.78, "Ingestion Broker\n(e.g. Kafka)", textcolor="#1a1a1a", proposed=True)

arrow(edge, gw)
arrow(gw, broker)

# Row 2: Stream processing / feature layer
stream = box(4.8, 3.00, 1.8, 0.78, "Stream Processor\n(windowed feature\ncomputation)", fontsize=7.2)
feat = box(7.1, 3.00, 1.8, 0.78, "Feature Store\n(hot path)", textcolor="#1a1a1a", proposed=True)

a = FancyArrowPatch((broker[0]+broker[2]/2, broker[1]), (stream[0]+stream[2]/2, stream[1]+stream[3]),
                     arrowstyle="-|>", mutation_scale=14, linewidth=1.4, color="#333333")
ax.add_patch(a)
arrow(stream, feat)

# Storage cold path
cold = box(7.1, 1.75, 1.8, 0.78, "Cold Storage /\nData Lake", textcolor="#1a1a1a", proposed=True)
a = FancyArrowPatch((feat[0]+feat[2]/2, feat[1]), (cold[0]+cold[2]/2, cold[1]+cold[3]),
                     arrowstyle="-|>", mutation_scale=14, linewidth=1.2, color="#666666", linestyle="--")
ax.add_patch(a)
ax.text(cold[0]+cold[2]/2+0.12, (feat[1]+cold[1]+cold[3])/2, "async archive",
        fontsize=8, ha="left", va="center", color="#444444")

# Row 1 continued: serving / decision
model = box(9.4, 3.00, 1.8, 0.78, "Behavior\nClassifier")
arrow(feat, model)

decision = box(9.4, 4.30, 1.8, 0.78, "Decision API\n(scoring, alerts)", textcolor="#1a1a1a", proposed=True)
a = FancyArrowPatch((model[0]+model[2]/2, model[1]+model[3]), (decision[0]+decision[2]/2, decision[1]),
                     arrowstyle="-|>", mutation_scale=14, linewidth=1.4, color="#333333")
ax.add_patch(a)

# Feedback loop. Section 4.6 writes decision outcomes back to the cold-path
# store as labeled training data, so the arrow ends at cold storage, not at the
# sensors. Routed down the right-hand side and back under the serving row.
FB_Y = cold[1] - 0.45
fb_x = decision[0] + decision[2] + 0.28
cold_x = cold[0] + cold[2]/2
ax.plot([decision[0] + decision[2], fb_x], [decision[1] + decision[3]/2, decision[1] + decision[3]/2],
        color="#7a3f08", linewidth=1.1, linestyle=":", zorder=1)
ax.plot([fb_x, fb_x], [decision[1] + decision[3]/2, FB_Y],
        color="#7a3f08", linewidth=1.1, linestyle=":", zorder=1)
ax.plot([fb_x, cold_x], [FB_Y, FB_Y],
        color="#7a3f08", linewidth=1.1, linestyle=":", zorder=1)
a = FancyArrowPatch((cold_x, FB_Y), (cold_x, cold[1]),
                     arrowstyle="-|>", mutation_scale=13,
                     linewidth=1.1, color="#7a3f08", linestyle=":")
ax.add_patch(a)
ax.text((fb_x + cold_x)/2, FB_Y - 0.14,
        "labeled outcomes (4.6)",
        fontsize=8, ha="center", va="top", color="#7a3f08")

# Scope annotation. Section 6 measures the inference stage only, excluding
# broker transport, queueing, feature-store lookup and response delivery, so the
# figure must not promise a per-stage breakdown the paper does not report.
# Section numbers map each component to the subsection that specifies it.
for _b, _sec in ((gw, "4.1"), (broker, "4.1"), (stream, "4.2"), (feat, "4.3"),
                 (cold, "4.4"), (model, "4.5"), (decision, "4.5")):
    ax.text(_b[0] + 0.04, _b[1] + _b[3] + 0.07, _sec,
            fontsize=7, color="#5a5a5a", ha="left", va="bottom")

from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor=EVALUATED, edgecolor="#1f3a5f",
                         label="function evaluated in this paper (Section 6)"),
                   Patch(facecolor=PROPOSED, edgecolor="#1f3a5f",
                         label="specified, not evaluated (Section 4)")],
          loc="lower left", bbox_to_anchor=(0.0, 0.0), fontsize=8.5,
          frameon=False, handlelength=1.8)

ax.text(0.2, 2.45, "Section 6 evaluates classification, inference, and ingestion.",
        fontsize=8.5, color="#333333", va="top")

# No title: the caption belongs in the article text, not in the graphics file
# (IOP author guidelines), and the paper already captions this as Figure 1.

os.makedirs("figures", exist_ok=True)
plt.tight_layout()
plt.savefig("figures/architecture_diagram.png", dpi=220)
print("saved architecture diagram")
