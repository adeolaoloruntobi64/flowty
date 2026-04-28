import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_text

# --------------------------------------------------
# Dataset
# (H, S, V, label, weight, strength)
#
# label:
#   1 = keep
#   0 = discard
#
# weight:
#   5 = strong requirement
#   1 = optional
# --------------------------------------------------

data_hsv = [

# ---------------- STRONG KEEP ----------------
(120,100,55.3,1,5,"strong"),
(120,100,38.4,1,5,"strong"),
(0,100,100,1,5,"strong"),
(60,44.9,88.2,1,5,"strong"),
(56.6,49.1,42.4,1,5,"strong"),
(56.3,49.0,39.2,1,5,"strong"),
(232.8,95.3,99.6,1,5,"strong"),
(232.9,95.5,69.4,1,5,"strong"),
(57.4,100,91.8,1,5,"strong"),
(128,43,39.2,1,5,"strong"),
#(120,41.5,16.1,1,5,"strong"),
(120,42.2,42.7,1,5,"strong"),
#(120,43.5,18.0,1,5,"strong"),
(120,42.2,35.3,1,5,"strong"),
(313.2,96.1,100,1,5,"strong"),
(32.7,100,98.4,1,5,"strong"),
(120,41.4,93.7,1,5,"strong"),
(180,100,100,1,5,"strong"),
(300,100,50.2,1,5,"strong"),
(0,0,100,1,5,"strong"),
(240,15.9,74.1,1,5,"strong"),
(0,45.1,100,1,5,"strong"),
(0,50.4,49,1,5,"strong"),
(357.6,50.5,38.0,1,5,"strong"),
(120,100,100,1,5,"strong"),
(60,45.3,71,1,5,"strong"),
(60,45,54.5,1,5,"strong"),
(60,46.3,47.5,1,5,"strong"),
(58.7,50.5,37.3,1,5,"strong"),

(40,75,20.4,1,5,"strong"),
(64.6,23.5,21.6,1,5,"strong"),
(72,19.2,20.4,1,5,"strong"),
(39.5,75.9,21.2,1,5,"strong"),

# ---------------- WEAK KEEP ----------------
(54.8,50,18,1,1,"weak"),
(56.5,50,13.3,1,1,"weak"),
(120,41.9,12.2,1,1,"weak"),
(343.3,48.6,14.5,1,1,"weak"),
(54.4,60.6,27.8,1,1,"weak"),
(120,41.5,16.1,1,1,"weak"),
(120,43.5,18.0,1,1,"weak"),

# ---------------- STRONG DISCARD ----------------
(57.1,100,8.2,0,5,"strong"),
(57.3,100,8.6,0,5,"strong"),
(0,50,13.3,0,5,"strong"),
(21.4,62.2,17.6,0,5,"strong"),
(27.4,66.6,20.8,0,5,"strong"),
(22.8,61.7,18.4,0,5,"strong"),
(356.8,52.8,14.1,0,5,"strong"),
(336.7,62.1,11.4,0,5,"strong"),
(42.6,79.2,18.8,0,5,"strong"),
(55.3,95,15.7,0,5,"strong"),
(0,73.5,13.3,0,5,"strong"),
(232.5,96.0,9.8,0,5,"strong"),
(0,74.2,12.2,0,5,"strong"),
(326.4,86.2,11.4,0,5,"strong"),
(300,100,5.9,0,5,"strong"),
(184,91.8,19.2,0,5,"strong"),
(340.0,82.5,15.7,0,5,"strong"),
(313.9,96.6,22.7,0,5,"strong"),
(120,100,11.8,0,5,"strong"),
(207.7,68.4,22.4,0,5,"strong"),
(313.6,97.8,17.6,0,5,"strong"),
(180,100,18.4,0,5,"strong"),
(286.5,64.6,18.8,0,5,"strong"),
(279.5,59.4,25.1,0,5,"strong"),
(0,74.4,15.3,0,5,"strong"),
(300,9.3,16.9,0,5,"strong"),
(32.2,87.1,24.3,0,5,"strong"),
(32.9,44.3,27.5,0,5,"strong"),
(31.0,52.7,21.6,0,5,"strong"),
(233.9,92.5,20.8,0,5,"strong"),
(58,69.8,16.9,0,5,"strong"),

(242,42.6,26.7,0,5,"strong"),
(242.1,42.4,25.9,0,5,"strong"),
(246.7,40.9,8.6,0,5,"strong"),
(248.6,43.8,12.5,0,5,"strong"),

# ---------------- WEAK DISCARD ----------------
(96.8,96.9,12.5,0,1,"weak"),
(6.2,98.3,23.1,0,1,"weak"),
]

def hsv_to_hsl(h, s_v, v):
    l = v * (1 - s_v / 2)

    if l == 0 or l == 1:
        s_l = 0
    else:
        s_l = (v - l) / min(l, 1 - l)

    h_l = h 
    return h_l, s_l, l

data = []

for (h, s, v, n1, n2, s1) in data_hsv:
    nh, ns, nl = hsv_to_hsl(h, s/100, v/100)
    data.append((nh, ns * 100, nl * 100, n1, n2, s1))

df = pd.DataFrame(data, columns=["H","S","L","label","weight","strength"])

X = df[["S","L"]]
y = df["label"]
weights = df["weight"]

model = DecisionTreeClassifier(max_depth=6)
model.fit(X, y, sample_weight=weights)

print("Decision Rules:\n")
print(export_text(model, feature_names=["S","L"]))

plt.figure(figsize=(8,8))

strong_keep = df[(df.label==1) & (df.strength=="strong")]
weak_keep = df[(df.label==1) & (df.strength=="weak")]
strong_discard = df[(df.label==0) & (df.strength=="strong")]
weak_discard = df[(df.label==0) & (df.strength=="weak")]

plt.scatter(strong_keep.S, strong_keep.L,
            color="green", marker="o", s=80, label="Strong Keep")

plt.scatter(weak_keep.S, weak_keep.L,
            color="lightgreen", marker="o", s=80, label="Weak Keep")

plt.scatter(strong_discard.S, strong_discard.L,
            color="red", marker="x", s=80, label="Strong Discard")

plt.scatter(weak_discard.S, weak_discard.L,
            color="orange", marker="x", s=80, label="Weak Discard")

s_vals = np.linspace(0,100,300)
v_vals = np.linspace(0,100,300)

Sg, Vg = np.meshgrid(s_vals, v_vals)
grid = pd.DataFrame(
    np.c_[Sg.ravel(), Vg.ravel()],
    columns=["S","L"]
)
pred = model.predict(grid).reshape(Sg.shape)

plt.contourf(Sg, Vg, pred, alpha=0.2)

plt.xlabel("Saturation (S)")
plt.ylabel("Lightness (L)")
plt.title("HSL Keep / Discard Classifier")
plt.legend()

plt.show()

"""
|--- V <= 27.65
|   |--- S <= 43.90
|   |   |--- S <= 25.40
|   |   |   |--- class: 0
|   |   |--- S >  25.40
|   |   |   |--- class: 1
|   |--- S >  43.90
|   |   |--- S <= 51.35
|   |   |   |--- class: 0
|   |   |--- S >  51.35
|   |   |   |--- class: 0
|--- V >  27.65
|   |--- class: 1

So
if V > 27.65 or 25.40 < S <= 43.90:
    keep
else:
    discard

|--- V <= 27.65
|   |--- S <= 51.35
|   |   |--- S <= 46.45
|   |   |   |--- class: 0
|   |   |--- S >  46.45
|   |   |   |--- class: 0
|   |--- S >  51.35
|   |   |--- class: 0
|--- V >  27.65
|   |--- class: 1


So

if V > 27.65:
    keep
else:
    discard
"""