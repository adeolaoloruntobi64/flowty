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

data = [

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

df = pd.DataFrame(data, columns=["H","S","V","label","weight","strength"])

# Train using S and V only
X = df[["H","S","V"]]
y = df["label"]
weights = df["weight"]

model = DecisionTreeClassifier(max_depth=4)
model.fit(X, y, sample_weight=weights)

print("Decision Rules:\n")
print(export_text(model, feature_names=["H","S","V"]))

H_vals = np.linspace(0, 360, 50)
S_vals = np.linspace(0, 100, 50)
V_vals = np.linspace(0, 100, 50)

Hg, Sg, Vg = np.meshgrid(H_vals, S_vals, V_vals)
grid = np.c_[Hg.ravel(), Sg.ravel(), Vg.ravel()]

pred = model.predict(grid)

keep_points = grid[pred == 1]

strong_keep = df[(df.label==1) & (df.strength=="strong")]
weak_keep = df[(df.label==1) & (df.strength=="weak")]
strong_discard = df[(df.label==0) & (df.strength=="strong")]
weak_discard = df[(df.label==0) & (df.strength=="weak")]

# I asked AI to make this because I was interested in viewing the data another way
# It was cool ig, but didn't really reveal anything to me
def plot_tree_cuboids(tree, feature_names, ax):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from sklearn.tree import _tree
    import numpy as np
    """
    Draws axis-aligned cuboids for tree leaves predicting 'keep' in 3D.
    """
    tree_ = tree.tree_
    
    def recurse(node, mins, maxs):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            # internal node
            feat = tree_.feature[node]
            thresh = tree_.threshold[node]

            # left child has feature <= threshold
            left_maxs = maxs.copy()
            left_maxs[feat] = thresh
            recurse(tree_.children_left[node], mins, left_maxs)

            # right child has feature > threshold
            right_mins = mins.copy()
            right_mins[feat] = thresh
            recurse(tree_.children_right[node], right_mins, maxs)
        else:
            # leaf node
            value = tree_.value[node][0]
            if np.argmax(value) == 1:  # keep
                # draw cuboid
                Hmin, Smin, Vmin = mins
                Hmax, Smax, Vmax = maxs
                # define cuboid vertices
                r = np.array([[Hmin,Smin,Vmin],
                              [Hmax,Smin,Vmin],
                              [Hmax,Smax,Vmin],
                              [Hmin,Smax,Vmin],
                              [Hmin,Smin,Vmax],
                              [Hmax,Smin,Vmax],
                              [Hmax,Smax,Vmax],
                              [Hmin,Smax,Vmax]])
                # define faces
                faces = [[r[0],r[1],r[2],r[3]],
                         [r[4],r[5],r[6],r[7]],
                         [r[0],r[1],r[5],r[4]],
                         [r[2],r[3],r[7],r[6]],
                         [r[1],r[2],r[6],r[5]],
                         [r[4],r[7],r[3],r[0]]]
                poly3d = Poly3DCollection(faces, alpha=0.1, facecolor='blue')
                ax.add_collection3d(poly3d)

    mins = np.array([0.0, 0.0, 0.0])
    maxs = np.array([360.0, 100.0, 100.0])
    recurse(0, mins, maxs)


fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(df[df.label==1].H, df[df.label==1].S, df[df.label==1].V,
           color='green', marker='o', s=80, label='Keep points')
ax.scatter(df[df.label==0].H, df[df.label==0].S, df[df.label==0].V,
           color='red', marker='x', s=80, label='Discard points')

plot_tree_cuboids(model, ["H","S","V"], ax)

ax.set_xlabel('Hue (H)')
ax.set_ylabel('Saturation (S)')
ax.set_zlabel('Value (V)')
ax.set_title('3D HSV Decision Tree Planes')
ax.legend()
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
"""