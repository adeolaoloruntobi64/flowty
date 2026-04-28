import os, csv, pandas as pd
import sklearn
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

"""
Train on good and bad set, include maybe set in good set if INCLUDE_MAYBE = True

"""

INCLUDE_MAYBE = False
DATA_FILE_MULTI = os.path.expanduser("~/pixel_training_dataset_multi.csv")
GROUPS = [
    ["R255", "G255", "B255"],
    ["H_sl", "L_sl", "S_sl"],
    ["H_sv", "S_sv", "V_sv"],
    ["Lab_L", "Lab_a", "Lab_b"],
    ["LCh_L", "LCh_C", "LCh_h"],
    ["Luv_L", "Luv_u", "Luv_v"],
    ["XYZ_x", "XYZ_y", "XYZ_z"],
    ["CMY_c", "CMY_m", "CMY_y"],
    ["CMYK_c", "CMYK_m", "CMYK_y", "CMYK_k"]
]

print(f"Reading {DATA_FILE_MULTI}")
df = pd.read_csv(DATA_FILE_MULTI)
print(f"Done reading {DATA_FILE_MULTI}")
if INCLUDE_MAYBE:
    df.loc[df["label"] == "maybe", "label"] = "good"
else:
    df = df[df["label"] != "maybe"].copy()
print(f"INCLUDE_MAYBE: {INCLUDE_MAYBE}")

def report_failures(name, X, y_true, y_pred, full_df):
    errors_mask = y_pred != y_true
    print(f"\n--- {name} Failures ({sum(errors_mask)} total) ---")
    
    if errors_mask.any():
        failed_samples = full_df.loc[X[errors_mask].index].copy()
        failed_samples['predicted'] = y_pred[errors_mask]
        print(failed_samples[['x', 'y', 'label', 'predicted', 'H_sv', 'S_sv', 'V_sv'] + group])
    else:
        print("None! Perfect score.")

def handle_group(group):
    X = df[group]
    y = df["label"]
    
    print(f"Fitting for: {group}")
    state = 3873465
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.20, random_state=state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=state
    )
    model = DecisionTreeClassifier(max_depth=3)
    scores = cross_val_score(model, X, y, cv=100)
    model.fit(X_train, y_train)
    val_preds = model.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)
    test_preds = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_preds)

    print(f"==========================================")
    print(f"--- Group: {group} ---")
    print(f"Validation Accuracy: {val_acc:.2%}")
    print(f"Final Test Accuracy:  {test_acc:.2%}")
    print(f"True Mean Accuracy: {scores.mean():.2%}")
    print(f"Stability (Std Dev): {scores.std():.4f}")
    report_failures("VALIDATION", X_val, y_val, val_preds, df)
    report_failures("FINAL TEST", X_test, y_test, test_preds, df)
    print(f"------------------------------------------")
    print(export_text(model, feature_names=group))
    print(f"==========================================\n")
    return (val_acc, test_acc, float(scores.mean()), float(scores.std()))

def handle_group_plane(group):
    X = df[group]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    scores = cross_val_score(model, X, y, cv=100)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    errors_mask = y_pred != y_test

    print(f"\n{'='*20} GROUP: {group} {'='*20}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")

    weights = model.coef_[0]
    intercept = model.intercept_[0]
    
    formula_parts = [f"({w:.4f} * {name})" for w, name in zip(weights, group)]
    formula = " + ".join(formula_parts) + f" + ({intercept:.4f})"
    
    print("\n--- GENERAL FORMULA (Decision Plane) ---")
    print(f"Prediction Value = {formula}")
    print("If Value > 0: 'good' | If Value < 0: 'bad'")
    print(f"==========================================")
    print(f"True Mean Accuracy: {scores.mean():.2%}")
    print(f"Stability (Std Dev): {scores.std():.4f}")
    print(f"==========================================")
    if errors_mask.any():
        print("\n--- Failed Samples (Misclassified) ---")
        failed_samples = df.loc[y_test[errors_mask].index].copy()
        failed_samples['predicted'] = y_pred[errors_mask]
        print(failed_samples[['x', 'y', 'label', 'predicted'] + group])
    else:
        print("\nPerfect! No failures in the test set.")

def handle_group_svm(group):
    X = df[group]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = SVC(kernel='linear')
    scores = cross_val_score(model, X, y, cv=10)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    errors_mask = y_pred != y_test

    print(f"\n{'='*20} GROUP: {group} {'='*20}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")

    weights = model.coef_[0]
    intercept = model.intercept_[0]
    
    formula_parts = [f"({w:.4f} * {name})" for w, name in zip(weights, group)]
    formula = " + ".join(formula_parts) + f" + ({intercept:.4f})"
    
    print("\n--- GENERAL FORMULA (Decision Plane) ---")
    print(f"Prediction Value = {formula}")
    print("If Value > 0: 'good' | If Value < 0: 'bad'")
    print(f"==========================================")
    print(f"True Mean Accuracy: {scores.mean():.2%}")
    print(f"Stability (Std Dev): {scores.std():.4f}")
    print(f"==========================================")
    if errors_mask.any():
        print("\n--- Failed Samples (Misclassified) ---")
        failed_samples = df.loc[y_test[errors_mask].index].copy()
        failed_samples['predicted'] = y_pred[errors_mask]
        print(failed_samples[['x', 'y', 'label', 'predicted'] + group])
    else:
        print("\nPerfect! No failures in the test set.")

#for g in GROUPS:
#    handle_group_svm(g)
highest = (-99999, -9999)
g = []
for group in GROUPS:
    acc = handle_group(group)
    g.append((acc, group))
    if acc > highest:
        print(f"New highest accuracy: {group} : {acc}")
        highest = acc
g = sorted(g, key=lambda x: x[0], reverse=True)
for i in g:
    print(f"{i[0]} : {i[1]}")

"""
|--- G255 <= 51.50
|   |--- R255 <= 83.50
|   |   |--- B255 <= 162.00
|   |   |   |--- class: bad
|   |   |--- B255 >  162.00
|   |   |   |--- class: good
|   |--- R255 >  83.50
|   |   |--- R255 <= 109.00
|   |   |   |--- class: good
|   |   |--- R255 >  109.00
|   |   |   |--- class: good
|--- G255 >  51.50
|   |--- R255 <= 42.00
|   |   |--- R255 <= 11.00
|   |   |   |--- class: good
|   |   |--- R255 >  11.00
|   |   |   |--- class: bad
|   |--- R255 >  42.00
|   |   |--- B255 <= 40.50
|   |   |   |--- class: good
|   |   |--- B255 >  40.50
|   |   |   |--- class: good

GOOD:

G255 <= 51.50 && R255 <= 83.50 && B255 > 162.00
G255 <= 51.50 && R255 >  83.50
G255 >  51.50 && R255 <= 42.00 && R255 <= 11.00
G255 >  51.50 && R255 >  42.00


OR
|--- G255 <= 51.50
|   |--- R255 <= 83.50
|   |   |--- B255 <= 162.00
|   |   |   |--- class: bad
|   |   |--- B255 >  162.00
|   |   |   |--- class: good
|   |--- R255 >  83.50
|   |   |--- class: good
|--- G255 >  51.50
|   |--- R255 <= 42.00
|   |   |--- R255 <= 11.00
|   |   |   |--- class: good
|   |   |--- R255 >  11.00
|   |   |   |--- class: bad
|   |--- R255 >  42.00
|   |   |--- class: good
"""