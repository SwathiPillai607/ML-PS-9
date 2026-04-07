import argparse
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True)
parser.add_argument("--target", required=True)
parser.add_argument("--components", nargs='+', type=int, required=True)

args = parser.parse_args()

# Load dataset
df = pd.read_csv(args.data)

# Separate features and target
X = df.drop(columns=[args.target])
y = df[args.target]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# Classification on Original Data
clf = LogisticRegression(max_iter=500)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

orig_acc = accuracy_score(y_test, y_pred)
orig_f1 = f1_score(y_test, y_pred, average="macro")

print("\nOriginal Data Performance")
print("Accuracy:", orig_acc)
print("F1 Score:", orig_f1)

# LDA Transformations
for comp in args.components:

    lda = LinearDiscriminantAnalysis(n_components=comp)

    X_train_lda = lda.fit_transform(X_train, y_train)
    X_test_lda = lda.transform(X_test)

    clf = LogisticRegression(max_iter=500)
    clf.fit(X_train_lda, y_train)

    y_pred = clf.predict(X_test_lda)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    print(f"\nLDA with {comp} component(s)")
    print("Accuracy:", acc)
    print("F1 Score:", f1)

    # Save projection if 2D
    if comp == 2:
        lda_full = LinearDiscriminantAnalysis(n_components=2)
        X_lda_full = lda_full.fit_transform(X_scaled, y)

        projection = pd.DataFrame({
            "SampleId": range(len(X_lda_full)),
            "LD1": X_lda_full[:,0],
            "LD2": X_lda_full[:,1],
            "Class": y
        })

        projection.to_csv("lda_projection.csv", index=False)
        print("\nSaved lda_projection.csv")