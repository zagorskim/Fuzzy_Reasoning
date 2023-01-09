import sklearn.tree
from sklearn.datasets import make_moons as dataset
from sklearn.model_selection import train_test_split
from fuzzytree import FuzzyDecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from mlxtend.plotting import plot_decision_regions
import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt

# Data prep
X, y = dataset(n_samples=300, noise=0.5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Classifiers
for i in range(2, 5):
    print(f"Tree depth: {i}")
    maxdepth = i
    clf_fuzz = FuzzyDecisionTreeClassifier()
    clf_fuzz.max_depth = maxdepth
    clf_fuzz.fit(X_train, y_train)
    clf_sk = DecisionTreeClassifier()
    clf_sk.max_depth = maxdepth
    clf_sk.fit(X_train, y_train)

    # Prediction scores printing
    print(f"fuzzytree: {clf_fuzz.score(X_test, y_test)}")
    print(f"  sklearn: {clf_sk.score(X_test, y_test)}")

# Plots
clf_sk = DecisionTreeClassifier()
clf_sk.max_depth = 3
sklearn.tree.plot_tree(clf_sk.fit(X_train, y_train))

gs = gridspec.GridSpec(2, 2)
fig = plt.figure(figsize=(10,8))
labels = ['Fuzzy Decision Tree', 'Classic Decision Tree']
for clf, lab, grd in zip([clf_fuzz, clf_sk],
                         labels, [[0, 0], [0, 1]]):
    ax = plt.subplot(gs[grd[0], grd[1]])
    fig = plot_decision_regions(X=X_train, y=y_train, clf=clf, legend=2)
    plt.title("%s (train)" % lab)
for clf, lab, grd in zip([clf_fuzz, clf_sk],
                         labels, [[1, 0], [1, 1]]):
    ax = plt.subplot(gs[grd[0], grd[1]])
    fig = plot_decision_regions(X=X_test, y=y_test, clf=clf, legend=2)
    plt.title("%s (test)" % lab)
plt.show()




# .exe produced with PyInstaller fails like this
# Traceback (most recent call last):
#   File "test.py", line 3, in <module>
#   File "PyInstaller\loader\pyimod02_importers.py", line 499, in exec_module
#   File "sklearn\model_selection\__init__.py", line 23, in <module>
#   File "PyInstaller\loader\pyimod02_importers.py", line 499, in exec_module
#   File "sklearn\model_selection\_validation.py", line 32, in <module>
#   File "PyInstaller\loader\pyimod02_importers.py", line 499, in exec_module
#   File "sklearn\metrics\__init__.py", line 42, in <module>
#   File "PyInstaller\loader\pyimod02_importers.py", line 499, in exec_module
#   File "sklearn\metrics\cluster\__init__.py", line 22, in <module>
#   File "PyInstaller\loader\pyimod02_importers.py", line 499, in exec_module
#   File "sklearn\metrics\cluster\_unsupervised.py", line 16, in <module>
#   File "PyInstaller\loader\pyimod02_importers.py", line 499, in exec_module
#   File "sklearn\metrics\pairwise.py", line 33, in <module>
#   File "PyInstaller\loader\pyimod02_importers.py", line 499, in exec_module
#   File "sklearn\metrics\_pairwise_distances_reduction\__init__.py", line 89, in <module>
#   File "PyInstaller\loader\pyimod02_importers.py", line 499, in exec_module
#   File "sklearn\metrics\_pairwise_distances_reduction\_dispatcher.py", line 11, in <module>
#   File "sklearn\metrics\_pairwise_distances_reduction\_base.pyx", line 1, in init sklearn.metrics._pairwise_distances_reduction._base
# ModuleNotFoundError: No module named 'sklearn.metrics._pairwise_distances_reduction._datasets_pair'
# [19380] Failed to execute script 'test' due to unhandled exception!
