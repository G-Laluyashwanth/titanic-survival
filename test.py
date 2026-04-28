from src.data import load_train, split_features_and_targets
from src.pipeline import build_pipeline
from src.train import train_and_save

train = load_train()
print(train.head())

X, y = split_features_and_targets(train)
print(X.shape, y.shape)

clf = build_pipeline(X)
print(clf)

train_and_save()