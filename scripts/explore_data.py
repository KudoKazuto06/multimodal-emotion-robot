import pandas as pd

train = pd.read_csv("../data/raw/train_sent_emo.csv")
dev = pd.read_csv("../data/raw/dev_sent_emo.csv")
test = pd.read_csv("../data/raw/test_sent_emo.csv")

print("Training shape:", train.shape)
print("Validation shape:", dev.shape)
print("Test shape:", test.shape)

print("\nColumns:")
print(train.columns.tolist())

print("\nSample utterances:")
print(train[["Utterance", "Emotion"]].head(10))

print("\nEmotion distribution:")
print(train["Emotion"].value_counts())

print("\nMissing values:")
print(train.isnull().sum())