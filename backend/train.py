"""Usage: python train.py [labelled_wav_dir]  -> writes model.joblib"""
import sys
import joblib
from crynet.model import MODEL_PATH, train

joblib.dump(train(sys.argv[1] if len(sys.argv) > 1 else None), MODEL_PATH)
print("saved", MODEL_PATH)
