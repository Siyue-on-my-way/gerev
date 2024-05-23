from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline
import torch


import os
print(os.listdir('../greve_model'))

bi_encoder = SentenceTransformer('../greve_model/multi-qa-MiniLM-L6-cos-v1')

cross_encoder_small = CrossEncoder('../greve_model/ms-marco-TinyBERT-L-2-v2')
cross_encoder_large = CrossEncoder('../greve_model/ms-marco-MiniLM-L-6-v2')

qa_model = pipeline('question-answering', model='../greve_model/roberta-base-squad2')

