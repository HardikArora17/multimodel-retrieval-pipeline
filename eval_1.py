from rouge_score import rouge_scorer
import pandas as pd
import numpy as np
from bert_score import score
from sentence_transformers import SentenceTransformer, util
import torch
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bart_model = SentenceTransformer('facebook/bart-large-cnn')

def calc_bertscore(y_list, x_list, lang="en", model_type="bert-large-uncased"):
    P, R, F1 = score(y_list, x_list, lang=lang, verbose=True, model_type=model_type)
    df['Bert F1 Score'] = F1
    df['Bert Precision'] = P
    df['Bert Recall'] = R

def calc_rougescore(y,x):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    z = []
    for i in range(len(df)):
        scores = scorer.score(y[i], x[i])
        z.append(scores)
    rouge1p = []
    rougeLp = []
    rouge1r = []
    rougeLr = []
    rouge1f1 = []
    rougeLf1 = []
    for i in range(len(df)):
        rouge1p.append(z[i]['rouge1'].precision)
        rouge1r.append(z[i]['rouge1'].recall)
        rouge1f1.append(z[i]['rouge1'].fmeasure)
        rougeLp.append(z[i]['rougeL'].precision)
        rougeLr.append(z[i]['rougeL'].recall)
        rougeLf1.append(z[i]['rougeL'].fmeasure)
    df['Rouge 1 Precision'] = rouge1p
    df['Rouge 1 Recall'] = rouge1r
    df['Rouge 1 F1 Score'] = rouge1f1
    df['Rouge L Precision'] = rougeLp
    df['Rouge L Recall'] = rougeLr
    df['Rouge L F1 Score'] = rougeLf1

def calc_bartscore(y,x):
    bartres = []
    for i in range(len(df)):
        embedding1 = bart_model.encode(y[i], convert_to_tensor = True)
        embedding2 = bart_model.encode(x[i], convert_to_tensor = True)
        similarity_score = util.pytorch_cos_sim(embedding1, embedding2)
        bartres.append(similarity_score)
    bartres = [tensor.item() for tensor in bartres]
    df['Bart Score'] = bartres

save_path = 'csv_data'
for file_name in os.listdir(save_path):
  file_path = os.path.join(save_path, file_name)
  df = pd.read_csv(file_path)
  print(file_name)
  print(f"NO OF QUESTIONS: {len(df)}")
  df = df.rename(columns={'question':'queries', 'gold_answer':'x', 'predicted_answer':'y'})

  calc_bertscore(df['y'].to_list(), df['x'].to_list())
  calc_rougescore(df['y'].to_list(), df['x'].to_list())
  calc_bartscore(df['y'].to_list(), df['x'].to_list())
  
  df.to_csv(os.path.join(save_path,file_name[:-4]+"_metrics.csv"))
  rouge_columns = [
    'Rouge 1 Precision', 'Rouge 1 Recall', 'Rouge 1 F1 Score',
    'Rouge L Precision', 'Rouge L Recall', 'Rouge L F1 Score',
    'Bart Score', 'Bert F1 Score', 'Bert Precision', 'Bert Recall']

  print("\n=== Aggregate Scores ===")
  for col in rouge_columns:
      print(f"{col}: {df[col].mean():.4f}")
  break

