from selfcheckgpt.modeling_selfcheck import SelfCheckNLI
import torch
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
selfcheck_nli = SelfCheckNLI(device=device)

def calc_nli_score():
    main_answer = []
    sentences = []
    main_sentences = []
    scores = []
    samplepass = []
    for i in range(len(df)):
        main_answer.append(df['Sample 1'][i])
    for i in range(len(main_answer)):
        sentences.append(sent_tokenize(main_answer[i]))
    for i in range(len(sentences)):
        ans = ""
        for idx, sentence in enumerate(sentences[i], start=1):
            ans+= f"{idx}. {sentence}"
        main_sentences.append(ans)
    for i in range(len(df)):
        df['Sample 1'][i] = main_sentences[i]
    for i in range(len(df)):
        samples = []
        for j in range(10):
            samples.append(df[f"Sample {j+2}"][i])
        samplepass.append(samples)
    for i in range(len(sentences)):
        score = selfcheck_nli.predict(
            sentences = sentences[i],                         
            sampled_passages = samplepass[i]
        )
        scores.append(score)
    df['Scores'] = scores

csv_file_path = ''
df = pd.read_csv(csv_file_path)
calc_nli_score()
df.to_csv('selfcheck-nli-score.csv')
