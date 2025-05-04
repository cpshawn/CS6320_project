import configparser
import csv
import json
import os
import time
import spacy
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

####################################################################################################
# This script calls the fine-tuned OpenAI model for grammar correction.
# The test data comes from:
# (1) Kaggle C4_200M, with original sentence and corrected sentence.
# We compare AI-corrected sentence vs original sentence to see how much it improves,
# and also AI-corrected sentence vs corrected sentence from dataset to see how close they're.
# (2) Manually input sentences. It can correct the sentence and show how much it improves comparing
# with the original sentence.
####################################################################################################

config = configparser.ConfigParser()
config.read('config.ini')

test_data = r'openai_dataset/test_data.jsonl'
os.environ["OPENAI_API_KEY"] = openai_API_Key = config['Parameter']['openaiAPIKey']
openai_assistant_id = config['Parameter']['openaiAssistantID']

client = OpenAI()

summarization = r'summarization.csv'

# Check grammar similarity with spacy: https://spacy.io/usage/linguistic-features#vectors-similarity
nlp = spacy.load("en_core_web_lg")

# Check sentence similarity with Sentence Transformers: https://huggingface.co/sentence-transformers
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Check sentence contradictions with RoBERTa: https://huggingface.co/docs/transformers/en/model_doc/roberta
contradiction_checker = pipeline("text-classification", model="roberta-large-mnli")

def get_corrected_content_from_ai(original_sentence):
    try:
        thread = client.beta.threads.create(messages=[{"role": "user", "content": original_sentence}])
        run = client.beta.threads.runs.create(thread_id=thread.id, assistant_id=openai_assistant_id)

        while run.status != "completed":
            run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
            time.sleep(1)

        messages = client.beta.threads.messages.list(thread_id=thread.id).data
        return messages[0].content[0].text.value
    except:
        return ''

# Ensure both sentences have the same meaning
def check_meaning(original, corrected):
    emb_orig = sentence_model.encode(original, convert_to_tensor=True)
    emb_corr = sentence_model.encode(corrected, convert_to_tensor=True)
    semantic_sim = util.pytorch_cos_sim(emb_orig, emb_corr).item()

    orig_doc = nlp(original)
    corr_doc = nlp(corrected)

    # Check keyword overlap
    orig_keywords = {token.lemma_.lower() for token in orig_doc if token.pos_ in ["NOUN", "VERB", "ADJ"]}
    corr_keywords = {token.lemma_.lower() for token in corr_doc if token.pos_ in ["NOUN", "VERB", "ADJ"]}
    keyword_overlap = len(orig_keywords & corr_keywords) / max(len(orig_keywords), 1)

    contradiction = contradiction_checker(
        {"text": original, "text_pair": corrected},
        top_k=None
    )

    contra_score = 0.0
    for label_info in contradiction:
        if label_info['label'] == 'contradiction':
            contra_score = label_info['score']
            break

    return semantic_sim, keyword_overlap, contra_score

def main():
    while True:
        enter = input('Enter "1" for grammar correction testing on existing data...\nEnter "2" for free typing...\n')
        if enter in ("1", "2"):
            break
        print("Invalid input. Please enter 1 or 2.\n")
    if enter == "1":
        print(f"Testing on data {test_data} created by 'generate_dataset.py'...")
        with open(summarization, 'w', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'Original', 'Target', 'Result',
                'Grammar_OrigRes', 'Semantic_OrigRes',
                'Keywords_OrigRes', 'Contra_OrigRes',
                'Grammar_TargRes', 'Semantic_TargRes',
                'Keywords_TargRes', 'Contra_TargRes'
            ])

        print(f'Reading test data {test_data}')
        with open(test_data, "r") as f:
            data = [json.loads(line) for line in f]
            for item in data:
                original = item['messages'][1]['content']
                target = item['messages'][2]['content']
                result = get_corrected_content_from_ai(original)

                print(f'Original: {original}')
                print(f'Target: {target}')
                print(f'Result: {result}')

                # Compare Result with Original (improvement check)
                sem_sim_or, key_or, contra_or = check_meaning(original, result)
                doc_ori = nlp(original)
                doc_res = nlp(result)
                grammar_or = doc_res.similarity(doc_ori)

                # Compare Result with Target (accuracy check)
                sem_sim_tr, key_tr, contra_tr = check_meaning(target, result)
                doc_target = nlp(target)
                grammar_tr = doc_res.similarity(doc_target)

                # Write to CSV
                print(f'Writing result to {summarization}')
                with open(summarization, 'a', newline='', encoding='utf-8-sig') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([
                        original, target, result,
                        f"{grammar_or:.2f}", f"{sem_sim_or:.2f}",
                        f"{key_or:.2f}", f"{contra_or:.2f}",
                        f"{grammar_tr:.2f}", f"{sem_sim_tr:.2f}",
                        f"{key_tr:.2f}", f"{contra_tr:.2f}"
                    ])

                print(f"\n[Original vs Result]")
                print(f"Grammar: {grammar_or:.2f} | Semantic: {sem_sim_or:.2f}")
                print(f"Keyword Overlap: {key_or:.2f} | Contradiction Risk: {contra_or:.2f}")

                print(f"\n[Target vs Result]")
                print(f"Grammar: {grammar_tr:.2f} | Semantic: {sem_sim_tr:.2f}")
                print(f"Keyword Overlap: {key_tr:.2f} | Contradiction Risk: {contra_tr:.2f}")
                print('=' * 100 + '\n')
    else:
        while True:
            enter = input('Enter text to be corrected, or "Exit" to end...')
            if enter == 'Exit':
                break
            else:
                result = get_corrected_content_from_ai(enter)
                print(f'Original: {enter}')
                print(f'Result: {result}')

                # Compare Result with Original
                sem_sim_or, key_or, contra_or = check_meaning(enter, result)
                doc_ori = nlp(enter)
                doc_res = nlp(result)

                grammar_or = doc_res.similarity(doc_ori)

                print(f"\n[Original vs Result]")
                print(f"Grammar: {grammar_or:.2f} | Semantic: {sem_sim_or:.2f}")
                print(f"Keyword Overlap: {key_or:.2f} | Contradiction Risk: {contra_or:.2f}")

if __name__ == "__main__":
    main()