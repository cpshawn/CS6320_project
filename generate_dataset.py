import json
import random

from datasets import load_dataset

####################################################################################################
# This script pulls data from Kaggle C4_200M dataset for grammar correction
# and generate training/testing data files with a ratio of 80% ---- 20%.
# The generated datasets fit into OpenAI API inputs, can be uploaded for fine-tuning use directly.
# Therefore, the training data will be used for fine-tuning in train_grammar_model.py,
# and the test data will be used in test_grammar_model.py for evaluating the model behavior
####################################################################################################


# For a total of 500 examples, I split the examples into 80% training data, 10% test data and 10% validation data
total = 500
train_size = int(total * 0.8)
test_size = total - train_size

prompt = """
You are a professional grammar correction AI. Your task is to:
1. Fix grammatical, punctuation, and spelling errors
2. Convert Unicode representations of special characters/symbols back to their original form
3. Preserve original meaning, tone, and style

Special character handling rules:
- Recognize common Unicode sequences (e.g., \u201C → “, \u2019 → ’, \u00A0 → space)
- Only convert Unicode if it represents a common symbol/punctuation
- Leave Unicode as-is if it's explicitly part of the content

Make only necessary edits. Respond ONLY with corrected text. Notice there might be multiple issues in one input.

Examples:
Original: "He said \u201CHello world\u201D but it\u2019s cold."
Corrected: "He said "Hello world" but it's cold."

Original: "theyre going to the caf\u00e9 at 3pm"
Corrected: "They're going to the café at 3 p.m."
"""

# The training dataset must match OpenAI's fine-tuning requirements:
# https://platform.openai.com/docs/guides/fine-tuning
# And for ease, I have both training and testing datasets created in the same format
def generate_dataset(data_list, tag):
    formatted_data = []
    for pair in data_list:
        formatted_data.append({
            "messages": [
                {
                    "role": "system",
                    "content": prompt
                },
                {
                    "role": "user",
                    "content": pair['input']
                },
                {
                    "role": "assistant",
                    "content": pair['output']
                }
            ]
        })

    filename = f"openai_dataset/{tag}.jsonl"

    # Write to JSON Lines file
    with open(filename, 'w', encoding='utf-8') as f:
        for entry in formatted_data:
            f.write(json.dumps(entry) + '\n')


def main():
    print("Collecting data from hugging face...")

    # Data source: C4_200M from kaggle: https://www.kaggle.com/datasets/felixstahlberg/the-c4-200m-dataset-for-gec
    dataset = load_dataset("liweili/c4_200m", split="train", streaming=True, trust_remote_code=True)
    shuffled_dataset = dataset.shuffle(seed=random.randint(0, 2**32 - 1), buffer_size=10000)
    ds = shuffled_dataset.take(total)
    print("Data collected, splitting...")

    # Generating testing data
    test_data = list(ds.take(test_size))
    generate_dataset(test_data, 'test_data')
    print("Test data generated.")

    # Generating training data
    training_data = list(ds.skip(test_size))
    generate_dataset(training_data, 'training_data')
    print("Training data generated.")
    print('=' * 100)

if __name__ == '__main__':
    main()
