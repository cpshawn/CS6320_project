import configparser
import os
import time
from openai import OpenAI

###################################################################################################################
# This script calls OpenAI API to perform fine-tuning for a better grammar correction model.
# The training dataset comes from Kaggle C4_200M,
# consisting a series of examples of original sentences with grammar or spelling issues and the corrected sentences
###################################################################################################################

config = configparser.ConfigParser()
config.read('config.ini')
os.environ["OPENAI_API_KEY"] = config['Parameter']['openaiAPIKey']

def main():
    training_file_path = 'History/1/training_data.jsonl'
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    with open(training_file_path, "rb") as f:
        file_response = client.files.create(file=f, purpose='fine-tune')
    training_file_id = file_response.id
    training_response = client.fine_tuning.jobs.create(training_file=training_file_id, model="gpt-4o-2024-08-06")
    fine_tune_id = training_response.id
    print(f"Fine-tuning started. Job ID: {fine_tune_id}")
    while True:
        response = client.fine_tuning.jobs.retrieve(fine_tune_id)
        status = response.status
        print(f"Fine-tuning status: {status}")
        if status == "succeeded":
            print("Fine-tuning completed successfully.")
            return response.fine_tuned_model  # This is the new model ID
        elif status in ["failed", "cancelled"]:
            print("Fine-tuning failed or was cancelled.")
            return None
        time.sleep(30)

if __name__ == "__main__":
    main()