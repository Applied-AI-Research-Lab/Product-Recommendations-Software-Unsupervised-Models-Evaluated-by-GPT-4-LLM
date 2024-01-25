import os
import openai
from openai import OpenAI
import json
import pandas as pd


class GptEvaluation:
    def __init__(self, model_id):
        self.model_id = model_id

    def gpt_conversation(self, conversation):
        client = OpenAI()
        # response = openai.ChatCompletion.create(
        completion = client.chat.completions.create(
            model=self.model_id,
            messages=conversation
        )
        return completion.choices[0].message

    def gpt_evaluation(self, title, recommendation):
        conversation = []
        conversation.append({'role': 'system',
                             'content': "You're assisting a customer as a salesperson, and they've added the product with the title [" + title + "] to their basket."})
        conversation.append({'role': 'user',
                             'content': "Evaluate whether it's a good idea to recommend adding the product with the title [" + recommendation + "] as an extra. Provide the reason in JSON format why the customer might accept your offer {\"accept\": 1, \"reason\": 'add here the reason for accept'} or the reason they could decline your offer {\"accept\": 0, \"reason\": 'add here the reason for decline'}."})
        conversation = self.gpt_conversation(conversation)  # get the response from GPT model
        content = conversation.content
        try:
            result_dict = json.loads(content)
            return result_dict
        except json.JSONDecodeError:
            # Handle the case where the content is not a valid JSON
            return {'error': 'Invalid JSON format in GPT response'}

    def evaluate(self, model, dataset):
        # Read the CSV file into a DataFrame
        df = pd.read_csv(dataset)

        # Check if the evaluation column is already present in the header
        evaluation_column = model + 'Evaluation'
        if evaluation_column not in df.columns:
            # If not, add the column to the header with NaN as the initial value
            df[evaluation_column] = None

        # Check if the reason column is already present in the header
        reason_column = model + 'Reason'
        if reason_column not in df.columns:
            # If not, add the column to the header with NaN as the initial value
            df[reason_column] = None

        # Update the CSV file with the new header (if columns were added)
        if evaluation_column not in df.columns or reason_column not in df.columns:
            df.to_csv(dataset, index=False)

        # Set the dtype of the reason column to object
        df = df.astype({reason_column: 'object'})

        # Iterate over each row in the DataFrame
        for index, row in df.iterrows():
            title = row['title']
            recommendation = row[model + 'Recommendation']

            # If the evaluation column is NaN
            if pd.isnull(row[evaluation_column]):
                evaluation = self.gpt_evaluation(title, recommendation)
                print(evaluation)

                # continue if error
                if 'error' in evaluation:
                    continue

                # Update the DataFrame with the evaluation result
                df.at[index, evaluation_column] = int(evaluation['accept'])

                # Check if the reason column is NaN before assigning the string value
                if pd.isnull(row[reason_column]):
                    df.at[index, reason_column] = evaluation['reason']

                # Update the CSV file with the new evaluation values
                df.to_csv(dataset, index=False)

                # break

            # Add a delay of 5 seconds (reduced for testing)
            # time.sleep(1)

# Example:
# model_id = 'gpt-4'
# gpt_evaluator = GptEvaluation(model_id)
# gpt_evaluator.evaluate('kMeans', 'Datasets/recommendations.csv')
# gpt_evaluator.evaluate('cbf', 'Datasets/recommendations.csv')
# gpt_evaluator.evaluate('hierarchical', 'Datasets/recommendations.csv')
