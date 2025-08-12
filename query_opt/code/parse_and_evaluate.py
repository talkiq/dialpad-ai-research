import json
import os
import re
from typing import List, Tuple

import evaluate
import pandas as pd


class MetricsEvaluator:
    """
    A class to evaluate model outputs against reference summaries using ROUGE and BERTScore.
    Handles file processing, response cleaning, and metric computation.
    """

    def __init__(self, directory: str = 'outputs'):
        """
        Initialize the evaluator with the directory containing CSV files.
        Loads the metric evaluators once for efficiency.
        """
        self.directory = directory
        self.rouge = evaluate.load('rouge')
        self.bertscore = evaluate.load('bertscore')
        self.files = self._collect_csv_files()

    def _collect_csv_files(self) -> List[str]:
        """
        Collect all CSV files in the specified directory.
        """
        return [
            os.path.join(self.directory, filename)
            for filename in os.listdir(self.directory)
            if os.path.isfile(os.path.join(self.directory, filename)) and filename.endswith('.csv')
        ]

    @staticmethod
    def _clean_response(response: str) -> str:
        """
        Clean and normalize the model response string to make it parseable as JSON.
        This includes heuristic fixes based on observed output patterns.
        """
        response = str(response).strip()
        response = response.split('[/INST]')[-1]
        response = '[' + ('[').join(response.split('[')[1:])
        if response[0] != '[' and ':' in response:
            parts = response.split(":", 1)
            response = parts[1].strip() if len(parts) > 1 else response
        elif response[0] != '[':
            response = '[' + response.strip()
        response = response.replace("\n", " ")
        response = response.replace(",\"", ",")
        response = MetricsEvaluator._replace_quotes(response)
        response = response.replace("```", "")
        response = response.replace("```json", "")
        response = response.replace("'", " ")
        response = response.replace("â€˜", " ")
        response = response.replace("â€œ", " ")
        response = response.replace("[ / JSONObjects]", " ")
        response = re.sub(r'}(?=\s*{)', '},', response)
        response = response.split('#Transcript End')[-1].strip()
        if response[-1] != ']':
            response = (']').join(response.split(']')[:-1]) + ']'
        return response

    @staticmethod
    def _replace_quotes(text: str) -> str:
        """
        Replace double quotes with single quotes in text, except for specific keys like "query" and "summary".
        """
        def replacer(match):
            word = match.group(0)
            if word not in ['"query"', '"summary"']:
                return word.replace('"', "'")
            return word

        pattern = re.compile(r'"(\b(?!query\b|summary\b)[\w-]+\b)"')
        return pattern.sub(replacer, text)

    def _process_row(self, row: pd.Series, predictions: List[str], queries: List[str], references: List[str]) -> Tuple[int, int]:
        """
        Process a single row from the DataFrame.
        Extracts references, cleans and parses the response, appends predictions.
        Returns (match_increment, unmatch_increment).
        """
        json_references = json.loads(row['reference'])
        for item in json_references:
            queries.append(item['query'].strip())
            references.append(item['summary'])

        response = self._clean_response(str(row['summary']))
        match_inc, unmatch_inc = 0, 0
        try:
            json_predictions = json.loads(response)
            for index, item in enumerate(json_predictions):
                if index < len(json_references):
                    try:
                        predictions.append(str(item['summary']))
                    except:
                        predictions.append('')
            match_inc = 1
        except json.JSONDecodeError:
            unmatch_inc = 1

        # Pad with empty strings if fewer predictions than required
        while len(predictions) < len(queries):
            predictions.append('')

        return match_inc, unmatch_inc

    def evaluate_file(self, filename: str) -> None:
        """
        Evaluate a single CSV file: process rows, compute metrics, and print results.
        """
        df = pd.read_csv(filename)
        match, unmatch = 0, 0
        predictions, queries, references = [], [], []

        for _, row in df.iterrows():
            m_inc, u_inc = self._process_row(row, predictions, queries, references)
            match += m_inc
            unmatch += u_inc

        if match + unmatch == 0:
            print(f"{filename}: No rows to evaluate.")
            return

        results_rouge = self.rouge.compute(predictions=predictions, references=references)
        results_bert = self.bertscore.compute(predictions=predictions, references=references, model_type="microsoft/deberta-xlarge-mnli")
        avg_bert_f1 =  sum(results_bert['f1']) / len(results_bert['f1']) if results_bert['f1'] else 0

        accuracy = 100 * (match / (match + unmatch))
        print(
            f"{filename} Format Following Accuracy: {accuracy:.2f}% "
            f"ROUGE: {results_rouge} "
            f"BERTScore: {avg_bert_f1:.4f}"
        )

    def run(self) -> None:
        """
        Run evaluation on all collected CSV files.
        """
        for filename in self.files:
            self.evaluate_file(filename)


if __name__ == "__main__":
    evaluator = MetricsEvaluator()
    evaluator.run()