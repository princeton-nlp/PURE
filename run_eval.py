import os
import json
import argparse

from shared.data_structures import Dataset, evaluate_predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction_file', type=str, default=None, required=True)
    args = parser.parse_args()

    data = Dataset(args.prediction_file)
    eval_result = evaluate_predictions(data)
    print('Evaluation result %s'%(args.prediction_file))
    print('NER - P: %f, R: %f, F1: %f'%(eval_result['ner']['precision'], eval_result['ner']['recall'], eval_result['ner']['f1']))
    print('REL - P: %f, R: %f, F1: %f'%(eval_result['relation']['precision'], eval_result['relation']['recall'], eval_result['relation']['f1']))
    print('REL (strict) - P: %f, R: %f, F1: %f'%(eval_result['strict_relation']['precision'], eval_result['strict_relation']['recall'], eval_result['strict_relation']['f1']))
