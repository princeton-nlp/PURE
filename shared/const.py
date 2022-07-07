import json

task_ner_labels = {
    'ace04': ['FAC', 'WEA', 'LOC', 'VEH', 'GPE', 'ORG', 'PER'],
    'ace05': ['FAC', 'WEA', 'LOC', 'VEH', 'GPE', 'ORG', 'PER'],
    'scierc': ['Method', 'OtherScientificTerm', 'Task', 'Generic', 'Material', 'Metric'],
}

task_rel_labels = {
    'ace04': ['PER-SOC', 'OTHER-AFF', 'ART', 'GPE-AFF', 'EMP-ORG', 'PHYS'],
    'ace05': ['ART', 'ORG-AFF', 'GEN-AFF', 'PHYS', 'PER-SOC', 'PART-WHOLE'],
    'scierc': ['PART-OF', 'USED-FOR', 'FEATURE-OF', 'CONJUNCTION', 'EVALUATE-FOR', 'HYPONYM-OF', 'COMPARE'],
}

def get_labelmap(label_list):
    label2id = {}
    id2label = {}
    for i, label in enumerate(label_list):
        label2id[label] = i + 1
        id2label[i + 1] = label
    return label2id, id2label

class TaskLabels:
    def __init__(self, args, logger):
        self.task_ner_labels = {
            'ace04': ['FAC', 'WEA', 'LOC', 'VEH', 'GPE', 'ORG', 'PER'],
            'ace05': ['FAC', 'WEA', 'LOC', 'VEH', 'GPE', 'ORG', 'PER'],
            'scierc': ['Method', 'OtherScientificTerm', 'Task', 'Generic', 'Material', 'Metric'],
        }
        self.task_rel_labels = {
            'ace04': ['PER-SOC', 'OTHER-AFF', 'ART', 'GPE-AFF', 'EMP-ORG', 'PHYS'],
            'ace05': ['ART', 'ORG-AFF', 'GEN-AFF', 'PHYS', 'PER-SOC', 'PART-WHOLE'],
            'scierc': ['PART-OF', 'USED-FOR', 'FEATURE-OF', 'CONJUNCTION', 'EVALUATE-FOR', 'HYPONYM-OF', 'COMPARE'],
        }

        if args.task not in ['ace04', 'ace05', 'scierc']:
            self.task_ner_labels[args.task] = set()
            self.task_rel_labels[args.task] = set()
            self._autopopulate_tasks(args)
            self.task_ner_labels[args.task] = list(self.task_ner_labels[args.task])
            self.task_rel_labels[args.task] = list(self.task_rel_labels[args.task])
            logger.info(f'Auto-populating labels for task: {args.task}')
        else:
            logger.info(f'Using pre-loaded labels for task: {args.task}')
        logger.info('NER Labels: ' + str(self.task_ner_labels[args.task]))
        logger.info('REL Labels: ' + str(self.task_rel_labels[args.task]))

    def _get_unique_labels(self, filepath, task):
        with open(filepath) as f:
            for line in f.readlines():
                data = json.loads(line.strip())
                for sentence in data['ner']:
                    for ner in sentence:
                        self.task_ner_labels[task].add(ner[2])
                for sentence in data['relations']:
                    for rel in sentence:
                        self.task_rel_labels[task].add(rel[4])

    def _autopopulate_tasks(self, args):
        self._get_unique_labels(args.train_data, args.task)
        self._get_unique_labels(args.dev_data, args.task)
        self._get_unique_labels(args.test_data, args.task)
