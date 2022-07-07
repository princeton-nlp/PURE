import json
import os

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
        self.task = args.task

        if self.task not in ['ace04', 'ace05', 'scierc']:
            if 'entity_output_dir' in vars(args).keys():
                if not os.path.exists(os.path.join(args.entity_output_dir, 'tasklabels.json')):
                    logger.info('Task Labels not found in entity output directory. Please run run_entity.py before running this !')
                    exit(-1)
                logger.info(f'Reading task labels from: ' + os.path.join(args.entity_output_dir, 'tasklabels.json'))
                with open(os.path.join(args.entity_output_dir, 'tasklabels.json')) as f:
                    tasklabeljson = json.load(f)
                self.task_ner_labels = tasklabeljson['task_ner_labels']
                self.task_rel_labels = tasklabeljson['task_rel_labels']
            else:
                if os.path.exists(os.path.join(args.output_dir, 'tasklabels.json')):
                    with open(os.path.join(args.output_dir, 'tasklabels.json')) as f:
                        tasklabeljson = json.load(f)
                    self.task_ner_labels = tasklabeljson['task_ner_labels']
                    self.task_rel_labels = tasklabeljson['task_rel_labels']
                
                logger.info(f'Auto-populating labels for task: {self.task}')
                self.task_ner_labels[self.task] = set()
                self.task_rel_labels[self.task] = set()
                self._autopopulate_tasks(args)
                self.task_ner_labels[self.task] = list(self.task_ner_labels[self.task])
                self.task_rel_labels[self.task] = list(self.task_rel_labels[self.task])
                with open(os.path.join(args.output_dir, 'tasklabels.json'), 'w') as f:
                    tasklabeljson = dict()
                    tasklabeljson['task_ner_labels'] = self.task_ner_labels
                    tasklabeljson['task_rel_labels'] = self.task_rel_labels
                    json.dump(tasklabeljson, f)
                logger.info(f'Saved task labels at: ' + os.path.join(args.output_dir, 'tasklabels.json'))
        else:
            logger.info(f'Using pre-loaded labels for task: {self.task}')
        logger.info('NER Labels: ' + str(self.task_ner_labels[self.task]))
        logger.info('REL Labels: ' + str(self.task_rel_labels[self.task]))

    def _get_unique_labels(self, filepath):
        with open(filepath) as f:
            for line in f.readlines():
                data = json.loads(line.strip())
                if 'ner' in data.keys():
                    for sentence in data['ner']:
                        for ner in sentence:
                            self.task_ner_labels[self.task].add(ner[2])
                if 'relations' in data.keys():
                    for sentence in data['relations']:
                        for rel in sentence:
                            self.task_rel_labels[self.task].add(rel[4])

    def _autopopulate_tasks(self, args):
        self._get_unique_labels(args.train_data)
        self._get_unique_labels(args.dev_data)
        self._get_unique_labels(args.test_data)

    def get_labelmap(self):
        label2id = {}
        id2label = {}
        for i, label in enumerate(self.task_ner_labels[self.task]):
            label2id[label] = i + 1
            id2label[i + 1] = label
        return label2id, id2label
