"""
This code is based on the file in SpanBERT repo: https://github.com/facebookresearch/SpanBERT/blob/master/code/run_tacred.py
"""

import argparse
import logging
import os
import random
import time
import json
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter

from torch.nn import CrossEntropyLoss

from transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from relation.models import BertForRelationApprox
# from relation.models import AlbertForRelationApprox
from transformers import AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup

from relation.utils import generate_relation_data, decode_sample_id
from shared.const import task_rel_labels, task_ner_labels
from shared.data_structures import Dataset

CLS = "[CLS]"
SEP = "[SEP]"

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, position_ids, input_mask, labels, sub_obj_ids, sub_obj_masks, meta, max_seq_length):
        # Padding
        self.num_labels = len(labels)
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        position_ids += padding
        input_mask += padding

        label_padding = [0] * (max_seq_length // 4 - len(labels))
        ids_padding = [[0, 0]] * (max_seq_length // 4 - len(labels))
        labels += label_padding
        sub_obj_masks += label_padding
        sub_obj_ids += ids_padding

        # Compute the attention mask matrix
        attention_mask = []
        for _, from_mask in enumerate(input_mask):
            attention_mask_i = []
            for to_mask in input_mask:
                if to_mask <= 1:
                    attention_mask_i.append(to_mask)
                elif from_mask == to_mask and from_mask > 0:
                    attention_mask_i.append(1)
                else:
                    attention_mask_i.append(0)
            attention_mask.append(attention_mask_i)

        self.input_ids = input_ids
        self.position_ids = position_ids
        self.original_input_mask = input_mask
        self.input_mask = attention_mask
        self.segment_ids = [0] * len(input_ids)
        self.labels = labels
        self.sub_obj_ids = sub_obj_ids
        self.sub_obj_masks = sub_obj_masks
        self.meta = meta

def add_marker_tokens(tokenizer, ner_labels):
    new_tokens = ['<SUBJ_START>', '<SUBJ_END>', '<OBJ_START>', '<OBJ_END>']
    for label in ner_labels:
        new_tokens.append('<SUBJ_START=%s>'%label)
        new_tokens.append('<SUBJ_END=%s>'%label)
        new_tokens.append('<OBJ_START=%s>'%label)
        new_tokens.append('<OBJ_END=%s>'%label)
    for label in ner_labels:
        new_tokens.append('<SUBJ=%s>'%label)
        new_tokens.append('<OBJ=%s>'%label)
    tokenizer.add_tokens(new_tokens)
    logger.info('# vocab after adding markers: %d'%len(tokenizer))

def get_features_from_file(filename, label2id, max_seq_length, tokenizer, special_tokens, use_gold, context_window, batch_computation=False, unused_tokens=True):

    def get_special_token(w):
        if w not in special_tokens:
            if unused_tokens:
                special_tokens[w] = "[unused%d]" % (len(special_tokens) + 1)
            else:
                special_tokens[w] = ('<' + w + '>').lower()
        return special_tokens[w]

    num_shown_examples = 0
    features = []

    nrel = 0

    data = Dataset(filename)
    for doc in data:
        for i, sent in enumerate(doc):     
            sid = i           
            nrel += len(sent.relations)

            if use_gold:
                sent_ner = sent.ner
            else:
                sent_ner = sent.predicted_ner

            if len(sent_ner) <= 1:
                continue
            
            text = sent.text
            sent_start = 0
            sent_end = len(text)

            if context_window > 0:
                add_left = (context_window-len(sent.text)) // 2
                add_right = (context_window-len(sent.text)) - add_left

                j = i - 1
                while j >= 0 and add_left > 0:
                    context_to_add = doc[j].text[-add_left:]
                    text = context_to_add + text
                    add_left -= len(context_to_add)
                    sent_start += len(context_to_add)
                    sent_end += len(context_to_add)
                    j -= 1

                j = i + 1
                while j < len(doc) and add_right > 0:
                    context_to_add = doc[j].text[:add_right]
                    text = text + context_to_add
                    add_right -= len(context_to_add)
                    j += 1

            tokens = [CLS]
            token_start = {}
            token_end = {}
            for i, token in enumerate(text):
                token_start[i] = len(tokens)
                for sub_token in tokenizer.tokenize(token):
                    tokens.append(sub_token)
                token_end[i] = len(tokens) - 1
            tokens.append(SEP)
            num_tokens = len(tokens)
            assert(num_tokens + 4 <= max_seq_length)

            position_ids = list(range(len(tokens)))
            marker_mask = 1
            input_mask = [1] * len(tokens)
            labels = []
            sub_obj_ids = []
            sub_obj_masks = []
            sub_obj_pairs = []

            gold_rel = {}
            for rel in sent.relations:
                gold_rel[rel.pair] = rel.label
            for x in range(len(sent_ner)):
                for y in range(len(sent_ner)):
                    if x == y:
                        continue
                    sub = sent_ner[x]
                    obj = sent_ner[y]
                    label = label2id[gold_rel.get((sub.span, obj.span), 'no_relation')]
                    SUBJECT_START_NER = get_special_token("SUBJ_START=%s"%sub.label)
                    SUBJECT_END_NER = get_special_token("SUBJ_END=%s"%sub.label)
                    OBJECT_START_NER = get_special_token("OBJ_START=%s"%obj.label)
                    OBJECT_END_NER = get_special_token("OBJ_END=%s"%obj.label)
                    
                    if (len(tokens) + 4 > max_seq_length) or (not(batch_computation) and len(tokens) > num_tokens):
                        input_ids = tokenizer.convert_tokens_to_ids(tokens)
                        features.append(
                            InputFeatures(input_ids=input_ids,
                            position_ids=position_ids,
                            input_mask=input_mask,
                            labels=labels,
                            sub_obj_ids=sub_obj_ids,
                            sub_obj_masks=sub_obj_masks,
                            meta={'doc_id': doc._doc_key, 'sent_id': sid, 'sub_obj_pairs': sub_obj_pairs},
                            max_seq_length=max_seq_length,))
                        
                        tokens = tokens[:num_tokens]
                        position_ids = list(range(len(tokens)))
                        marker_mask = 1
                        input_mask = [1] * len(tokens)
                        labels = []
                        sub_obj_ids = []
                        sub_obj_masks = []
                        sub_obj_pairs = []

                    tokens = tokens + [SUBJECT_START_NER, SUBJECT_END_NER, OBJECT_START_NER, OBJECT_END_NER]
                    position_ids = position_ids + [token_start[sent_start+sub.span.start_sent], token_end[sent_start+sub.span.end_sent], token_start[sent_start+obj.span.start_sent], token_end[sent_start+obj.span.end_sent]]
                    
                    marker_mask += 1
                    input_mask  = input_mask + [marker_mask] * 4

                    labels.append(label)
                    sub_obj_ids.append([len(tokens) - 4, len(tokens) - 2])
                    sub_obj_masks.append(1)
                    sub_obj_pairs.append([sub.span, obj.span])

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            features.append(
                InputFeatures(input_ids=input_ids,
                position_ids=position_ids,
                input_mask=input_mask,
                labels=labels,
                sub_obj_ids=sub_obj_ids,
                sub_obj_masks=sub_obj_masks,
                meta={'doc_id': doc._doc_key, 'sent_id': sid, 'sub_obj_pairs': sub_obj_pairs},
                max_seq_length=max_seq_length,))

            if num_shown_examples < 20:
                num_shown_examples += 1
                logger.info("*** Example ***")
                logger.info("guid: %s" % (doc._doc_key))
                logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in features[-1].input_ids]))
                logger.info("position_ids: %s" % " ".join([str(x) for x in features[-1].position_ids]))
                logger.info("input_mask: %s" % " ".join([str(x) for x in features[-1].original_input_mask]))
                logger.info("labels: %s" % " ".join([str(x) for x in features[-1].labels]))
                logger.info("sub_obj_ids: %s" % " ".join(['(%d, %d)'%(x[0], x[1]) for x in features[-1].sub_obj_ids]))
                logger.info("sub_obj_masks: %s" % " ".join([str(x) for x in features[-1].sub_obj_masks]))
                logger.info("sub_obj_spans: %s" % " ".join([str(x) for x in features[-1].meta['sub_obj_pairs']]))
 
    max_num_tokens = 0
    max_num_pairs = 0
    num_label = 0
    for feat in features:
        if len(feat.input_ids) > max_num_tokens:
            max_num_tokens = len(feat.input_ids)
        if len(feat.sub_obj_ids) > max_num_pairs:
            max_num_pairs = len(feat.sub_obj_ids)
        num_label += feat.num_labels

    logger.info('Max # tokens: %d'%max_num_tokens)
    logger.info('Max # pairs: %d'%max_num_pairs)

    logger.info('Total labels: %d'%(num_label))
    logger.info('# labels per sample on average: %f'%(num_label / len(features)))

    return data, features, nrel


def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def compute_f1(preds, labels, e2e_ngold):
    n_gold = n_pred = n_correct = 0
    for pred, label in zip(preds, labels):
        if pred != 0:
            n_pred += 1
        if label != 0:
            n_gold += 1
        if (pred != 0) and (label != 0) and (pred == label):
            n_correct += 1
    if n_correct == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    else:
        prec = n_correct * 1.0 / n_pred
        recall = n_correct * 1.0 / n_gold
        if prec + recall > 0:
            f1 = 2.0 * prec * recall / (prec + recall)
        else:
            f1 = 0.0

        if e2e_ngold is not None:
            e2e_recall = n_correct * 1.0 / e2e_ngold
            e2e_f1 = 2.0 * prec * e2e_recall / (prec + e2e_recall)
        else:
            e2e_recall = e2e_f1 = 0.0
        return {'precision': prec, 'recall': e2e_recall, 'f1': e2e_f1, 'task_recall': recall, 'task_f1': f1, 
        'n_correct': n_correct, 'n_pred': n_pred, 'n_gold': e2e_ngold, 'task_ngold': n_gold}


def evaluate(model, device, eval_dataloader, e2e_ngold=None, verbose=True):
    c_time = time.time()
    model.eval()
    eval_loss = 0
    nb_eval_steps = 0
    all_preds = []
    all_labels = []
    for input_ids, input_position, input_mask, segment_ids, labels, sub_obj_ids, sub_obj_masks in eval_dataloader:
        batch_labels = labels
        batch_masks = sub_obj_masks
        input_ids = input_ids.to(device)
        input_position = input_position.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        labels = labels.to(device)
        sub_obj_ids = sub_obj_ids.to(device)
        sub_obj_masks = sub_obj_masks.to(device)
        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask, labels=None, sub_obj_ids=sub_obj_ids, sub_obj_masks=None, input_position=input_position)
        loss_fct = CrossEntropyLoss()
        active_loss = (sub_obj_masks.view(-1) == 1)
        active_logits = logits.view(-1, logits.shape[-1])
        active_labels = torch.where(
            active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
        )
        tmp_eval_loss = loss_fct(active_logits, active_labels)
        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1

        batch_preds = np.argmax(logits.detach().cpu().numpy(), axis=2)
        
        for i in range(batch_preds.shape[0]):
            for j in range(batch_preds.shape[1]):
                if batch_masks[i][j] == 1:
                    all_preds.append(batch_preds[i][j])
                    all_labels.append(batch_labels[i][j])

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    eval_loss = eval_loss / nb_eval_steps
    result = compute_f1(all_preds, all_labels, e2e_ngold=e2e_ngold)
    result['accuracy'] = simple_accuracy(all_preds, all_labels)
    result['eval_loss'] = eval_loss
    if verbose:
        logger.info("***** Eval results (used time: %.3f s) *****"%(time.time()-c_time))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
    return all_preds, result

def print_pred_json(eval_data, eval_features, preds, id2label, output_file):
    rels = dict()
    p = 0
    for feat in eval_features:
        doc_sent = '%s@%d'%(feat.meta['doc_id'], feat.meta['sent_id'])
        if doc_sent not in rels:
            rels[doc_sent] = []
        for pair in feat.meta['sub_obj_pairs']:
            sub = pair[0]
            obj = pair[1]
            # get the next prediction
            pred = preds[p]
            p += 1
            if pred != 0:
                rels[doc_sent].append([sub.start_doc, sub.end_doc, obj.start_doc, obj.end_doc, id2label[pred]])

    js = eval_data.js
    for doc in js:
        doc['predicted_relations'] = []
        for sid in range(len(doc['sentences'])):
            k = '%s@%d'%(doc['doc_key'], sid)
            doc['predicted_relations'].append(rels.get(k, []))
    
    logger.info('Output predictions to %s..'%(output_file))
    with open(output_file, 'w') as f:
        f.write('\n'.join(json.dumps(doc) for doc in js))

def setseed(seed):
    random.seed(seed)
    np.random.seed(args.seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_trained_model(output_dir, model, tokenizer):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    logger.info('Saving model to %s'%output_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)
    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(output_dir)

def main(args):
    if 'albert' in args.model:
        raise ValueError("ALBERT approximation model is not supported by the current implementation, as Huggingface's Transformers ALBERT doesn't support an attention mask with a shape of [batch_size, from_seq_length, to_seq_length].")
        # RelationModel = AlbertForRelationApprox
        args.add_new_tokens = True
    else:
        RelationModel = BertForRelationApprox

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()

    setseed(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.do_train:
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "train.log"), 'w'))
    else:
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "eval.log"), 'w'))
    logger.info(sys.argv)
    logger.info(args)
    logger.info("device: {}, n_gpu: {}".format(
        device, n_gpu))

    # get label_list
    if os.path.exists(os.path.join(args.output_dir, 'label_list.json')):
        with open(os.path.join(args.output_dir, 'label_list.json'), 'r') as f:
            label_list = json.load(f)
    else:
        label_list = [args.negative_label] + task_rel_labels[args.task]
        with open(os.path.join(args.output_dir, 'label_list.json'), 'w') as f:
            json.dump(label_list, f)
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}
    num_labels = len(label_list)

    tokenizer = AutoTokenizer.from_pretrained(args.model, do_lower_case=args.do_lower_case)
    if args.add_new_tokens:
        add_marker_tokens(tokenizer, task_ner_labels[args.task])

    if os.path.exists(os.path.join(args.output_dir, 'special_tokens.json')):
        with open(os.path.join(args.output_dir, 'special_tokens.json'), 'r') as f:
            special_tokens = json.load(f)
    else:
        special_tokens = {}

    if args.do_eval and (args.do_train or not(args.eval_test)):
        eval_dataset, eval_features, eval_nrel = get_features_from_file(
            os.path.join(args.entity_output_dir, args.entity_predictions_dev), label2id, args.max_seq_length, tokenizer, special_tokens, use_gold=args.eval_with_gold, context_window=args.context_window, batch_computation=args.batch_computation, unused_tokens=not(args.add_new_tokens))
        logger.info("***** Dev *****")
        logger.info("  Num examples = %d", len(eval_features))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_position_ids = torch.tensor([f.position_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_labels = torch.tensor([f.labels for f in eval_features], dtype=torch.long)
        all_sub_obj_ids = torch.tensor([f.sub_obj_ids for f in eval_features], dtype=torch.long)
        all_sub_obj_masks = torch.tensor([f.sub_obj_masks for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_position_ids, all_input_mask, all_segment_ids, all_labels, all_sub_obj_ids, all_sub_obj_masks)
        eval_dataloader = DataLoader(eval_data, batch_size=args.eval_batch_size)
    with open(os.path.join(args.output_dir, 'special_tokens.json'), 'w') as f:
        json.dump(special_tokens, f)

    if args.do_train:
        train_dataset, train_features, train_nrel = get_features_from_file(
            args.train_file, label2id, args.max_seq_length, tokenizer, special_tokens, use_gold=True, context_window=args.context_window, batch_computation=args.batch_computation, unused_tokens=not(args.add_new_tokens))
        if args.train_mode == 'sorted' or args.train_mode == 'random_sorted':
            train_features = sorted(train_features, key=lambda f: np.sum(f.input_mask))
        else:
            random.shuffle(train_features)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_position_ids = torch.tensor([f.position_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_labels = torch.tensor([f.labels for f in train_features], dtype=torch.long)
        all_sub_obj_ids = torch.tensor([f.sub_obj_ids for f in train_features], dtype=torch.long)
        all_sub_obj_masks = torch.tensor([f.sub_obj_masks for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_position_ids, all_input_mask, all_segment_ids, all_labels, all_sub_obj_ids, all_sub_obj_masks)
        train_dataloader = DataLoader(train_data, batch_size=args.train_batch_size)
        train_batches = [batch for batch in train_dataloader]

        num_train_optimization_steps = len(train_dataloader) * args.num_train_epochs

        logger.info("***** Training *****")
        logger.info("  Num examples = %d", len(train_features))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        best_result = None
        eval_step = max(1, len(train_batches) // args.eval_per_epoch)
        
        lr = args.learning_rate
        model = RelationModel.from_pretrained(
            args.model, cache_dir=str(PYTORCH_PRETRAINED_BERT_CACHE), num_rel_labels=num_labels)
        if hasattr(model, 'bert'):
            model.bert.resize_token_embeddings(len(tokenizer))
        elif hasattr(model, 'albert'):
            model.albert.resize_token_embeddings(len(tokenizer))
        else:
            raise TypeError("Unknown model class")

        model.to(device)
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)

        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer
                        if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer
                        if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr, correct_bias=not(args.bertadam))
        scheduler = get_linear_schedule_with_warmup(optimizer, int(num_train_optimization_steps * args.warmup_proportion), num_train_optimization_steps)

        start_time = time.time()
        global_step = 0
        tr_loss = 0
        nb_tr_examples = 0
        nb_tr_steps = 0
        for epoch in range(int(args.num_train_epochs)):
            model.train()
            logger.info("Start epoch #{} (lr = {})...".format(epoch, lr))
            if args.train_mode == 'random' or args.train_mode == 'random_sorted':
                random.shuffle(train_batches)
            for step, batch in enumerate(train_batches):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_position, input_mask, segment_ids, labels, sub_obj_ids, sub_obj_masks = batch
                loss = model(input_ids, segment_ids, input_mask, labels=labels, sub_obj_ids=sub_obj_ids, sub_obj_masks=sub_obj_masks, input_position=input_position)
                if n_gpu > 1:
                    loss = loss.mean()

                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if (step + 1) % eval_step == 0:
                    logger.info('Epoch: {}, Step: {} / {}, used_time = {:.2f}s, loss = {:.6f}'.format(
                                epoch, step + 1, len(train_batches),
                                time.time() - start_time, tr_loss / nb_tr_steps))
                    save_model = False
                    if args.do_eval:
                        preds, result = evaluate(model, device, eval_dataloader, e2e_ngold=eval_nrel)
                        model.train()
                        result['global_step'] = global_step
                        result['epoch'] = epoch
                        result['learning_rate'] = lr
                        result['batch_size'] = args.train_batch_size

                        if (best_result is None) or (result[args.eval_metric] > best_result[args.eval_metric]):
                            best_result = result
                            logger.info("!!! Best dev %s (lr=%s, epoch=%d): %.2f" %
                                        (args.eval_metric, str(lr), epoch, result[args.eval_metric] * 100.0))
                            save_trained_model(args.output_dir, model, tokenizer)

    evaluation_results = {}
    if args.do_eval:
        if args.eval_test:
            eval_dataset, eval_features, eval_nrel = get_features_from_file(
                os.path.join(args.entity_output_dir, args.entity_predictions_test), label2id, args.max_seq_length, tokenizer, special_tokens, use_gold=args.eval_with_gold, context_window=args.context_window, batch_computation=args.batch_computation, unused_tokens=not(args.add_new_tokens))
            logger.info("***** Test *****")
            logger.info("  Num examples = %d", len(eval_features))
            logger.info("  Batch size = %d", args.eval_batch_size)
            all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
            all_position_ids = torch.tensor([f.position_ids for f in eval_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
            all_labels = torch.tensor([f.labels for f in eval_features], dtype=torch.long)
            all_sub_obj_ids = torch.tensor([f.sub_obj_ids for f in eval_features], dtype=torch.long)
            all_sub_obj_masks = torch.tensor([f.sub_obj_masks for f in eval_features], dtype=torch.long)
            eval_data = TensorDataset(all_input_ids, all_position_ids, all_input_mask, all_segment_ids, all_labels, all_sub_obj_ids, all_sub_obj_masks)
            eval_dataloader = DataLoader(eval_data, batch_size=args.eval_batch_size)
        model = RelationModel.from_pretrained(args.output_dir, num_rel_labels=num_labels)
        model.to(device)
        preds, result = evaluate(model, device, eval_dataloader, e2e_ngold=eval_nrel)
        logger.info('*** Evaluation Results ***')
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
        print_pred_json(eval_dataset, eval_features, preds, id2label, os.path.join(args.output_dir, args.prediction_file))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, type=str, required=True)
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--eval_per_epoch", default=10, type=int,
                        help="How many times it evaluates on dev set per epoch")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--negative_label", default="no_relation", type=str)
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--train_file", default=None, type=str, help="The path of the training data.")
    parser.add_argument("--train_mode", type=str, default='random_sorted', choices=['random', 'sorted', 'random_sorted'])
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--eval_test", action="store_true", help="Whether to evaluate on final test set.")
    parser.add_argument("--eval_with_gold", action="store_true", help="Whether to evaluate the relation model with gold entities provided.")
    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_metric", default="f1", type=str)
    parser.add_argument("--learning_rate", default=None, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--bertadam", action="store_true", help="If bertadam, then set correct_bias = False")

    parser.add_argument("--entity_output_dir", type=str, default=None, help="The directory of the prediction files of the entity model")
    parser.add_argument("--entity_predictions_dev", type=str, default="ent_pred_dev.json", help="The entity prediction file of the dev set")
    parser.add_argument("--entity_predictions_test", type=str, default="ent_pred_test.json", help="The entity prediction file of the test set")

    parser.add_argument("--prediction_file", type=str, default="predictions.json", help="The prediction filename for the relation model")

    parser.add_argument('--task', type=str, default=None, required=True, choices=['ace04', 'ace05', 'scierc'])
    parser.add_argument('--context_window', type=int, default=0)

    parser.add_argument('--add_new_tokens', action='store_true', 
                        help="Whether to add new tokens as marker tokens instead of using [unusedX] tokens.")
    parser.add_argument('--batch_computation', action='store_true',
                        help="Whether to use batch computation to speedup the inference.")

    args = parser.parse_args()
    main(args)
