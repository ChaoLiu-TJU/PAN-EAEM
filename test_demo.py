import torch
import json
import models
import numpy as np
import sys
from torch.autograd import Variable
from fewshot_re_kit.data_loader import JSONFileDataLoader
from fewshot_re_kit.framework import FewShotREFramework
from fewshot_re_kit.sentence_encoder import CNNSentenceEncoder, CNNEntityEncoder
from models.pan import PAN

input_filename = './data/pred_5_5.json'

max_length = 40
entity_length = 6
N = 5
K = 5
test_data_loader = JSONFileDataLoader('./data/train.json', './data/glove.6B.50d.json', max_length=max_length)
framework = FewShotREFramework(None, None, test_data_loader)
sentence_encoder = CNNSentenceEncoder(test_data_loader.word_vec_mat, max_length)
entity_encoder = CNNEntityEncoder(test_data_loader.word_vec_mat, entity_length)
model = PAN(sentence_encoder, entity_encoder).eval()
framework.set_model(model, 'checkpoint_55/pan.pth.tar')

content = json.load(open(input_filename))

ans = []
for i in content:
	N = len(i['meta_train'])
	K = len(i['meta_train'][0])
	Q = 1
	support = {'word': [], 'head': [], 'tail': [], 'pos1': [], 'pos2': [], 'mask': []}
	query = {'word': [], 'head': [], 'tail': [], 'pos1': [], 'pos2': [], 'mask': []}
	for j in i['meta_train']:
		support_set = {'word': [], 'head': [], 'tail': [], 'pos1': [], 'pos2': [], 'mask': []}
		for k in j:
			cur_ref_data_word, cur_ref_data_head, cur_ref_data_tail, data_pos1, data_pos2, data_mask, data_length = test_data_loader.lookup(k)
			support_set['word'].append(cur_ref_data_word)
			support_set['head'].append(cur_ref_data_head)
			support_set['tail'].append(cur_ref_data_tail)
			support_set['pos1'].append(data_pos1)
			support_set['pos2'].append(data_pos2)
			support_set['mask'].append(data_mask)
		support['word'].append(support_set['word'])
		support['head'].append(support_set['head'])
		support['tail'].append(support_set['tail'])
		support['pos1'].append(support_set['pos1'])
		support['pos2'].append(support_set['pos2'])
		support['mask'].append(support_set['mask'])
	for j in range(N):
		cur_ref_data_word, cur_ref_data_head, cur_ref_data_tail, data_pos1, data_pos2, data_mask, data_length = test_data_loader.lookup(i['meta_test'])
		query_set = {'word': [], 'head': [], 'tail': [], 'pos1': [], 'pos2': [], 'mask': []}
		query_set['word'].append(cur_ref_data_word)
		query_set['head'].append(cur_ref_data_head)
		query_set['tail'].append(cur_ref_data_tail)
		query_set['pos1'].append(data_pos1)
		query_set['pos2'].append(data_pos2)
		query_set['mask'].append(data_mask)
		query['word'].append(query_set['word'])
		query['head'].append(query_set['head'])
		query['tail'].append(query_set['tail'])
		query['pos1'].append(query_set['pos1'])
		query['pos2'].append(query_set['pos2'])
		query['mask'].append(query_set['mask'])

	support['word'] = Variable(torch.from_numpy(np.stack(support['word'], 0)).long().view(-1, max_length))
	support['head'] = Variable(torch.from_numpy(np.stack(support['head'], 0)).long().view(-1, entity_length))
	support['tail'] = Variable(torch.from_numpy(np.stack(support['tail'], 0)).long().view(-1, entity_length))
	support['pos1'] = Variable(torch.from_numpy(np.stack(support['pos1'], 0)).long().view(-1, max_length)) 
	support['pos2'] = Variable(torch.from_numpy(np.stack(support['pos2'], 0)).long().view(-1, max_length)) 
	support['mask'] = Variable(torch.from_numpy(np.stack(support['mask'], 0)).long().view(-1, max_length)) 
	query['word'] = Variable(torch.from_numpy(np.stack(query['word'], 0)).long().view(-1, max_length))
	query['head'] = Variable(torch.from_numpy(np.stack(query['head'], 0)).long().view(-1, entity_length))
	query['tail'] = Variable(torch.from_numpy(np.stack(query['tail'], 0)).long().view(-1, entity_length))
	query['pos1'] = Variable(torch.from_numpy(np.stack(query['pos1'], 0)).long().view(-1, max_length)) 
	query['pos2'] = Variable(torch.from_numpy(np.stack(query['pos2'], 0)).long().view(-1, max_length)) 
	query['mask'] = Variable(torch.from_numpy(np.stack(query['mask'], 0)).long().view(-1, max_length))

	logits, pred = framework.predict(support, query, N, K, Q)
	ans.append((int)(pred.numpy()[0]))

json.dump(ans, sys.stdout)
file_name = 'pred-{}-{}.json'.format(N, K)
with open(file_name, 'w') as f:
	json.dump(ans, f)