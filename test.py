import models
from fewshot_re_kit.data_loader import JSONFileDataLoader
from fewshot_re_kit.framework import FewShotREFramework
from fewshot_re_kit.sentence_encoder import CNNSentenceEncoder, CNNEntityEncoder
from models.pan import PAN
import sys

model_name = 'pan'
N = 5
K = 5
noise_rate = 0
if len(sys.argv) > 1:
    model_name = sys.argv[1]
if len(sys.argv) > 2:
    N = int(sys.argv[2])
if len(sys.argv) > 3:
    K = int(sys.argv[3])
if len(sys.argv) > 4:
    noise_rate = float(sys.argv[4])

print("{}-way-{}-shot Few-Shot Relation Classification".format(N, K))
print("Model: {}".format(model_name))

max_length = 40
entity_length = 6
train_data_loader = JSONFileDataLoader('./data/train.json', './data/glove.6B.50d.json', max_length=max_length)
val_data_loader = JSONFileDataLoader('./data/val.json', './data/glove.6B.50d.json', max_length=max_length)
test_data_loader = JSONFileDataLoader('./data/test.json', './data/glove.6B.50d.json', max_length=max_length)

framework = FewShotREFramework(train_data_loader, val_data_loader, test_data_loader)
sentence_encoder = CNNSentenceEncoder(train_data_loader.word_vec_mat, max_length)
entity_encoder = CNNEntityEncoder(test_data_loader.word_vec_mat, entity_length)

if model_name == 'pan':
    model = PAN(sentence_encoder, entity_encoder).cuda()
    ckpt='checkpoint/pan.pth.tar'
else:
    raise NotImplementedError

acc = 0
for i in range(5):
    acc += framework.eval(model, 4, N, K, 100, 3000, ckpt=ckpt, noise_rate=noise_rate)
acc /= 5.0
print("ACC: {}".format(acc))

