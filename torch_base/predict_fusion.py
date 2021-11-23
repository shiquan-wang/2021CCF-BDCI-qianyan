from genericpath import exists
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 程序可调用的GPU的ID

import random
import torch
import numpy as np
from tqdm import tqdm
import time
import logging
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from sklearn.metrics import accuracy_score, f1_score
from transformers import BertModel, BertConfig
from transformers import AdamW
from transformers import BertTokenizer
from transformers.optimization import get_linear_schedule_with_warmup
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from torch.cuda.amp import autocast, GradScaler
from torch.optim.optimizer import Optimizer
import math
import Levenshtein
from pypinyin import lazy_pinyin
import jieba



# 设置参数及文件路径

max_seq_length = 60  # 输入文本最大长度
learning_rate = 2e-5  # 模型学习率
num_epochs = 10  # 训练最大迭代次数
batch_size = 32  # 训练时每个batch中的样本数
patience = 5  # 早停轮数
file_name = 'model_fusion_erine_ernie'#'baseline_testA'  # 指定输出文件的名字
#model_name_or_path = '/Work20/2020/wangshiquan/model/chinese-roberta-wwm-ext/'  # 预训练模型权重载入路径
model_name_or_path = '/Work20/2020/wangshiquan/model/ernie_grame/'  # 预训练模型权重载入路径
robert_path = '/Work20/2020/wangshiquan/model/ernie_grame/'  # 预训练模型权重载入路径
#robert_path = '/Work20/2020/wangshiquan/model/chinese-roberta-wwm-ext/' #robert 路径
train_input = '/Work20/2020/wangshiquan/competiion/21bdci/data/train.txt'  # 完成预处理的训练集载入路径
#test_input = '../data/dev.txt'  # 完成预处理的测试集集载入路径
test_input = '/Work20/2020/wangshiquan/competiion/21bdci/data/test_A.tsv'  # 完成预处理的测试集集载入路径
random_seed = 42  # 随机种子
submit_path = './submit/test_prob/'
output_path = './output/'
trained_model_path = '/home/2020/wangshiquan/work20/competiion/21bdci/torch_base/model_save/ernie_baseline_all_databest_17.bin' #ernie best acc 86.86
trained_robert_path = '/home/2020/wangshiquan/work20/competiion/21bdci/torch_base/model_save/erine_bs32_data_enhance/erine_bs32_data_enhance_best_1.bin'


def seed_everything(seed=random_seed):
    '''
    固定随机种子
    :param random_seed: 随机种子数目
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_everything()

# 创建一个logger
file_path = './log/'
if not os.path.exists(file_path):
    os.mkdir(file_path)
logger = logging.getLogger('mylogger')
logger.setLevel(logging.DEBUG)
timestamp = time.strftime("%Y.%m.%d_%H.%M.%S", time.localtime())
fh = logging.FileHandler(file_path + file_name + '_predict.txt')
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s][%(levelname)s] ## %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)


logger.info(f"model path : {trained_model_path} ,{trained_robert_path}")


class InputExample(object):
    def __init__(self, s1, s2, label=None):
        self.s1 = s1
        self.s2 = s2
        self.label = label


class InputFeatures(object):
    def __init__(self,
                 choices_features,
                 label

                 ):
        _, input_ids, input_mask, segment_ids = choices_features[0]
        self.choices_features = {
            'input_ids': input_ids,
            'input_mask': input_mask,
            'segment_ids': segment_ids
        }
        self.label = label


def read_data(file_name):
    examples = []
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            if line:
                line = line.split('\t')
                examples.append(InputExample(s1=line[0], s2=line[1], label=int(line[2]) if len(line) == 3 else None))
    return examples


def read_examples(dir, split='train'):
    examples = []
    for path in os.listdir(dir):
        if split == 'train':
            for file_name in os.listdir(dir + path):
                example = read_data(os.path.join(dir+path, file_name))
                examples.extend(example)
        else:
            example = read_data(os.path.join(dir, path))
            examples.extend(example)
    return examples


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.

    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 is_training):
    # 将文本输入样例，转换为数字特征，用于模型计算
    features = []
    for example_index, example in enumerate(examples):

        s1 = tokenizer.tokenize(example.s1)
        s2 = tokenizer.tokenize(example.s2)
        _truncate_seq_pair(s1, s2, max_seq_length)

        choices_features = []

        tokens = ["[CLS]"] + s1 + ["[SEP]"] + s2 + ["[SEP]"]
        segment_ids = [0] * (len(s1) + 2) + [1] * (len(s2) + 1)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        padding_length = max_seq_length - len(input_ids) + 3
        input_ids += ([0] * padding_length)
        input_mask += ([0] * padding_length)
        segment_ids += ([0] * padding_length)
        choices_features.append((tokens, input_ids, input_mask, segment_ids))

        label = example.label
        if example_index < 1 and is_training:
            logger.info("*** Example ***")
            logger.info("idx: {}".format(example_index))
            logger.info("tokens: {}".format(' '.join(tokens).replace('\u2581', '_')))
            logger.info("input_ids: {}".format(' '.join(map(str, input_ids))))
            logger.info("input_mask: {}".format(len(input_mask)))
            logger.info("segment_ids: {}".format(len(segment_ids)))
            logger.info("label: {}".format(label))

        features.append(
            InputFeatures(
                choices_features=choices_features,
                label=label
            )
        )
    return features


def select_field(features, field):
    return [
        feature.choices_features[field] for feature in features
    ]


class NeuralNet(nn.Module):
    def __init__(self, model_name_or_path, hidden_size=768, num_class=2):
        super(NeuralNet, self).__init__()

        self.config = BertConfig.from_pretrained(model_name_or_path, num_labels=num_class)
        self.config.output_hidden_states = True
        self.bert = BertModel.from_pretrained(model_name_or_path, config=self.config, )
        for param in self.bert.parameters():
            param.requires_grad = True
        self.weights = nn.Parameter(torch.rand(13, 1))
        
        # self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size*2, num_class)
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.2) for _ in range(5)#dropout = 0.2
        ])

    def forward(self, input_ids, input_mask, segment_ids, y=None, loss_fn=None):
        output = self.bert(input_ids, token_type_ids=segment_ids,
                                                                attention_mask=input_mask)
        # last_hidden = output.last_hidden_state
        last_hidden = output[0]
        # all_hidden_states = output.hidden_states
        all_hidden_states = output[2]
        batch_size = input_ids.shape[0]
        ht_cls = torch.cat(all_hidden_states)[:, :1, :].view(   #13 * batch * 1 * 768
            13, batch_size, 1, 768)
        atten = torch.sum(ht_cls * self.weights.view(
            13, 1, 1, 1), dim=[1, 3])
        atten = F.softmax(atten.view(-1), dim=0)
        feature = torch.sum(ht_cls * atten.view(13, 1, 1, 1), dim=[0, 2])
        f = torch.mean(last_hidden, 1)
        feature = torch.cat((feature, f), 1)
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                h = self.fc(dropout(feature))
                if loss_fn is not None:
                    loss = loss_fn(h, y)
            else:
                hi = self.fc(dropout(feature))
                h = h + hi
                if loss_fn is not None:
                    loss = loss + loss_fn(hi, y)
        if loss_fn is not None:
            return h / len(self.dropouts), loss / len(self.dropouts)
        return h / len(self.dropouts)

class MutiModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward():
        return 

class RAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.buffer = [[None, None, None] for ind in range(10)]
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                buffered = self.buffer[int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = math.sqrt(
                            (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (
                            N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size * group['lr'], exp_avg, denom)
                else:
                    p_data_fp32.add_(-step_size * group['lr'], exp_avg)

                p.data.copy_(p_data_fp32)

        return loss

def metric(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    return acc, f1

def set_lr(optimizer, value):
    for p in optimizer.param_groups:
        p['lr'] = value

class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def prob_postprocess(y_pred):
    #prior = np.array([0.6903327690476333, 0.3096672309523667]) # 训练集 oppo正负样本比例
    prior = np.array([0.52495854, 0.47504146]) # 所有训练集的正负样本比例

    # print("y_pred", y_pred)

    y_pred_uncertainty = -(y_pred * np.log(y_pred)).sum(1) / np.log(2)

    # print("y_pred_uncertainty", y_pred_uncertainty)
    
    threshold = 0.90
    y_pred_confident = y_pred[y_pred_uncertainty < threshold]
    y_pred_unconfident = y_pred[y_pred_uncertainty >= threshold]

    # print("y_pred_confident : ",y_pred_confident)
    # print("y_pred_unconfident", y_pred_unconfident)

    right, alpha, iters = 0, 1, 1
    post = []
    for i, y in enumerate(y_pred_unconfident):
        Y = np.concatenate([y_pred_confident, y[None]], axis=0)
        # print("Y : ", Y)
        for j in range(iters):
            Y = Y ** alpha
            Y /= Y.sum(axis=0, keepdims=True)
            Y *= prior[None]
            Y /= Y.sum(axis=1, keepdims=True)
        y = Y[-1]
        # print(f"y : {y}")
        post.append(y.tolist())
        # print("post :", post)

    

    post = np.array(post)

    # print("post : ", post)
    y_pred[y_pred_uncertainty >= threshold] = post

    return y_pred

# 加载数据
tokenizer = BertTokenizer.from_pretrained(model_name_or_path, do_lower_case=True)


#test_examples = read_examples(test_input, split='test')
test_examples = read_data(test_input)
test_features = convert_examples_to_features(
    test_examples, tokenizer, max_seq_length, True)
test_input_ids = torch.tensor(select_field(test_features, 'input_ids'), dtype=torch.long)
test_input_mask = torch.tensor(select_field(test_features, 'input_mask'), dtype=torch.long)
test_segment_ids = torch.tensor(select_field(test_features, 'segment_ids'), dtype=torch.long)

oof_test = np.zeros((len(test_examples), 2), dtype=np.float32)

logger.info('================     predict        ===============')

test = torch.utils.data.TensorDataset(test_input_ids, test_input_mask, test_segment_ids)

test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

model = NeuralNet(model_name_or_path)
model_bert = NeuralNet(robert_path)
# ema = EMA(model, 0.999)
# ema.register()
# for epoch in range(num_epochs):
#     train_loss = 0.
#     lr_list = []
#     # if epoch > 2:
#     #     set_lr(optimizer, 2e-5)
#     model.train()
#     for batch in tqdm(train_loader):
#         optimizer.zero_grad()
#         batch = tuple(t.cuda() for t in batch)
#         x_ids, x_mask, x_sids, y_truth = batch
#         with autocast():
#             y_pred, loss = model(x_ids, x_mask, x_sids, y=y_truth, loss_fn=loss_fn)
#         scaler.scale(loss.mean()).backward()
#         scaler.step(optimizer)
#         scale = scaler.get_scale()
#         scaler.update()
#         ema.update()
#         # skip_lr_sched = (scale != scaler.get_scale())
#         # if not skip_lr_sched:
#         #     scheduler.step()
#         train_loss += loss.mean().item() / len(train_loader)

#     ema.apply_shadow()
model.eval()
model_bert.eval()
val_loss = 0.

# 得到模型对测试集的预测结果
#model.load_state_dict(torch.load('./model_save/ernie_baseline_all_databest_17.bin'))
model.load_state_dict(torch.load(trained_model_path))
model_bert.load_state_dict(torch.load(trained_robert_path))

model.cuda()
model_bert.cuda()

test_preds_fold = np.zeros((len(test_examples), 2))
#test_preds_fold_robert = np.zeros((len(test_examples), 2))
model.eval()
model_bert.eval()

with torch.no_grad():
    for i, batch in tqdm(enumerate(test_loader)):
        batch = tuple(t.cuda() for t in batch)
        x_ids, x_mask, x_sids = batch
        with autocast():
            y_pred = model(x_ids, x_mask, x_sids).detach()
            y_pred_robert = model_bert(x_ids, x_mask, x_sids).detach()
        test_preds_fold[i * batch_size:(i + 1) * batch_size] = F.softmax((y_pred + y_pred_robert) / 2, dim=1).cpu().numpy()


oof_test += test_preds_fold #


# 保存概率文件
# np.savetxt('./submit/train_prob/train_bert_' + file_name + '.txt', oof_train)
if not os.path.exists(submit_path + file_name + '/'):
    os.mkdir(submit_path + file_name + '/')
np.savetxt(submit_path + file_name + '/submit.txt', oof_test)
# acc, f1 = metric(all_label, np.argmax(oof_train, axis=1))
# logger.info('epoch: best, acc: %.8f, f1: %.8f \n' % (acc, f1))

# analysis = pd.DataFrame()
# analysis['s1'] = [line.s1 for line in train_examples]
# analysis['s2'] = [line.s2 for line in train_examples]
# analysis['label'] = [line.label for line in train_examples]
# analysis['pred'] = np.argmax(oof_train, axis=1).tolist()
# analysis[analysis['label'] != analysis['pred']].to_csv('analysis_{}.csv'.format(f1), index=False)


# 后处理
#print(oof_test[0 : 5])

oof_test = prob_postprocess(oof_test)
y_preds = np.argmax(oof_test, axis=1)
logger.info(Counter(y_preds))
if not os.path.exists(output_path + file_name + '/'):
    os.mkdir(output_path + file_name + '/')
with open(output_path + file_name +'/predict.csv', 'w', encoding="utf-8") as f:
    for y_pred in y_preds:
        f.write(str(y_pred) + "\n")


def compare_pinyin(s1, s2):
    s1_pinyin = ""
    s2_pinyin = ""
    for w in jieba.cut(s1):
        s1_pinyin += ''.join(lazy_pinyin(w))
    for w in jieba.cut(s2):
        s2_pinyin += ''.join(lazy_pinyin(w))
    return s1_pinyin == s2_pinyin


def postprocess(data, pred):
    post = []
    for line, lable in tqdm(zip(data, pred)):
        # r1 = correct(line.s1, line.s2)  # 339
        r2 = compare_pinyin(line.s1, line.s2)  # 339
        if r2:
            post.append(1)
        else:
            post.append(lable)
    post = np.array(post)
    print(np.count_nonzero(post != pred))
    return post

post = postprocess(test_examples, y_preds)

if not os.path.exists(output_path + file_name + '/'):
    os.mkdir(output_path + file_name + '/')

with open(output_path + file_name +'/post_predict.csv', 'w', encoding="utf-8") as f:
    for y_pred in post:
        f.write(str(y_pred) + "\n")