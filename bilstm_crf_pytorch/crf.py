import torch
import torch.nn as nn
import torch.nn.functional as F

def to_scalar(var):
    return var.view(-1).detach().tolist()[0]

def argmax(vec):
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)

def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

def argmax_batch(vecs):
    _, idx = torch.max(vecs, 1)
    return idx

def log_sum_exp_batch(vecs):
    maxi = torch.max(vecs, 1)[0]
    maxi_bc = maxi[:, None].repeat(1, vecs.shape[1])
    recti_ = torch.log(torch.sum(torch.exp(vecs - maxi_bc), 1))
    return maxi + recti_

class CRF(nn.Module):
    def __init__(self, tagset_size, tag_dictionary, device, is_bert=None):#字典长度，字典，设备
        super(CRF, self).__init__()

        self.START_TAG = "<START>"
        self.STOP_TAG = "<STOP>"
        if is_bert:
            self.START_TAG = "[CLS]"
            self.STOP_TAG = "[SEP]"
        self.tag_dictionary = tag_dictionary
        self.tagset_size = tagset_size
        self.device = device
        self.transitions = torch.randn(tagset_size, tagset_size)#[37,37]
        # self.transitions = torch.zeros(tagset_size, tagset_size)
        #将start与stop两个参数从要学习的label中隔离开来
        self.transitions.detach()[self.tag_dictionary[self.START_TAG], :] = -10000
        self.transitions.detach()[:, self.tag_dictionary[self.STOP_TAG]] = -10000

        self.transitions = self.transitions.to(device)
        #将self.transitions里面的参数设置为可学习参数，除了start与end
        self.transitions = nn.Parameter(self.transitions)

    def _viterbi_decode(self, feats):
        backpointers = []
        backscores = []
        scores = []
        init_vvars = (torch.FloatTensor(1, self.tagset_size).to(self.device).fill_(-10000.0))
        init_vvars[0][self.tag_dictionary[self.START_TAG]] = 0
        forward_var = init_vvars

        for feat in feats:
            next_tag_var = (
                    forward_var.view(1, -1).expand(self.tagset_size, self.tagset_size)
                    + self.transitions
            )
            _, bptrs_t = torch.max(next_tag_var, dim=1)
            viterbivars_t = next_tag_var[range(len(bptrs_t)), bptrs_t]
            forward_var = viterbivars_t + feat
            backscores.append(forward_var)
            backpointers.append(bptrs_t)

        terminal_var = (
                forward_var
                + self.transitions[self.tag_dictionary[self.STOP_TAG]]
        )
        #这里保证tensor不会让把状态传给状态是start的标签,同时也不会出现stop之后再次transfer
        terminal_var.detach()[self.tag_dictionary[self.STOP_TAG]] = -10000.0
        terminal_var.detach()[self.tag_dictionary[self.START_TAG]] = -10000.0
        best_tag_id = argmax(terminal_var.unsqueeze(0))
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id.item())
        best_scores = []
        for backscore in backscores:
            softmax = F.softmax(backscore, dim=0)
            _, idx = torch.max(backscore, 0)
            prediction = idx.item()
            best_scores.append(softmax[prediction].item())
            scores.append([elem.item() for elem in softmax.flatten()])
        swap_best_path, swap_max_score = (
            best_path[0],
            scores[-1].index(max(scores[-1])),
        )
        scores[-1][swap_best_path], scores[-1][swap_max_score] = (
            scores[-1][swap_max_score],
            scores[-1][swap_best_path],
        )
        start = best_path.pop()
        assert start == self.tag_dictionary[self.START_TAG]
        best_path.reverse()
        return best_scores, best_path, scores
    #lens=32为bach长度,feats = [32, 275, 37] 275为batch里面最长的句子长度，37为id2label长度
    def _forward_alg(self, feats, lens_):#forward algorithm
        #设立初始化alpha,37个，设定初始概率分布，统一设定为-10000.0
        init_alphas = torch.FloatTensor(self.tagset_size).fill_(-10000.0)
        #把起始点标为0,因为起始点在第35（36）位所以第35为0.0
        init_alphas[self.tag_dictionary[self.START_TAG]] = 0.0
        #创建一个值为零的向量，将最大长度275 扩大到276，在开头加入初始概率分布
        forward_var = torch.zeros(
            feats.shape[0],#32
            feats.shape[1] + 1,#276
            feats.shape[2],#37
            dtype=torch.float,
            device=self.device,
        )
        #让forward var的第一排参数都变成第35位为0的emb list,设定初始状态分布Pi,这里是init_alpha
        forward_var[:, 0, :] = init_alphas[None, :].repeat(feats.shape[0], 1)
        #transitions.shape = [32, 37, 37]
        #transitions 马尔可夫转移概率,转移概率从前一刻时刻状态变成后一刻时刻状态,
        #transitions需要为每一个样本准备一个初始分布，所以要32个。一个batch 32样本。transitions = [32, 37, 37]
        transitions = self.transitions.view(
            1, self.transitions.shape[0], self.transitions.shape[1]
        ).repeat(feats.shape[0], 1, 1)
        #emit_score 处理张量为[32batch长度,37分类长度]
        for i in range(feats.shape[1]):#表示为最多是275个字中的第i个字 #进行动态规划递推时刻t=2,3,...T时刻的局部状态 这里是i
            #这里定义发射概率分布，一个batch中第i个字emit_score size=[32,37]
            emit_score = feats[:, i, :]
            #这边相加是什么意思？
            tag_var = ( #tag_var = 上一个状态+转移状态+现在观测状态
                emit_score[:, :, None].repeat(1, 1, transitions.shape[2])
                + transitions
                + forward_var[:, i, :][:, :, None]
                .repeat(1, 1, transitions.shape[2])
                .transpose(2, 1)
            )#选出可能性最大的状态
            max_tag_var, _ = torch.max(tag_var, dim=2)
            #target_value 减去模型计算出来的最大可能值，得到损失?????????????????
            tag_var = tag_var - max_tag_var[:, :, None].repeat(
                1, 1, transitions.shape[2]
            )
            #agg_=[32,37]计算 log-sum-exp
            agg_ = torch.log(torch.sum(torch.exp(tag_var), dim=2))
            cloned = forward_var.clone()
            #赋值到第276列 max_tag_var + agg_？？？？？？？？
            cloned[:, i + 1, :] = max_tag_var + agg_
            forward_var = cloned
        #?????????????????????????
        forward_var = forward_var[range(forward_var.shape[0]), lens_, :]
        #???????????????????
        terminal_var = forward_var + self.transitions[self.tag_dictionary[self.STOP_TAG]][None, :].repeat(forward_var.shape[0], 1)
        alpha = log_sum_exp_batch(terminal_var)
        return alpha

    # # Gives the score of a provided tag sequence
    def _score_sentence(self, feats, tags, lens_):
        start = torch.LongTensor([self.tag_dictionary[self.START_TAG]]).to(self.device)
        start = start[None, :].repeat(tags.shape[0], 1)
        stop = torch.LongTensor([self.tag_dictionary[self.STOP_TAG]]).to(self.device)
        stop = stop[None, :].repeat(tags.shape[0], 1)
        pad_start_tags = torch.cat([start, tags], 1)
        pad_stop_tags = torch.cat([tags, stop], 1)
        for i in range(len(lens_)):
            pad_stop_tags[i, lens_[i] :] = self.tag_dictionary[self.STOP_TAG]
        score = torch.FloatTensor(feats.shape[0]).to(self.device)
        for i in range(feats.shape[0]):
            r = torch.LongTensor(range(lens_[i])).to(self.device)
            score[i] = torch.sum(
                self.transitions[
                    pad_stop_tags[i, : lens_[i] + 1], pad_start_tags[i, : lens_[i] + 1]
                ]
            ) + torch.sum(feats[i, r, tags[i, : lens_[i]]])
        return score

    def _obtain_labels(self, feature, id2label, input_lens):
        tags = []
        all_tags = []
        for feats, length in zip(feature, input_lens):
            confidences, tag_seq, scores = self._viterbi_decode(feats[:length])
            tags.append([id2label[tag] for tag in tag_seq])
            all_tags.append([[id2label[score_id] for score_id, score in enumerate(score_dist)] for score_dist in scores])
        return tags, all_tags

    def calculate_loss(self, scores, tag_list, lengths):
        return self._calculate_loss_old(scores, lengths, tag_list)

    def _calculate_loss_old(self, features, lengths, tags):
        forward_score = self._forward_alg(features, lengths)
        gold_score = self._score_sentence(features, tags, lengths)
        score = forward_score - gold_score
        return score.mean()


