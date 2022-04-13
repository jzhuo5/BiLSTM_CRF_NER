from torch.nn import LayerNorm
import torch.nn as nn
from crf import CRF

class SpatialDropout(nn.Dropout2d):
    def __init__(self, p=0.6):
        super(SpatialDropout, self).__init__(p=p)

    def forward(self, x):
        x = x.unsqueeze(2)  # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x

class NERModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size,
                 label2id, device, drop_p=0.1):
        super(NERModel, self).__init__()
        self.emebdding_size = embedding_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)#embadding shape =[vocab_size,embedding_size]
        #
        self.bilstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size,
                              batch_first=True, num_layers=2, dropout=drop_p,
                              bidirectional=True)
        self.dropout = SpatialDropout(drop_p)
        #因为是双向，两个方向上的参数concat一起，所以要x2
        self.layer_norm = LayerNorm(hidden_size * 2)
        self.classifier = nn.Linear(hidden_size * 2, len(label2id))#(输入，输出)
        self.crf = CRF(tagset_size=len(label2id), tag_dictionary=label2id, device=device)

    def forward(self, inputs_ids, input_mask):
        #forward propagation
        #inputs_ids = [32,275]
        embs = self.embedding(inputs_ids)
        #embs = torch.Size(32, 275, 128])
        embs = self.dropout(embs)
        #在第二个位置上插入一个维度，让input_mask的维度和embs一样,
        #input_mask掩盖掉个别字
        #[32, 275, 128] *[32, 275, 1]
        embs = embs * input_mask.float().unsqueeze(2)
        #seqence_output.shape=[20, 54, 768],其中768=2*384,train:[[32, 275, 768]]
        seqence_output, _ = self.bilstm(embs)
        seqence_output = self.layer_norm(seqence_output)
        #features.shape=[20, 54, 37]= numpy.dot([20, 54, 768],[768,37]),train:[32, 275, 768]
        #classifier将768个特征，即每个字2*384个特征转换为37个,train:[32, 275, 37]
        features = self.classifier(seqence_output)
        return features

    def forward_loss(self, input_ids, input_mask, input_lens, input_tags=None):
        features = self.forward(input_ids, input_mask)
        if input_tags is not None:
            return features, self.crf.calculate_loss(features, tag_list=input_tags, lengths=input_lens)
        else:
            return features