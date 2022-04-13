import json
from vocabulary import Vocabulary
from pathlib import Path

class CluenerProcessor:
    """Processor for the chinese ner data set."""
    def __init__(self, data_dir):
        #建设词袋，创建文件路径
        self.vocab = Vocabulary()
        self.data_dir = Path(data_dir)
    #制作词袋 用于后面训练
    def get_vocab(self):
        #设置pkl文件保存路径
        vocab_path = self.data_dir / 'vocab.pkl'
        #如果已经存在pkl的话就直接使用本地的文件
        if vocab_path.exists():
            self.vocab.load_from_file(str(vocab_path))
        else:
        #列出需要提取的文件名然后列表循环
            files = ["train.json", "dev.json", "test.json"]
            for file in files:
                with open(str(self.data_dir / file), 'r', encoding='utf-8') as fr:
                    #循环每一行
                    for line in fr:
                        #除去两边空格
                        line = json.loads(line.strip())
                        #将content list化然后更新到update里面去
                        text = line['content']
                        #更新词袋
                        self.vocab.update(list(text))
            #更新完词袋，创建
            self.vocab.build_vocab()
            #保存词袋
            self.vocab.save(vocab_path)

    def get_train_examples(self):
        """See base class."""
        #创建train文件路径
        path = str(self.data_dir / "train.json")
        train_example = self._create_examples(path, "train")
        return train_example

    def get_dev_examples(self):
        """See base class."""
        # 创建dev文件路径
        return self._create_examples(str(self.data_dir / "dev.json"), "dev")

    def get_test_examples(self):
        """See base class."""
        #创建test 文件路径
        return self._create_examples(str(self.data_dir / "test.json"), "test")

    #处理文件格式，转换成模型能够消化的格式
    def _create_examples(self, input_path, mode):
        examples = []
        with open(input_path, 'r', encoding='utf-8') as fr:
            idx = 0
            for line in fr:
                json_d = {}
                #加载json去空格
                line_s = json.loads(line.strip())
                content = line_s['content']
                #将字list化
                words = list(content)
                #生成和word同样长度的list 用来存放labels,用乘法最快
                labels = ['O'] * len(words)
                #根据格式，label放在annotation里面，判断是否有annotation
                if line_s.get('annotation', None) != None:
                    #有直接用list循环把label值放入label_list
                    label_list = [x.get('label', None)[0] for x in line_s['annotation']]
                    #提取label在文档中的位置，这里是存在points里面所以直接提取
                    text_index = [[x.get('points', None)[0].get('start', None), x.get('points', None)[0].get('end', None)] for x in line_s['annotation']]
                    #这里主要判断label_list里面是否是空值
                    if (label_list is not None) & (len(text_index) == len(label_list)):
                        for i in range(len(label_list)):
                            label = label_list[i]
                            start = text_index[i][0]
                            end = text_index[i][1]
                            if start != end:
                                #实体开头第一个字要是B-开头
                                labels[start] = 'B-' + label
                                #print('B-' + label)
                                #剩下的以I-开头
                                labels[start + 1:end + 1] = ['I-' + label] * (end - start)
                            else:
                                labels[start] = 'B-' + label
                    #如果不满足上述条件，这里会产生报错
                    else:
                        assert print('index_length:', len(text_index), '  and label_list:', len(label_list))

        # with open(input_path, 'r', encoding='utf-8') as f:
        #     idx = 0
        #     for line in f:
        #         json_d = {}
        #         line = json.loads(line.strip())
        #         text = line['text']
        #         label_entities = line.get('label', None)
        #         words = list(text)
        #         labels = ['O'] * len(words)
        #         if label_entities is not None:
        #             for key, value in label_entities.items():
        #                 for sub_name, sub_index in value.items():
        #                     for start_index, end_index in sub_index:
        #                         assert ''.join(words[start_index:end_index + 1]) == sub_name
        #                         if start_index == end_index:
        #                             labels[start_index] = 'S-' + key
        #                         else:
        #                             labels[start_index] = 'B-' + key
        #                             labels[start_index + 1:end_index + 1] = ['I-' + key] * (len(sub_name) - 1)
                json_d['id'] = f"{mode}_{idx}"
                json_d['context'] = "|".join(words)
                json_d['tag'] = "|".join(labels)

                json_d['raw_context'] = content
                # if len(words) != len(labels) or len(labels) != len(json_d['context'].split('|')):
                #     print(json_d['id'],' ',len(words),' ',len(labels),len(json_d['context'].split('|')))
                # else:
                #     print('Check')
                idx += 1
                examples.append(json_d)
                # print(json_d)
        return examples

if __name__ == '__main__':
    data_dir = 'D:/Git_CCI/zjy_test_project/CLUENER2020/bilstm_crf_pytorch/dataset'
    file = 'train.json'
    a = CluenerProcessor(data_dir)
    train_example = a.get_train_examples()