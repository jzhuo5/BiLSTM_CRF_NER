import json
import torch
import argparse
import torch.nn as nn
from torch import optim
import config
from model import NERModel
from dataset_loader import DatasetLoader
from progressbar import ProgressBar
from ner_metrics import SeqEntityScore
from data_processor import CluenerProcessor
from lr_scheduler import ReduceLROnPlateau
from utils_ner import get_entities
from common import (init_logger,
                    logger,
                    json_to_text,
                    load_model,
                    AverageMeter,
                    seed_everything)

def train(args,model,processor):
    train_dataset = load_and_cache_examples(args, processor, data_type='train')
    #将样本向量化
    train_loader = DatasetLoader(data=train_dataset, batch_size=args.batch_size,
                                 shuffle=False, seed=args.seed, sort=True,
                                 vocab=processor.vocab, label2id=args.label2id)
    #通过parameters() 函数返回迭代器用requires_grad判断是否为可学习参数，将可学习参数放入parameters 列表中
    parameters = [p for p in model.parameters() if p.requires_grad]
    #选择优化器
    optimizer = optim.Adam(parameters, lr=args.learning_rate)
    #设定定期学习率衰减
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3,
                                  verbose=1, epsilon=1e-4, cooldown=0, min_lr=0, eps=1e-8)
    #设定f1衰减
    best_f1 = 0
    for epoch in range(1, 1 + args.epochs):
        print(f"Epoch {epoch}/{args.epochs}")
        #加入进度条
        pbar = ProgressBar(n_total=len(train_loader), desc='Training')
        #?????????????????????????????
        train_loss = AverageMeter()
        model.train()
        assert model.training
        for step, batch in enumerate(train_loader):
            input_ids, input_mask, input_tags, input_lens = batch
            input_ids = input_ids.to(args.device)
            input_mask = input_mask.to(args.device)
            input_tags = input_tags.to(args.device)
            # try:
            #     print(input_ids.shape)
            # except:
            #     print(len(input_ids))
            # try:
            #     print(input_mask.shape)
            # except:
            #     print(len(input_mask))
            # try:
            #     print(input_tags.shape,' ',step)
            # except:
            #     print(len(input_tags))
            # try:
            #     print(input_lens.shape,' ',step)
            # except:
            #     print(len(input_lens))
            #前馈，并计算loss
            features, loss = model.forward_loss(input_ids, input_mask, input_lens, input_tags)
            #This function accumulates gradients in the leaves,计算叶子节点的梯度值
            loss.backward()
            #设置一个梯度剪切的阈值，如果在更新梯度的时候，梯度超过这个阈值，则会将其限制在这个范围之内，防止梯度爆炸。
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
            #执行一次优化步骤
            optimizer.step()
            #梯度清零
            optimizer.zero_grad()
            pbar(step=step, info={'loss': loss.item()})
            #更新（添加新的）梯度
            train_loss.update(loss.item(), n=1)
        print(" ")
        #计算平均梯度
        train_log = {'loss': train_loss.avg}
        if 'cuda' in str(args.device):
            torch.cuda.empty_cache()
        #测试查看效果。
        eval_log, class_info = evaluate(args, model, processor)
        logs = dict(train_log, **eval_log)
        show_info = f'\nEpoch: {epoch} - ' + "-".join([f' {key}: {value:.4f} ' for key, value in logs.items()])
        logger.info(show_info)
        scheduler.epoch_step(logs['eval_f1'], epoch)
        if logs['eval_f1'] > best_f1:
            logger.info(f"\nEpoch {epoch}: eval_f1 improved from {best_f1} to {logs['eval_f1']}")
            logger.info("save model to disk.")
            best_f1 = logs['eval_f1']
            if isinstance(model, nn.DataParallel):
                model_stat_dict = model.module.state_dict()
            else:
                model_stat_dict = model.state_dict()
            state = {'epoch': epoch, 'arch': args.arch, 'state_dict': model_stat_dict}
            model_path = args.output_dir / 'best-model.bin'
            torch.save(state, str(model_path))
            print("Eval Entity Score: ")
            for key, value in class_info.items():
                info = f"Subject: {key} - Acc: {value['acc']} - Recall: {value['recall']} - F1: {value['f1']}"
                logger.info(info)

def evaluate(args,model,processor):
    eval_dataset = load_and_cache_examples(args, processor, data_type='dev')
    eval_dataloader = DatasetLoader(data=eval_dataset, batch_size=args.batch_size,
                                 shuffle=False, seed=args.seed, sort=False,
                                 vocab=processor.vocab, label2id=args.label2id)
    pbar = ProgressBar(n_total=len(eval_dataloader), desc="Evaluating")

    metric = SeqEntityScore(args.id2label, markup=args.markup)
    eval_loss = AverageMeter()
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(eval_dataloader):
            input_ids, input_mask, input_tags, input_lens = batch
            input_ids = input_ids.to(args.device)
            input_mask = input_mask.to(args.device)
            input_tags = input_tags.to(args.device)

            features, loss = model.forward_loss(input_ids, input_mask, input_lens, input_tags)
            #update
            eval_loss.update(val=loss.item(), n=input_ids.size(0))
            tags, _ = model.crf._obtain_labels(features, args.id2label, input_lens)
            input_tags = input_tags.cpu().numpy()
            target = [input_[:len_] for input_, len_ in zip(input_tags, input_lens)]
            metric.update(pred_paths=tags, label_paths=target)
            pbar(step=step)
    print(" ")
    eval_info, class_info = metric.result()
    eval_info = {f'eval_{key}': value for key, value in eval_info.items()}
    result = {'eval_loss': eval_loss.avg}
    result = dict(result, **eval_info)
    return result, class_info

def predict(args,model,processor):
    model_path = args.output_dir / 'best-model.bin'
    model = load_model(model, model_path=str(model_path))
    test_data = []
    with open(str(args.data_dir / "test.json"), 'r', encoding='utf-8') as f:
        idx = 0
        for line in f:
            json_d = {}
            line = json.loads(line.strip())
            text = line['content']
            words = list(text)
            labels = ['O'] * len(words)
            json_d['id'] = idx
            json_d['context'] = "|".join(words)
            json_d['tag'] = "|".join(labels)
            json_d['raw_context'] = text
            idx += 1
            test_data.append(json_d)
    pbar = ProgressBar(n_total=len(test_data))
    results = []
    for step, line in enumerate(test_data):
        token_a = line['context'].split("|")
        input_ids = [processor.vocab.to_index(w) for w in token_a]
        input_mask = [1] * len(token_a)
        input_lens = [len(token_a)]
        model.eval()
        with torch.no_grad():
            input_ids = torch.tensor([input_ids], dtype=torch.long)
            input_mask = torch.tensor([input_mask], dtype=torch.long)
            input_lens = torch.tensor([input_lens], dtype=torch.long)
            input_ids = input_ids.to(args.device)
            input_mask = input_mask.to(args.device)
            features = model.forward_loss(input_ids, input_mask, input_lens, input_tags=None)
            tags, _ = model.crf._obtain_labels(features, args.id2label, input_lens)
        label_entities = get_entities(tags[0], args.id2label)
        json_d = {}
        json_d['id'] = step
        json_d['tag_seq'] = "|".join(tags[0])
        json_d['entities'] = label_entities
        results.append(json_d)
        pbar(step=step)
    print(" ")
    output_predic_file = str(args.output_dir / "test_prediction.txt")
    output_submit_file = str(args.output_dir / "test_submit.json")
    with open(output_predic_file, "w", encoding='utf-8') as writer:
        for record in results:
            writer.write(json.dumps(record) + '\n')
    test_text = []
    with open(str(args.data_dir / 'test.json'), 'r', encoding='utf-8') as fr:
        for line in fr:
            test_text.append(json.loads(line))
    test_submit = []
    for x, y in zip(test_text, results):
        json_d = {}
        json_d['id'] = y['id']
        json_d['label'] = {}
        entities = y['entities']
        words = list(x['content'])
        if len(entities) != 0:
            for subject in entities:
                tag = subject[0]
                start = subject[1]
                end = subject[2]
                word = "".join(words[start:end + 1])
                if tag in json_d['label']:
                    if word in json_d['label'][tag]:
                        json_d['label'][tag][word].append([start, end])
                    else:
                        json_d['label'][tag][word] = [[start, end]]
                else:
                    json_d['label'][tag] = {}
                    json_d['label'][tag][word] = [[start, end]]
        test_submit.append(json_d)
    json_to_text(output_submit_file, test_submit)

def load_and_cache_examples(args,processor, data_type='train'):
    # Load data features from cache or dataset file
    cached_examples_file = args.data_dir / 'cached_crf-{}_{}_{}'.format(
        data_type,
        args.arch,
        str(args.task_name))
    if cached_examples_file.exists():
        logger.info("Loading features from cached file %s", cached_examples_file)
        examples = torch.load(cached_examples_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if data_type == 'train':
            examples = processor.get_train_examples()
        elif data_type == 'dev':
            examples = processor.get_dev_examples()
        logger.info("Saving features into cached file %s", cached_examples_file)
        torch.save(examples, str(cached_examples_file))
    return examples

def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--do_train", default=False, action='store_true')
    parser.add_argument('--do_eval', default=False, action='store_true')
    parser.add_argument("--do_predict", default=False, action='store_true')

    parser.add_argument('--markup', default='bios', type=str, choices=['bios', 'bio'])
    parser.add_argument("--arch", default='bilstm_crf', type=str)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--embedding_size', default=128, type=int)
    parser.add_argument('--hidden_size', default=384, type=int)
    parser.add_argument("--grad_norm", default=5.0, type=float, help="Max gradient norm.")
    parser.add_argument("--task_name", type=str, default='ner')
    args = parser.parse_args()
    args.data_dir = config.data_dir
    #判断预设的路径是否存在，不存在则创建
    if not config.output_dir.exists():
        args.output_dir.mkdir()
    args.output_dir = config.output_dir / '{}'.format(args.arch)
    if not args.output_dir.exists():
        args.output_dir.mkdir()
    init_logger(log_file=str(args.output_dir / '{}-{}.log'.format(args.arch, args.task_name)))
    #设置整个开发环境的seed
    seed_everything(args.seed)
    #启用cuda
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # if args.gpu!='':
    #     args.device = torch.device(f"cuda:{args.gpu}")
    # else:
    #     args.device = torch.device("cpu")
    #clabel2id 本来就是字典,这边调换一下键与值的顺序
    args.id2label = {i: label for i, label in enumerate(config.label2id)}
    args.label2id = config.label2id
    #将数据导入数据处理中,详情见data_processor
    processor = CluenerProcessor(data_dir=config.data_dir)
    #制作词袋,详情见data_processor
    processor.get_vocab()
    # print(processor)
    #输入模型超超参数
    model = NERModel(vocab_size=len(processor.vocab), embedding_size=args.embedding_size,
                     hidden_size=args.hidden_size, device=args.device, label2id=args.label2id)
    model.to(args.device)
    print('this is main')
    args.do_train = True
    if args.do_train:
        print("do_train")
        train(args, model, processor)
    args.do_eval = True
    if args.do_eval:
        print("do_eval")
        model_path = args.output_dir / 'best-model.bin'
        model = load_model(model, model_path=str(model_path))
        result, class_info = evaluate(args, model, processor)
        print('result: ', result)
        print('class_info: ', class_info)
    args.do_predict = False
    if args.do_predict:
        print("do_predict")
        predict(args, model, processor)

if __name__ == "__main__":
    main()
    print('finish')

