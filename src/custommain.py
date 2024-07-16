import re
import random
import time
from statistics import mode
import pickle
import os
import datetime
from PIL import Image
import numpy as np
import pandas
import torch
import torch.nn as nn
from torchvision import transforms
from torch import Tensor
import wandb
from transformers import CLIPProcessor, CLIPModel, AutoProcessor, AutoModel
from torch.optim.lr_scheduler import CyclicLR
from torch.cuda.amp import autocast, GradScaler

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


#テキストの前処理
def process_text(text):
    # lowercase
    text = text.lower()

    # 数詞を数字に変換
    num_word_to_digit = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10'
    }
    for word, digit in num_word_to_digit.items():
        text = text.replace(word, digit)

    # 小数点以外のピリオドを削除
    text = re.sub(r'(?<!\d)\.(?!\d)', '', text)

    # 冠詞の削除
    text = re.sub(r'\b(a|an|the)\b', '', text)

    # 短縮形のカンマの追加
    contractions = {
        "dont": "don't", "isnt": "isn't", "arent": "aren't", "wont": "won't",
        "cant": "can't", "wouldnt": "wouldn't", "couldnt": "couldn't"
    }
    for contraction, correct in contractions.items():
        text = text.replace(contraction, correct)

    # 英数字、アンダースコア、空白、シングルクォート、コロン以外のすべての記号や句読点をスペースに変換
    # ^キャレット記号によって除外を表現．[]キャラクタークラスの中でキャレット記号が最初に置かれた場合そのクラスを含まない文字にマッチするという意味になる
    text = re.sub(r"[^\w\s':]", ' ', text)

    # カンマの前にある一つ以上の空白にマッチし，それをカンマだけに置き換える
    text = re.sub(r'\s+,', ',', text)

    # 連続するスペースを1つに変換
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# 1. データローダーの作成
class VQADataset(torch.utils.data.Dataset):
    def __init__(self, df_path, image_dir, transform=None, answer=True, dict_path='answer_dict.pkl'):
        self.transform = transform  # 画像の前処理
        self.image_dir = image_dir  # 画像ファイルのディレクトリ
        self.df = pandas.read_json(df_path)  # 画像ファイルのパス，question, answerを持つDataFrame
        self.answer = answer
        self.dict_path = dict_path


        if os.path.exists(self.dict_path):
            # 既存の辞書を読み込む
            with open(self.dict_path, 'rb') as f:
                self.answer2idx = pickle.load(f)
        else:
            # 新しい辞書を作成
            self.answer2idx = {}

        if self.answer:
            # DataFrameから辞書を更新
            self.update_dict_from_df()

            # CSVから辞書を更新
            self.update_dict_from_csv('class_mapping.csv')
            print(len(self.answer2idx))

            # 辞書を逆にして保存
            self.idx2answer = {v: k for k, v in self.answer2idx.items()}

            # 辞書をファイルに保存
            with open(self.dict_path, 'wb') as f:
                pickle.dump(self.answer2idx, f)

    def update_dict_from_df(self):
        for answers in self.df["answers"]:
            for answer in answers:
                word = process_text(answer["answer"])
                if word not in self.answer2idx:
                    self.answer2idx[word] = len(self.answer2idx)

    def update_dict_from_csv(self, csv_path):
        class_data = pandas.read_csv(csv_path)
        for _, row in class_data.iterrows():
            word = row['answer']
            if word not in self.answer2idx:
                self.answer2idx[word] = row['class_id']

    def update_dict(self, dataset):
        """
        検証用データ，テストデータの辞書を訓練データの辞書に更新する．

        Parameters
        ----------
        dataset : Dataset
            訓練データのDataset
        """
        self.answer2idx = dataset.answer2idx
        self.idx2answer = dataset.idx2answer

    def __getitem__(self, idx):
        """
        対応するidxのデータ（画像，質問，回答）を取得．

        Parameters
        ----------
        idx : int
            取得するデータのインデックス

        Returns
        -------
        image : torch.Tensor  (C, H, W)
            画像データ
        question : torch.Tensor  (vocab_size)
            質問文をone-hot表現に変換したもの
        answers : torch.Tensor  (n_answer)
            10人の回答者の回答のid
        mode_answer_idx : torch.Tensor  (1)
            10人の回答者の回答の中で最頻値の回答のid
        """
        image = Image.open(f"{self.image_dir}/{self.df['image'][idx]}")
        # questionそのままを渡す
        question = self.df['question'][idx]

        if self.answer:
            answers = [self.answer2idx[process_text(answer["answer"])] for answer in self.df["answers"][idx]]
            mode_answer_idx = mode(answers)  # 最頻値を取得（正解ラベル）
            return image, question, torch.Tensor(answers), int(mode_answer_idx)

        else:
            return image, question

    def __len__(self):
        return len(self.df)


# 2. 評価指標の実装
# 簡単にするならBCEを利用する
# 10人の回答の内9人の回答を選択し，その10パターンのAccの平均をそのデータに対するAccとする．
# iが除外する回答の選択，jがそれ以外の回答を回る．
def VQA_criterion(batch_pred: torch.Tensor, batch_answers: torch.Tensor):
    total_acc = 0.

    for pred, answers in zip(batch_pred, batch_answers):
        acc = 0.
        for i in range(len(answers)):
            num_match = 0
            for j in range(len(answers)):
                if i == j:
                    continue
                if pred == answers[j]:
                    num_match += 1
            acc += min(num_match / 3, 1)
        total_acc += acc / 10

    return total_acc / len(batch_pred)


class VQAModel(nn.Module):
    def __init__(self, n_answer: int, model):
        super().__init__()
        self.model = model
        self.pool = nn.AdaptiveAvgPool1d(1)  # 平均プーリングで次元を削減
        self.fc1 = nn.Linear(256, 256)  # 次元を削減した全結合層
        self.bn1 = nn.BatchNorm1d(256)  # Batch Normalization
        self.relu = nn.ReLU()  # ReLU Activation
        self.fc2 = nn.Linear(256, n_answer)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs):
        outputs = self.model(**inputs)
        features = outputs.last_hidden_state
        features = features.mean(dim=1)  # 平均プーリング
        output = self.fc1(features)
        output = self.bn1(output)  # Apply Batch Normalization
        output = self.relu(output)  # Apply ReLU Activation
        output = self.fc2(output)
        return self.log_softmax(output)

def calculate_soft_labels(answers, num_classes, device, unanswerable_idx):
    batch_size, num_answers = answers.shape

    answers = answers.long()

    soft_labels = torch.zeros(batch_size, num_classes).to(device)

    for i in range(batch_size):
        label_counts = torch.zeros(num_classes).to(device)
        
        # 各回答に対するカウント
        for j in range(num_answers):
            answer_idx = answers[i, j]
            if answer_idx != unanswerable_idx:
                label_counts[answer_idx] += 1  # unanswerableでない回答は2倍にカウント
            else:
                label_counts[answer_idx] += 1  # unanswerableの回答は通常通りカウント
        
        # ソフトラベルを計算（正規化）
        soft_labels[i] = label_counts / label_counts.sum()
    
    return soft_labels

def rescale(x):
    return x * 0.00392156862745098

# 4. 学習の実装
#def train(model, dataloader, optimizer, scheduler_cyclic,criterion, device, dataset, unanswerable_idx, processor):
def train(model, dataloader, optimizer,criterion, device, dataset, unanswerable_idx, processor):
    dataset = dataset
    model.train()
    scaler = GradScaler()  # Mixed Precision Trainingのためのスケーラー


    total_loss = 0
    total_acc = 0
    simple_acc = 0

    start = time.time()
    for batch_idx, (image, question, answers, mode_answer) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}/{len(dataloader)}")
        print_memory_usage()  # メモリ使用量をプリント
        inputs = processor(images=image, text=question, return_tensors="pt", padding=True, truncation=True).to(device)
        answers, mode_answer = answers.to(device), mode_answer.to(device)
        
        soft_labels = calculate_soft_labels(answers, num_classes=len(dataset.answer2idx), device=device, unanswerable_idx=unanswerable_idx)
        soft_labels = soft_labels.to(device)

        optimizer.zero_grad()
        with autocast():  # Mixed Precision Trainingのためのautocast
            pred = model(inputs)
            loss = criterion(pred, soft_labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        #scheduler_cyclic.step()  # CyclicLRの更新

        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)  # VQA accuracy
        simple_acc += (pred.argmax(1) == mode_answer).float().mean().item()  # simple accuracy

        # 不要なテンソルを削除し、メモリを解放
        del inputs, answers, mode_answer, soft_labels, pred, loss
        torch.cuda.empty_cache()

    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start

def print_memory_usage():
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # メガバイトに変換
    cached = torch.cuda.memory_reserved() / (1024 ** 2)  # メガバイトに変換
    print(f"Allocated Memory: {allocated:.2f} MB")
    print(f"Cached Memory: {cached:.2f} MB")

class CustomCollate:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def collate_fn(self, batch):
        images, questions, answers, mode_answers = zip(*batch)
        answers = torch.stack(answers, dim=0)
        mode_answers = torch.tensor(mode_answers, dtype=torch.long)
        return images, questions, answers, mode_answers

    def collate_fntest(self, batch):
        images, questions = zip(*batch)
        return images, questions

def main():
    wandb.init(
        project="vqa_project",
        config={
        "architecture": "Grounded DINO",
        "dataset": "VizWiz",
        "epochs": 10,
        "learning_rate": "Exponential",
        "batch_size":64
        }
    )
    # deviceの設定
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model_id = "IDEA-Research/grounding-dino-base"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device)

    #トークナイザー
    tokenizer = processor.tokenizer

    #すべてのパラメータを凍結
    for param in model.parameters():
        param.requires_grad = False

    # collate_fnの生成
    collate_obj = CustomCollate(tokenizer)

    # DINOに合わせた画像前処理にデータ拡張を追加
    transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # 画像をランダムに水平フリップ
    transforms.RandomVerticalFlip(),  # 画像をランダムに垂直フリップ
    transforms.RandomRotation(20),  # 20度の範囲でランダムに回転
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),  # 色調のランダムな調整
    transforms.Resize((800, 1333)),  # shortest_edge: 800, longest_edge: 1333
    transforms.ToTensor(),  # PIL ImageをTensorに変換
    transforms.Lambda(rescale),  # rescale_factor: 1/255
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 正規化
    ])

    # 検証およびテスト用のトランスフォーム
    val_transform = transforms.Compose([
        transforms.Resize((800, 1333)),  # shortest_edge: 800, longest_edge: 1333
        transforms.ToTensor(),  # PIL ImageをTensorに変換
        transforms.Lambda(rescale),  # rescale_factor: 1/255
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 正規化
    ])

    train_dataset = VQADataset(df_path="./data/train.json", image_dir="./data/train", transform=transform)
    test_dataset = VQADataset(df_path="./data/valid.json", image_dir="./data/valid", transform=val_transform, answer=False)
    test_dataset.update_dict(train_dataset)
    unanswerable_idx = train_dataset.answer2idx['unanswerable']

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn = collate_obj.collate_fn, num_workers=2, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn = collate_obj.collate_fntest, num_workers=2, pin_memory=True)

    model = VQAModel(n_answer=len(train_dataset.answer2idx), model = model).to(device)

    # optimizer / criterion
    num_epoch = 10
    criterion = nn.KLDivLoss(reduction='batchmean')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    # CyclicLRの設定
    #scheduler_cyclic = CyclicLR(optimizer, base_lr=0.0001, max_lr=0.001, step_size_up=50, mode='triangular')
    # ExponentialLRの設定
    scheduler_exp = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.75)

    # train model
    for epoch in range(num_epoch):
        # エポック開始時の学習率を確認
        current_lr = optimizer.param_groups[0]['lr']
        print(f"エポック {epoch+1} の開始時の学習率: {current_lr:.6f}")
        model.train()
        train_loss, train_acc, train_simple_acc, train_time = train(model, train_loader, optimizer, criterion, device, train_dataset, unanswerable_idx, processor)
        #train_loss, train_acc, train_simple_acc, train_time = train(model, train_loader, optimizer, scheduler_cyclic, criterion, device, train_dataset, unanswerable_idx, processor)
        # wandbによるログ記録
        wandb.log({"loss": train_loss, "accuracy": train_acc, "simple_accuracy": train_simple_acc})
        print(f"【{epoch + 1}/{num_epoch}】\n"
              f"train time: {train_time:.2f} [s]\n"
              f"train loss: {train_loss:.4f}\n"
              f"train acc: {train_acc:.4f}\n"
              f"train simple acc: {train_simple_acc:.4f}")
        scheduler_exp.step()  # ExponentialLRの更新

        # 提出用ファイルの作成
        if (epoch >= 5):
            model.eval()
            submission = []
            for image, question in test_loader:
                image, question = image.to(device), question.to(device)
                pred = model(question, image)
                pred = pred.argmax(1).cpu().item()
                submission.append(pred)

            # 現在の日時を取得し、フォルダ名を生成
            current_time = datetime.datetime.now()
            folder_name = current_time.strftime('%Y%m%d%H%M')
            folder_path = os.path.join('./', folder_name)
            # フォルダが存在しない場合は作成
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
    
            # 提出用ファイルの保存
            submission = [train_dataset.idx2answer[id] for id in submission]
            submission = np.array(submission)
            submission_path = os.path.join(folder_path, "submission_3.npy")
            np.save(submission_path, submission)


    model_path = os.path.join(folder_path, "model.pth")
    torch.save(model.state_dict(), model_path)
    wandb.finish()

if __name__ == "__main__":
    main()
