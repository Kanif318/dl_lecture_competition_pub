import re
import random
import time
from statistics import mode
import pickle
import os

from PIL import Image
import numpy as np
import pandas
import torch
import torch.nn as nn
from torchvision import transforms
from transformers import AutoTokenizer, AutoModel
from torch import Tensor
import torch.nn.functional as F
import csv
import wandb
from transformers import CLIPProcessor, CLIPModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
# テキストエンコーダーを取得
text_encoder = model.text_model.to(device)
# 画像エンコーダーを取得
vision_encoder = model.vision_model.to(device)
#トークナイザー
tokenizer = processor.tokenizer

#すべてのパラメータを凍結
for param in text_encoder.parameters():
    param.requires_grad = False
for param in vision_encoder.parameters():
    param.requires_grad = False



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

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

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
        image = self.transform(image)
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
    def __init__(self, n_answer: int):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        # Normalize入れる
        self.fc = nn.Sequential(
            nn.Linear(1280, 1024),  # 最初の層のユニット数を増やす
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, n_answer)
        )
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, image, question):

        image_feature = self.vision_encoder(image).pooler_output  # 画像の特徴量
        question_feature = self.text_encoder(**question).pooler_output # テキストの特徴量
        # print("image_feature", image_feature.shape)
        # print("question_feature", question_feature.shape)

        x = torch.cat([image_feature, question_feature], dim=1)
        x = self.fc(x)

        return self.log_softmax(x)

def calculate_soft_labels(answers, num_classes):
    batch_size, num_answers = answers.shape

    answers = answers.long()

    soft_labels = torch.zeros(batch_size, num_classes).to(device)
    #answersはバッチで入ってきているので，バッチサイズ＊回答数
    #answerは10個（回答数）のインデックス
    # バッチ内の各データポイントに対して処理
    for i in range(batch_size):
        label_counts = torch.zeros(num_classes).to(device)
        
        # 各回答に対するカウント
        for j in range(num_answers):
            label_counts[answers[i, j]] += 1
        
        # ソフトラベルを計算（正規化）
        soft_labels[i] = label_counts / label_counts.sum()
    
    return soft_labels

# 4. 学習の実装
def train(model, dataloader, optimizer, criterion, device, dataset):
    dataset = dataset
    model.train()

    total_loss = 0
    total_acc = 0
    simple_acc = 0

    start = time.time()
    for image, question, answers, mode_answer in dataloader:
        # print("image", image)
        # print("question", question)
        image, question, answers, mode_answer = \
            image.to(device), question.to(device), answers.to(device), mode_answer.to(device)
        
        soft_labels = calculate_soft_labels(answers, num_classes=len(dataset.answer2idx))
        soft_labels = soft_labels.to(device)
        pred = model(image, question)
        loss = criterion(pred, soft_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)  # VQA accuracy
        simple_acc += (pred.argmax(1) == mode_answer).float().mean().item()  # simple accuracy

    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start


def eval(model, dataloader, optimizer, criterion, device):
    model.eval()

    total_loss = 0
    total_acc = 0
    simple_acc = 0

    start = time.time()
    for image, question, answers, mode_answer in dataloader:
        image, question, answer, mode_answer = \
            image.to(device), question.to(device), answers.to(device), mode_answer.to(device)

        pred = model(image, question)
        loss = criterion(pred, mode_answer.squeeze())

        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)  # VQA accuracy
        simple_acc += (pred.argmax(1) == mode_answer).mean().item()  # simple accuracy

    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start

def custom_collate_fn(batch):
    images, questions, answers, mode_answers = zip(*batch)
    # 画像はそのままスタック
    images = torch.stack(images, dim=0)
    # 質問はパディング
    questions = tokenizer(questions, padding=True, return_tensors='pt')
    # 回答も同様にパディング
    answers = torch.stack(answers, dim=0)
    mode_answers = torch.tensor(mode_answers, dtype = torch.long)
    return images, questions, answers, mode_answers

def custom_collate_fntest(batch):
    images, questions = zip(*batch)
    # 画像はそのままスタック
    images = torch.stack(images, dim=0)
    # 質問はパディング
    questions = tokenizer(questions, padding=True, return_tensors='pt')
    return images, questions

def main():
    wandb.init(
        project="vqa_project",
        config={
        "learning_rate": 0.001,
        "architecture": "CLIP+FC",
        "dataset": "VizWiz",
        "epochs": 10,
        "fine-tuning": "no"
        }
    )

    # deviceの設定
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # CLIPに合わせた画像前処理
    transform = transforms.Compose([
        transforms.Resize(224),  # 最短辺を224にリサイズ
        transforms.CenterCrop(224),  # 中央を224x224でクロップ
        transforms.ToTensor(),  # PIL Imageまたはnumpy.ndarrayをTensorに変換
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],  # CLIPの平均
                            std=[0.26862954, 0.26130258, 0.27577711])  # CLIPの標準偏差
    ])

    # 検証およびテスト用のトランスフォーム
    val_transform = transforms.Compose([
        transforms.Resize(224),  # 最短辺を224にリサイズ
        transforms.CenterCrop(224),  # 中央を224x224でクロップ
        transforms.ToTensor(),  # PIL Imageまたはnumpy.ndarrayをTensorに変換
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],  # CLIPの平均
                            std=[0.26862954, 0.26130258, 0.27577711])  # CLIPの標準偏差
    ])

    train_dataset = VQADataset(df_path="./data/train.json", image_dir="./data/train", transform=transform)
    test_dataset = VQADataset(df_path="./data/valid.json", image_dir="./data/valid", transform=val_transform, answer=False)
    test_dataset.update_dict(train_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn = custom_collate_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn = custom_collate_fntest)

    model = VQAModel(n_answer=len(train_dataset.answer2idx)).to(device)

    # optimizer / criterion
    num_epoch = 10
    criterion = nn.KLDivLoss(reduction='batchmean')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # train model
    for epoch in range(num_epoch):
        train_loss, train_acc, train_simple_acc, train_time = train(model, train_loader, optimizer, criterion, device, train_dataset)
        # wandbによるログ記録
        wandb.log({"loss": train_loss, "accuracy": train_acc, "simple_accuracy": train_simple_acc})
        print(f"【{epoch + 1}/{num_epoch}】\n"
              f"train time: {train_time:.2f} [s]\n"
              f"train loss: {train_loss:.4f}\n"
              f"train acc: {train_acc:.4f}\n"
              f"train simple acc: {train_simple_acc:.4f}")

    # 提出用ファイルの作成
    model.eval()
    submission = []
    preds = []
    pred_answers = []
    for image, question in test_loader:
        image, question = image.to(device), question.to(device)
        pred = model(image, question)
        pred = pred.argmax(1).cpu().item()
        submission.append(pred)
        preds.append(pred)
        pred_answers.append(train_dataset.idx2answer[pred])

    # CSVファイルに保存
    with open('predictions.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Prediction Index', 'Answer Text'])
        for pred, answer in zip(preds, pred_answers):
            writer.writerow([pred, answer])
    submission = [train_dataset.idx2answer[id] for id in submission]
    submission = np.array(submission)
    torch.save(model.state_dict(), "model.pth")
    np.save("submission_3.npy", submission)
    wandb.finish()

if __name__ == "__main__":
    main()
