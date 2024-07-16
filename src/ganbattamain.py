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
import torch.nn.functional as F
from torchvision.transforms import functional as F
from torchvision import transforms
from torch import Tensor
import wandb
from transformers import CLIPProcessor, CLIPModel
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

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

        # 各ヘッドのための線形変換を個別に定義
        self.query_projs = nn.ModuleList([nn.Linear(dim, self.head_dim) for _ in range(num_heads)])
        self.key_projs = nn.ModuleList([nn.Linear(dim, self.head_dim) for _ in range(num_heads)])
        self.value_projs = nn.ModuleList([nn.Linear(dim, self.head_dim) for _ in range(num_heads)])
        
        self.softmax = nn.Softmax(dim=-1)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, queries, keys, values):
        # queries(text): (batch_size, sequence_length_text, dim)
        # keys, values(image): (batch_size, sequence_length_image, dim)

        # 各ヘッドで線形変換を適用
        queries_heads = [self.dropout(proj(queries)) for proj in self.query_projs]
        keys_heads = [self.dropout(proj(keys)) for proj in self.key_projs] 
        values_heads = [self.dropout(proj(values)) for proj in self.value_projs] 

        # 各ヘッドでアテンションを計算
        attended_values_heads = []
        for i in range(self.num_heads):
            attention_scores = torch.matmul(queries_heads[i], keys_heads[i].transpose(-2, -1)) / (self.head_dim ** 0.5)
            attention_probs = self.softmax(attention_scores)
            attention_probs = self.dropout(attention_probs)
            attended_values_heads.append(torch.matmul(attention_probs, values_heads[i]))

        # ヘッドの結果を結合
        attended_values = torch.cat(attended_values_heads, dim=-1)
        attended_values = self.dropout(attended_values)
        attended_values = self.norm(attended_values)
        
        return attended_values

class FeatureEnhancerLayer(nn.Module):
    def __init__(self, dim, num_heads):
        super(FeatureEnhancerLayer, self).__init__()
        self.self_attention_text = MultiHeadAttention(dim, num_heads)
        self.self_attention_image = MultiHeadAttention(dim, num_heads)
        self.cross_attention_text_to_image = MultiHeadAttention(dim, num_heads)
        self.cross_attention_image_to_text = MultiHeadAttention(dim, num_heads)
        self.ffn_text = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.ffn_image = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.norm_text = nn.LayerNorm(dim)
        self.norm_image = nn.LayerNorm(dim)

    def forward(self, text_features, image_features):
        # Self-Attention
        text_features = self.self_attention_text(text_features, text_features, text_features)
        image_features = self.self_attention_image(image_features, image_features, image_features)

        # Cross-Attention
        text_to_image_features = self.cross_attention_text_to_image(text_features, image_features, image_features)
        image_to_text_features = self.cross_attention_image_to_text(image_features, text_features, text_features)

        # シーケンス長を揃えるためにパディングを追加
        if text_to_image_features.size(1) > image_to_text_features.size(1):
            padding = text_to_image_features.size(1) - image_to_text_features.size(1)
            image_to_text_features = F.pad(image_to_text_features, (0, 0, 0, padding))
            image_features = F.pad(image_features, (0, 0, 0, padding))
        elif image_to_text_features.size(1) > text_to_image_features.size(1):
            padding = image_to_text_features.size(1) - text_to_image_features.size(1)
            text_to_image_features = F.pad(text_to_image_features, (0, 0, 0, padding))
            text_features = F.pad(text_features, (0, 0, 0, padding))

        # Add & Norm
        text_features = self.norm_text(image_to_text_features)
        image_features = self.norm_image(text_to_image_features)

        # Feed Forward Network
        text_features = self.ffn_text(text_features)
        image_features = self.ffn_image(image_features)

        return text_features, image_features

class VQAModel(nn.Module):
    def __init__(self, n_answer: int, text_dim, image_dim, text_encoder, vision_encoder, num_heads=8):
        super().__init__()
        self.text_encoder = text_encoder  # テキストエンコーダー
        self.image_encoder = vision_encoder  # 画像エンコーダー
        self.text_projection = nn.Linear(text_dim, image_dim)  # テキスト特徴量を画像特徴量と同じ次元に変換
        self.text_norm = nn.LayerNorm(image_dim)
        self.feature_enhancer = FeatureEnhancerLayer(image_dim, num_heads)
        self.fc1 = nn.Linear(image_dim*2, 4096)
        self.fc2 = nn.Linear(4096, n_answer)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, text, image):
        text_features = self.text_encoder(**text).last_hidden_state
        image_features = self.image_encoder(image).last_hidden_state
        projected_text_features = self.text_projection(text_features)  # テキスト特徴量を変換
        projected_text_features = self.text_norm(projected_text_features)
        
        # Feature Enhancer Layer
        enhanced_text_features, enhanced_image_features = self.feature_enhancer(projected_text_features, image_features)

        # 各特徴量の平均を計算
        pooled_image_features = torch.mean(enhanced_image_features, dim=1)  # 次元1に沿って平均
        pooled_text_features = torch.mean(enhanced_text_features, dim=1)  # 次元1に沿って平均
        #print("shapeimage", pooled_image_features.shape)
        #print("shapetext", pooled_text_features.shape)
        

        features = torch.cat([pooled_image_features, pooled_text_features], dim=1)
        # print("featuresshape", features.shape)
        output = self.fc1(features)
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
                label_counts[answer_idx] += 1.1  # unanswerableでない回答は1.1倍にカウント
            else:
                label_counts[answer_idx] += 1  # unanswerableの回答は通常通りカウント
        
        # ソフトラベルを計算（正規化）
        soft_labels[i] = label_counts / label_counts.sum()
    
    return soft_labels

# 4. 学習の実装
def train(model, dataloader, optimizer, scheduler_cyclic,criterion, device, dataset, unanswerable_idx):
    dataset = dataset
    model.train()
    scaler = GradScaler()  # Mixed Precision Trainingのためのスケーラー

    total_loss = 0
    total_acc = 0
    simple_acc = 0

    start = time.time()
    for image, question, answers, mode_answer in dataloader:
        # print("image", image)
        # print("question", question)
        image, question, answers, mode_answer = \
            image.to(device), question.to(device), answers.to(device), mode_answer.to(device)
        
        soft_labels = calculate_soft_labels(answers, num_classes=len(dataset.answer2idx), device=device, unanswerable_idx=unanswerable_idx)
        soft_labels = soft_labels.to(device)

        optimizer.zero_grad()
        with autocast():  # Mixed Precision Trainingのためのautocast
            pred = model(question, image)
            loss = criterion(pred, soft_labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler_cyclic.step()  # CyclicLRの更新

        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)  # VQA accuracy
        simple_acc += (pred.argmax(1) == mode_answer).float().mean().item()  # simple accuracy

    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start


class CustomCollate:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def collate_fn(self, batch):
        images, questions, answers, mode_answers = zip(*batch)
        images = torch.stack(images, dim=0)
        questions = self.tokenizer(list(questions), padding=True, return_tensors='pt')
        answers = torch.stack(answers, dim=0)
        mode_answers = torch.tensor(mode_answers, dtype=torch.long)
        return images, questions, answers, mode_answers

    def collate_fntest(self, batch):
        images, questions = zip(*batch)
        images = torch.stack(images, dim=0)
        questions = self.tokenizer(list(questions), padding=True, return_tensors='pt')
        return images, questions

def main():
    # wandb.init(
    #     config={
    #     "architecture": "CLIP+cross-attention+FC",
    #     "dataset": "VizWiz",
    #     "epochs": 100,
    #     "learning_rate": "Cyclic",
    #     "batch_size":512
    #     },
    #     reinit=True
    # )
    # deviceの設定
    set_seed(42)
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

    # collate_fnの生成
    collate_obj = CustomCollate(tokenizer)

    # CLIPに合わせた画像前処理にデータ拡張を追加
    transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # 画像をランダムに水平フリップ
    transforms.RandomVerticalFlip(),
    # transforms.RandomRotation(20),  # 20度の範囲でランダムに回転
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),  # 色調のランダムな調整
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
    unanswerable_idx = train_dataset.answer2idx['unanswerable']

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=512, shuffle=True, collate_fn = collate_obj.collate_fn, num_workers=int(os.cpu_count()/2), pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn = collate_obj.collate_fntest, num_workers=int(os.cpu_count()/2), pin_memory=True)

    model = VQAModel(n_answer=len(train_dataset.answer2idx), text_dim=512, image_dim=768, text_encoder=text_encoder, vision_encoder= vision_encoder).to(device)

    # optimizer / criterion
    num_epoch = 100
    criterion = nn.KLDivLoss(reduction='batchmean')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    # CyclicLRの設定
    scheduler_cyclic = CyclicLR(optimizer, base_lr=0.0001, max_lr=0.0005, step_size_up=200, mode='triangular')
    # ExponentialLRの設定
    scheduler_exp = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.75)

    # train model
    for epoch in range(num_epoch):
        # エポック開始時の学習率を確認
        current_lr = optimizer.param_groups[0]['lr']
        print(f"エポック {epoch+1} の開始時の学習率: {current_lr:.6f}")
        model.train()
        train_loss, train_acc, train_simple_acc, train_time = train(model, train_loader, optimizer, scheduler_cyclic, criterion, device, train_dataset, unanswerable_idx)
        # wandbによるログ記録
        # wandb.log({"loss": train_loss, "accuracy": train_acc, "simple_accuracy": train_simple_acc})
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
            folder_name = f"{current_time.strftime('%Y%m%d%H%M')}_epoch{epoch+1}"
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
    # wandb.finish()

if __name__ == "__main__":
    main()
