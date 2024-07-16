from transformers import T5Tokenizer

# トークナイザーのロード
tokenizer = T5Tokenizer.from_pretrained('google/mt5-small')

# トークナイザーの語彙マッピングを取得
token_to_id = tokenizer.get_vocab()

# IDからトークンへの逆マッピングも作成可能
id_to_token = {id: token for token, id in token_to_id.items()}

# 結果の一部を表示
print(list(token_to_id.items())[:10])  # 最初の10個のトークンとIDを表示
print(list(id_to_token.items())[:10])  # 最初の10個のIDとトークンを表示