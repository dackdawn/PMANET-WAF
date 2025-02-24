from transformers import BertModel, BertTokenizer

model_name = "imvladikon/charbert-bert-wiki"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

model.save_pretrained("./charbert-bert-wiki")
tokenizer.save_pretrained("./charbert-bert-wiki")