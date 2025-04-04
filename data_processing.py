import collections
from pytorch_pretrained_bert import BertTokenizer
from tqdm import tqdm
import pandas as pd
import numpy as np


def dataPreprocess_bert(filename, input_ids, input_types, input_masks, label, urltype):
    """
    Preprocess data from a file containing URLs and labels.

    :param filename: The path to the data file.
    :param input_ids: List to store input char IDs.
    :param input_types: List to store segment IDs.
    :param input_masks: List to store attention masks.
    :param label: List to store labels.
    :param urltype: The type of URL (0 for benign, 1 for malware).
    :return: None
    """
    pad_size = 200
    # Also known as max_len (Based on preliminary analysis,
    # the maximum text length is 38, taking 32 to cover 99%)
    bert_path = "charbert-bert-wiki/"
    tokenizer = BertTokenizer(vocab_file=bert_path + "vocab.txt")  # Initialize the tokenizer
    with open(filename, encoding='utf-8') as f:
        for i, l in tqdm(enumerate(f)):
            x1 = l.strip()
            x1 = tokenizer.tokenize(x1)
            tokens = ["[CLS]"] + x1 + ["[SEP]"]

            # Get input_id, seg_id, att_mask
            ids = tokenizer.convert_tokens_to_ids(tokens)
            types = [0] * (len(ids))
            masks = [1] * len(ids)

            # Pad if short, truncate if long
            if len(ids) < pad_size:
                types = types + [1] * (pad_size - len(ids))  # Set segment to 1 for the masked part
                masks = masks + [0] * (pad_size - len(ids))
                ids = ids + [0] * (pad_size - len(ids))
            else:
                types = types[:pad_size]
                masks = masks[:pad_size]
                ids = ids[:pad_size]
            input_ids.append(ids)
            input_types.append(types)
            input_masks.append(masks)

            #         print(len(ids), len(masks), len(types))
            assert len(ids) == len(masks) == len(types) == pad_size
            if urltype == 1:
                label.append([1])
            elif urltype == 0:
                label.append([0])


def dataPreprocess_charbert(filename, input_ids, input_types, input_masks, char_ids, start_ids, end_ids, label,
                            urltype):
    """
    Preprocess data from a file containing URLs and labels.
    :param end_ids: Charbert Input
    :param start_ids: Charbert Input
    :param char_ids: Charbert Input
    :param filename: The path to the data file.
    :param input_ids: List to store input char IDs.
    :param input_types: List to store segment IDs.
    :param input_masks: List to store attention masks.
    :param label: List to store labels.
    :param urltype: The type of URL (0 for benign, 1 for malware).
    :return: None
    """
    pad_size = 200
    # Also known as max_len (Based on preliminary analysis,
    # the maximum text length is 38, taking 32 to cover 99%)
    bert_path = "charbert-bert-wiki/"
    tokenizer = BertTokenizer(vocab_file=bert_path + "vocab.txt")  # Initialize the tokenizer
    with open(filename, encoding='utf-8') as f:
        for i, l in tqdm(enumerate(f)):

            x1 = l.strip()
            x1 = tokenizer.tokenize(x1)
            tokens = ["[CLS]"] + x1 + ["[SEP]"]

            # input_id, seg_id, att_mask
            ids = tokenizer.convert_tokens_to_ids(tokens)
            types = [0] * (len(ids))
            masks = [1] * len(ids)

            if len(ids) < pad_size:
                types = types + [1] * (pad_size - len(ids))
                masks = masks + [0] * (pad_size - len(ids))
                ids = ids + [0] * (pad_size - len(ids))
            else:
                types = types[:pad_size]
                masks = masks[:pad_size]
                ids = ids[:pad_size]

            input_ids.append(ids)
            input_types.append(types)
            input_masks.append(masks)

            char, start, end = CharbertInput(ids)

            char_ids.append(char)
            start_ids.append(start)
            end_ids.append(end)
            #         print(len(ids), len(masks), len(types))
            assert len(ids) == len(masks) == len(types) == pad_size
            label.append([urltype])


def dataPreprocessFromCSV(filename, input_ids, input_types, input_masks, label, is_CharBert=True, class_to_id_dict=None):
    """
    Preprocess data from a CSV file containing URLs and labels.

    :param is_CharBert: If using the CharBert
    :param filename: The path to the CSV data file.
    :param input_ids: List to store input char IDs.
    :param input_types: List to store segment IDs.
    :param input_masks: List to store attention masks.
    :param label: List to store labels.
    :param class_dict: Dictionary mapping class names to class IDs
    :return: None
    """
    # 默认类别映射（如果未提供）
    if class_to_id_dict is None:
        class_to_id_dict = {
            "benign": 0,
            "malware": 1,
            "phishing": 2,
            "defacement": 3,
            # 其他类别
        }
    pad_size = 200
    bert_path = "charbert-bert-wiki/"
    tokenizer = BertTokenizer(vocab_file=bert_path + "vocab.txt") # Initialize the tokenizer

    data = pd.read_csv(filename, encoding='utf-8')
    char_ids = []
    start_ids = []
    end_ids = []
    for i, row in tqdm(data.iterrows(), total=len(data)):
        # 注意这里裁剪成小数据集了
        # if i >= 1000:
        #     break
        
        y = row['label'].lower()

        if y in class_to_id_dict:
            label.append([class_to_id_dict[y]])
        else:
            # continue
            raise ValueError(f"未知标签值: {y}，不在class_dict中")

        x1 = row['url']  # Replace with the column name in your CSV file where the text data is located
        x1 = tokenizer.tokenize(x1)
        tokens = ["[CLS]"] + x1 + ["[SEP]"]

        # Get input_id, seg_id, att_mask
        ids = tokenizer.convert_tokens_to_ids(tokens)
        types = [0] * (len(ids))
        masks = [1] * len(ids)

        # Pad if short, truncate if long
        if len(ids) < pad_size:
            types = types + [1] * (pad_size - len(ids))  # Set segment to 1 for the masked part
            masks = masks + [0] * (pad_size - len(ids))
            ids = ids + [0] * (pad_size - len(ids))
        else:
            types = types[:pad_size]
            masks = masks[:pad_size]
            ids = ids[:pad_size]
        input_ids.append(ids)
        input_types.append(types)
        input_masks.append(masks)
        if is_CharBert:
            char, start, end = CharbertInput(ids)
            char_ids.append(char)
            start_ids.append(start)
            end_ids.append(end)
        assert len(ids) == len(masks) == len(types) == pad_size

    if is_CharBert:
        return char_ids, start_ids, end_ids


def spiltDatast_bert(input_ids, input_types, input_masks, label):
    """
    Split the dataset into training and testing sets.

    :param input_ids: List of input character IDs.
    :param input_types: List of segment IDs.
    :param input_masks: List of attention masks.
    :param label: List of labels.
    :return: Split datasets for training and testing.
    """
    # Randomly shuffle the indices
    random_order = list(range(len(input_ids)))
    np.random.seed(2020)  # Fix the seed
    np.random.shuffle(random_order)
    print(random_order[:10])

    # Split the dataset into 80% training and 20% testing
    input_ids_train = np.array([input_ids[i] for i in random_order[:int(len(input_ids) * 0.8)]])
    input_types_train = np.array([input_types[i] for i in random_order[:int(len(input_ids) * 0.8)]])
    input_masks_train = np.array([input_masks[i] for i in random_order[:int(len(input_ids) * 0.8)]])
    y_train = np.array([label[i] for i in random_order[:int(len(input_ids) * 0.8)]])
    print("input_ids_train.shape:" + str(input_ids_train.shape))
    print("input_types_train.shape:" + str(input_types_train.shape))
    print("input_masks_train.shape:" + str(input_masks_train.shape))
    print("y_train.shape:" + str(y_train.shape))

    input_ids_test = np.array([input_ids[i] for i in random_order[int(len(input_ids) * 0.8):]])
    input_types_test = np.array([input_types[i] for i in random_order[int(len(input_ids) * 0.8):]])
    input_masks_test = np.array([input_masks[i] for i in random_order[int(len(input_ids) * 0.8):]])
    y_test = np.array([label[i] for i in random_order[int(len(input_ids) * 0.8):]])
    print("input_ids_test.shape:" + str(input_ids_test.shape))
    print("input_types_test.shape:" + str(input_types_test.shape))
    print("input_masks_test.shape:" + str(input_masks_test.shape))
    print("y_test.shape:" + str(y_test.shape))

    return input_ids_train, input_types_train, input_masks_train, y_train, input_ids_test, input_types_test, input_masks_test, y_test


def spiltDatast_charbert(input_ids, input_types, input_masks, char_ids, start_ids, end_ids, label):
    """
    Split the dataset into training and testing sets.

    :param end_ids: Charbert input
    :param start_ids:
    :param char_ids: Charbert input
    :param input_ids: List of input character IDs.
    :param input_types: List of segment IDs.
    :param input_masks: List of attention masks.
    :param label: List of labels.
    :return: Split datasets for training and testing.
    """
    # Randomly shuffle the indices
    random_order = list(range(len(input_ids)))
    np.random.seed(2020)  # Fix the seed
    np.random.shuffle(random_order)
    print(random_order[:10])

    # Split the dataset into 80% training and 20% testing
    input_ids_train = np.array([input_ids[i] for i in random_order[:int(len(input_ids) * 0.8)]])
    input_types_train = np.array([input_types[i] for i in random_order[:int(len(input_ids) * 0.8)]])
    input_masks_train = np.array([input_masks[i] for i in random_order[:int(len(input_ids) * 0.8)]])
    char_ids_train = np.array([char_ids[i] for i in random_order[:int(len(input_ids) * 0.8)]])
    start_ids_train = np.array([start_ids[i] for i in random_order[:int(len(input_ids) * 0.8)]])
    end_ids_train = np.array([end_ids[i] for i in random_order[:int(len(input_ids) * 0.8)]])
    y_train = np.array([label[i] for i in random_order[:int(len(input_ids) * 0.8)]])
    print("input_ids_train.shape:" + str(input_ids_train.shape))
    print("input_types_train.shape:" + str(input_types_train.shape))
    print("input_masks_train.shape:" + str(input_masks_train.shape))
    print("char_ids_train.shape:" + str(char_ids_train.shape))
    print("start_ids_train.shape:" + str(start_ids_train.shape))
    print("end_ids_train.shape:" + str(end_ids_train.shape))
    print("y_train.shape:" + str(y_train.shape))

    input_ids_test = np.array([input_ids[i] for i in random_order[int(len(input_ids) * 0.8):]])
    input_types_test = np.array([input_types[i] for i in random_order[int(len(input_ids) * 0.8):]])
    input_masks_test = np.array([input_masks[i] for i in random_order[int(len(input_ids) * 0.8):]])
    char_ids_test = np.array([char_ids[i] for i in random_order[int(len(input_ids) * 0.8):]])
    start_ids_test = np.array([start_ids[i] for i in random_order[int(len(input_ids) * 0.8):]])
    end_ids_test = np.array([end_ids[i] for i in random_order[int(len(input_ids) * 0.8):]])
    y_test = np.array([label[i] for i in random_order[int(len(input_ids) * 0.8):]])
    print("input_ids_test.shape:" + str(input_ids_test.shape))
    print("input_types_test.shape:" + str(input_types_test.shape))
    print("input_masks_test.shape:" + str(input_masks_test.shape))
    print("char_ids_test.shape:" + str(char_ids_test.shape))
    print("start_ids_test.shape:" + str(start_ids_test.shape))
    print("end_ids_test.shape:" + str(end_ids_test.shape))
    print("y_test.shape:" + str(y_test.shape))

    return (input_ids_train, input_types_train, input_masks_train, char_ids_train, start_ids_train, end_ids_train,
            y_train, input_ids_test, input_types_test, input_masks_test, char_ids_test, start_ids_test, end_ids_test,
            y_test)


def load_char_to_ids_dict(char_vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(char_vocab_file, "r", encoding="utf-8") as reader:
        chars = reader.readlines()
    for index, char in enumerate(chars):
        char = char.rstrip('\n')
        vocab[char] = index
    return vocab


# 移到外面来，这个加载过程耗时太长
CharbertInput_bert_path = "charbert-bert-wiki/"
CharbertInput_tokenizer = BertTokenizer(vocab_file=CharbertInput_bert_path + "vocab.txt")
### 这里的字符级vocab不知道哪来的，所以随便生成了一个
# Change back to use the vocab file in the charbert-bert-wiki
# char_vocab_file = "./simple_vocab.txt"
CharbertInput_char2ids_dict = load_char_to_ids_dict(char_vocab_file='charbert-bert-wiki/vocab.txt')

def CharbertInput(context):
    """Create the additional input for CharBERT."""
    max_length = 200
    char_maxlen = 200
    token = CharbertInput_tokenizer.convert_ids_to_tokens(context)
    token = " ".join(token)
    char_ids=[]
    start_ids=[]
    end_ids=[]

    token = token.strip("##")
    if len(char_ids) < char_maxlen:
        token = token.strip("##")
    for char_idx, c in enumerate(token):
        if len(char_ids) >= char_maxlen:
            break

        if char_idx == 0:
            start_ids.append(len(char_ids))
        if char_idx == len(token) - 1:
            end_ids.append(len(char_ids))

        if c in CharbertInput_char2ids_dict:
            cid = CharbertInput_char2ids_dict[c]
        else:
            cid = CharbertInput_char2ids_dict["[UNK]"]
        char_ids.append(cid)
    if len(char_ids) > char_maxlen:
        char_ids = char_ids[:char_maxlen]
    else:
        pad_len = char_maxlen - len(char_ids)
        char_ids = char_ids + [0] * pad_len
    while len(start_ids) < max_length:
        start_ids.append(char_maxlen - 1)
    while len(end_ids) < max_length:
        end_ids.append(char_maxlen - 1)

    return char_ids, start_ids, end_ids
