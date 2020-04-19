import pandas as pd
from sklearn import preprocessing
import pickle
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
from prepare_dataset import Dataset

#BERT_MODEL = 'bert-base-cased'
BERT_MODEL = 'C://Users//Hammer//PycharmProjects//Bert-fine-tune//model//cased_L-12_H-768_A-12//cased_L-12_H-768_A-12//'
NUM_LABELS = 2
EPOCHS = 1

def train(df, text_field, label_field, epochs=5, model_dir='C://Users//Hammer//PycharmProjects//Bert-fine-tune//model//save_model//'):
    #Load BERT model
    config = BertConfig.from_json_file(BERT_MODEL)
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, do_lower_case=True)
    model = BertForSequenceClassification.from_pretrained(BERT_MODEL, config=config)

    #Convert data to BERT input
    dt = Dataset(tokenizer)
    dataloader = dt.convert_to_dataloader_train(df, text_field, label_field)

    #train model
    predictor = FinBERT()
    predictor.train(tokenizer, dataloader, model, epochs)

    #save model
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

def evaluate(df, text_field, label_field, label_names, model_dir="C://Users//Hammer//PycharmProjects//Bert-fine-tune//model//save_model//"):
    #load pretrained model
    predictor = finBERT()
    predictor.load(model_dir=model_dir)
    tokenizer = BertTokenizer.from_pretrained(model_dir)


    # convert to BERT input
    dt = Dataset(tokenizer)
    dataloader = dt.convert_to_dataloader_eval(df, text_field, label_field)

    #evaluate
    predictor.evaluate(dataloader, df[label_field].to_list(), label_names)

def predict(text, label_dict, model_dir="C://Users//Hammer//PycharmProjects//Bert-fine-tune//model//save_model//"):
    #Load pretrained model
    predictor = finBERT()
    predictor.load(model_dir=model_dir)
    tokenizer = BertTokenizer.from_pretrained(model_dir)

    #convert to BERT input
    dt = Dataset(tokenizer)
    input_ids = dt.convert_example(text)

    #get result
    result = predictor.predict(input_ids, mapping=label_dict)
    return result



if __name__ == '__main__':

    file = 'C://Users//Hammer//PycharmProjects//Bert-fine-tune//data//train_1.xlsx'
    map_path = 'C://Users//Hammer//PycharmProjects//Bert-fine-tune//mapping'
    df_train = pd.read_excel(file)

    text_field = 'SentimentText'
    label_col = 'Sentiment'
    label_field = 'labels'

    labels = df_train[label_col].to_list()
    le = preprocessing.LabelEncoder()
    le.fit(list(set(labels)))
    print('Class list: {}'.format(str(le.classes_)))
    train_labels = list(le.transform(labels))
    mapping = dict(zip(range(len(le.classes_)), le.classes_))
    # save mapping label and encoder
    pickle.dump(mapping, open(map_path + "mapping_labels.p", "wb"))
    pickle.dump(le, open(map_path + "encoder_labels.p", "wb"))
    df_train[label_field] = le.transform(df_train[label_col].to_list())

    #train
    train(df_train, text_field, label_field, epochs=EPOCHS, model_dir=BERT_MODEL)

