import numpy as np
import pandas as pd
import pytorch_lightning as pl
# from appDeclaration import my_app
import torch
import torchmetrics
# from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import (DataLoader,
                              SequentialSampler, TensorDataset)
from tqdm.auto import tqdm
from transformers import (AdamW, RobertaForSequenceClassification,
                          RobertaTokenizerFast,
                          get_linear_schedule_with_warmup)

import trainer as tr
import webScrapping as ws

####################################################################################################
# tokenizer = AutoTokenizer.from_pretrained("Mithil/Bert")

# model = AutoModelForSequenceClassification.from_pretrained("Mithil/Bert")
###################################### BERT WORKING #################################################
result = []
labels = []
customer_reviews = None

##############################################################################################################
# ADDING ELECTRA START
class LitNLPModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = RobertaForSequenceClassification.from_pretrained(CFG.ELECTRA_MODEL, num_labels=2)
        self.f1_score =  torchmetrics.F1Score(num_classes = 2)
        self.recall = torchmetrics.Recall(num_classes = 2)#pl.metrics.F1(num_classes=2)
        
    def forward(self, b_input_ids, b_input_mask, b_labels):
        output = self.model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask,
                        labels=b_labels)
        return output
    
    def training_step(self, batch, batch_idx):
        b_input_ids = batch[0]
        b_input_mask = batch[1]
        b_labels = batch[2]
        z = self(b_input_ids, b_input_mask, b_labels)
        loss = z[0]
        return loss

    def validation_step(self, batch, batch_idx):
        b_input_ids = batch[0]
        b_input_mask = batch[1]
        b_labels = batch[2]
        z = self(b_input_ids, b_input_mask, b_labels)
        val_loss = z[0]
        logits = z[1]
        #logits = logits.detach().cpu().numpy()
        #label_ids = b_labels.to('cpu').numpy()
        self.log('val_loss', val_loss, prog_bar=True)
        self.log('val_f1_score', self.f1_score(logits, b_labels), prog_bar=True)
        return val_loss
    
    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=6e-6)
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=0, 
                                            num_training_steps=189*CFG.EPOCHS)
        return [optimizer], [scheduler]
    
    def flat_accuracy(self, preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)


class CFG:
    ROOT_DIR = '../input/reviews'
    BATCH_SIZE = 32
    # ELECTRA_MODEL = 'Mithil/86RecallRoberta'
    ELECTRA_MODEL = 'roberta-base'
    EPOCHS = 30
    DEVICE = 'cpu'


model = LitNLPModel()
# model = model.load_from_checkpoint('C:/Users/mithi/Desktop/Final Year Project/Extension/colorlib-search-3/colorlib-search-3/results/Output/Trainer.ckpt')
tokenizer = RobertaTokenizerFast.from_pretrained(CFG.ELECTRA_MODEL)
path = 'C:/Users/mithi/Desktop/Final Year Project/Extension/colorlib-search-3/colorlib-search-3/Trainer1.ckpt'
def run_inference(data_dir, model, device, batch_size:int = 32):
    # model.load_state_dict(torch.load('C:/Users/mithi/Desktop/Final Year Project/Extension/colorlib-search-3/colorlib-search-3/Trainer1.ckpt'), map_location=torch.device('cpu'))
    comments = data_dir['review_processed']

    indices = tokenizer.batch_encode_plus(list(comments), max_length=128, add_special_tokens=True, 
                                           return_attention_mask=True, pad_to_max_length=True,
                                           truncation=True)
    input_ids = indices["input_ids"]
    attention_masks = indices["attention_mask"]

    test_inputs = torch.tensor(input_ids)
    test_masks = torch.tensor(attention_masks)

    # Create the DataLoader.
    test_data = TensorDataset(test_inputs, test_masks)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)
# C:/Users/mithi/Desktop/Final Year Project/results
    print('Predicting labels...')
    preds = []
    for fold in range(5):
        # model.load_state_dict(torch.load('C:/Users/mithi/Desktop/Final Year Project/Extension/colorlib-search-3/colorlib-search-3/results/Output/Trainer.ckpt', map_location=torch.device('cpu')))
        # model = model['state_dict']
        # C:/Users/mithi/Desktop/Final Year Project/Extension/colorlib-search-3/colorlib-search-3/results/Output/Trainer1.ckpt
        model.load_state_dict(torch.load('Trainer1.ckpt', map_location=torch.device('cpu'))['state_dict'])
        model.eval()
        model.to(device)

        # Tracking variables 
        predictions = []

        # Predict 
        for batch in tqdm(test_dataloader, total=len(test_dataloader)):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask = batch

            with torch.no_grad():
                outputs = model(b_input_ids, b_input_mask, None)

            logits = outputs[0]

            logits = logits.detach().numpy()

            # Store predictions and true labels
            predictions.append(logits)

        flat_predictions = [item for sublist in predictions for item in sublist]
        flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
        preds.append(flat_predictions)
    return np.round(np.mean(preds, axis=0), 0)

# ELECTRA ENDS
#######################################################################################################
# tokenizer = AutoTokenizer.from_pretrained("Mithil/RobertaAmazonTrained")

# model = AutoModelForSequenceClassification.from_pretrained("Mithil/RobertaAmazonTrained")
##################################### ROBERTA NOT WORKING ##################################################################

device = 'cuda' if torch.cuda.is_available() else 'cpu'   # Running on CPU

def webScrape_reviews_intoDF(url):
    df_dict = {}
    try:
        cust_review = ws.webScrape_Reviews(url)
        cust = [i for i in cust_review if i]
        
        print(len(cust))
        print(len(cust_review))

        names = ws.productName(url)

        df = {'REVIEW_TEXT':list(cust)}
        global customer_reviews
        customer_reviews = cust
        print("THIS IS DF------------------------------------------------------------")
        # print(df)
        df = pd.DataFrame.from_dict(df)
        df = tr.preprocessPandas(df)
        print(len(df['review_processed']))
        df_dict = predict(df, names)
        print(df_dict)
    except Exception as e:
        print("inside webscrape_reviews_intoDF")
        print(str(e))

    return df_dict

def predict(sentences, names):
    r = []
    try:
        global result
        if len(sentences) != 0:
            preds = run_inference(sentences, model, CFG.DEVICE, batch_size=CFG.BATCH_SIZE)
            result = list(preds.astype(int))
            ret_dict = {"REVIEWS":customer_reviews,"Result": result}
            perc = calPerc(ret_dict)
            r = {"Name":names, "Percentage":perc}
    except Exception as e:
        print("inside predict")
        print(str(e))
    return r


def calPerc(dict):
    try:
        df = pd.DataFrame.from_dict(dict)
        items = df['Result'].value_counts()
        percentage = (items.get(1)/len(df['Result']))*100
        print(percentage)
        return percentage
    except Exception as e:
        print('inside calc perc')
        print(str(e))    