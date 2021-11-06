# import libraries
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

# importing the T5 modules, need transformers 2.9.0
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch import cuda
from rouge import Rouge
from torch import cuda

#  the device for GPU usage
device = 'cuda' if cuda.is_available() else 'cpu'

# Reading the dataframe and loading
class preprocess_dataset(Dataset):

    def __init__(self, dataframe, tokenizer, source_len, summ_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = summ_len
        self.highlights = self.data.highlights
        self.article = self.data.article

    def __len__(self):
        return len(self.highlights)

    def __getitem__(self, index):
        article = str(self.article[index])
        article = ' '.join(article.split())

        highlights = str(self.highlights[index])
        highlights = ' '.join(highlights.split())

        source = self.tokenizer.batch_encode_plus([article], max_length= self.source_len, pad_to_max_length=True,return_tensors='pt')
        target = self.tokenizer.batch_encode_plus([highlights], max_length= self.summ_len, pad_to_max_length=True,return_tensors='pt')

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long), 
            'source_mask': source_mask.to(dtype=torch.long), 
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long)
        }


# train function
def train(epoch, tokenizer, model, device, loader, optimizer):
    model.train()
    for _,data in enumerate(loader, 0):
        y = data['target_ids'].to(device, dtype = torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data['source_ids'].to(device, dtype = torch.long)
        mask = data['source_mask'].to(device, dtype = torch.long)

        outputs = model(input_ids = ids, attention_mask = mask, decoder_input_ids=y_ids, lm_labels=lm_labels)
        loss = outputs[0]
        
        if _%500==0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # xm.optimizer_step(optimizer)
        # xm.mark_step()

# validation function
def validate(epoch, tokenizer, model, device, loader):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['target_ids'].to(device, dtype = torch.long)
            ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)

            generated_ids = model.generate(
                input_ids = ids,
                attention_mask = mask, 
                max_length=150, 
                num_beams=2,
                repetition_penalty=2.5, 
                length_penalty=1.0, 
                early_stopping=True
                )
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in y]
            if _%100==0:
                print(f'Completed {_}')

            predictions.extend(preds)
            actuals.extend(target)
    return predictions, actuals

df = pd.read_csv('PATH')
df = df.drop(columns=['Unnamed: 0'])
#df = df.sample(n=100).reset_index(drop=True)

def main():

    TRAIN_BATCH_SIZE = 2
    VALID_BATCH_SIZE = 2 
    TRAIN_EPOCHS = 1  
    VAL_EPOCHS = 1 
    LEARNING_RATE = 1e-4    
    SEED = 42              
    MAX_LEN = 512
    SUMMARY_LEN = 30 

    # random seeds
    torch.manual_seed(SEED) 
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True

    # tokenzier for encoding the highlights
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    

    # import and Pre-Processing the domain data
    # adding the summarzie since it was how T5 model was trained for summarization task.
    df.article = 'summarize: ' + df.article
    print(df.head())

    
    # create Dataset and Dataloader
    train_size = 0.8
    train_dataset=df.sample(frac=train_size, random_state = SEED).reset_index(drop=True)
    val_dataset=df.drop(train_dataset.index).reset_index(drop=True)

    print("FULL Dataset: {}".format(df.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("TEST Dataset: {}".format(val_dataset.shape))


    # create the Training and Validation
    training_set = preprocess_dataset(train_dataset, tokenizer, MAX_LEN, SUMMARY_LEN)
    val_set = preprocess_dataset(val_dataset, tokenizer, MAX_LEN, SUMMARY_LEN)

    # define the parameters for creation of dataloaders
    train_params = {
        'batch_size': TRAIN_BATCH_SIZE,
        'shuffle': True,
        'num_workers': 0
        }

    val_params = {
        'batch_size': VALID_BATCH_SIZE,
        'shuffle': False,
        'num_workers': 0
        }

    # creation of Dataloaders for testing and validation
    training_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)


    
    # define model
    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    model = model.to(device)#model sent to GPU

    # defining the optimizer that will be used to tune the weights of the network in the training session 
    optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)

    # start training loop
    print('Initiating Fine-Tuning for the model on our dataset')

    for epoch in range(TRAIN_EPOCHS):
        train(epoch, tokenizer, model, device, training_loader, optimizer)


    # validation loop and save the resulting file with predictions and acutals in a dataframe
    print('Now generating summaries on our fine tuned model for the validation dataset and saving it in a dataframe')
    for epoch in range(VAL_EPOCHS):
        predictions, actuals = validate(epoch, tokenizer, model, device, val_loader)
        final_df = pd.DataFrame({'Generated highlights':predictions,'Actual highlights':actuals})
        final_df.to_csv('predictions.csv')
        print('Output Files generated for review')

# run main
if __name__ == '__main__':
    main()

# calculate ROUGE scores using prediction.csv file
pred = pd.read_csv('PATH')
pred = pred.drop(columns=['Unnamed: 0'])

hyps = pred["Generated highlights"]
refs = pred["Actual highlights"]

rouge = Rouge()
scores = rouge.get_scores(hyps, refs, avg=True)
scores