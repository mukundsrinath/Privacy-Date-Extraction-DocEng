'''

Trains and evaluates date classification into three classes, given extracted candidate date instances from extract_dates_from_soup.py 
1. Updated date
2. Effective date
3. Other

'''

from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch.optim as optim
from sklearn.metrics import classification_report
import torch
from torch.nn import BCEWithLogitsLoss
from torch.utils.tensorboard import SummaryWriter
import random

print(torch.cuda.is_available())
device = "cuda:1"
device_number = 1
writer = SummaryWriter()
tokenizer = RobertaTokenizer.from_pretrained('mukund/privbert')
MAX_SEQ_LEN = 512
BATCH_SIZE = 64
destination_folder = ''

source_directory = ''
with open(source_directory+'') as f:
    data = f.readlines()

x = []
y = []

for line in data:
    text, label = line.split(' || ')
    x.append(text)
    y.append(int(label.strip()))

c = list(zip(x, y))
random.shuffle(c)
x, y = zip(*c)

_train = x[0:558]
validation = x[558:558+186]
test = x[558+186:]

train_labels = y[0:558]
validation_labels= y[558:558+186]
test_labels = y[558+186:]

_train = [line for line in _train if (len(line) > 0 and not line.isspace())]
validation = [line for line in validation if (len(line) > 0 and not line.isspace())]
test = [line for line in test if (len(line) > 0 and not line.isspace())]

_train = tokenizer(_train, add_special_tokens=True, truncation=True, max_length=None, padding=True)
validation = tokenizer(validation, add_special_tokens=True, truncation=True, max_length=None, padding=True)
test = tokenizer(test, add_special_tokens=True, truncation=True, max_length=None, padding=True)

def batch(text, label, batch=BATCH_SIZE):
    length = len(text['input_ids'])
    for i in range(0, length, batch):
        yield torch.tensor(label[i:i+batch]), torch.tensor(text['input_ids'][i:i+batch]), torch.tensor(text['attention_mask'][i:i+batch])
        	
def save_checkpoint(save_path, model, valid_loss):

    if save_path == None:
        return
    
    state_dict = {'model_state_dict': model.state_dict(),
                  'valid_loss': valid_loss}
    
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')
	
def load_checkpoint(load_path, model):
    
    if load_path==None:
        return
    
    state_dict = torch.load(load_path, map_location=torch.device(device))
    print(f'Model loaded from <== {load_path}')
    
    model.load_state_dict(state_dict['model_state_dict'])
    return state_dict['valid_loss']
	
def train(model, optimizer, _train=_train, train_labels=train_labels, validation=validation, validation_labels=validation_labels, num_epochs = 5, eval_every = 8, file_path = destination_folder, best_valid_loss = float("Inf")):

    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    val_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []	
    model.train()
    for epoch in range(num_epochs):
        for labels, text_ids, text_masks in batch(_train, train_labels):
            labels = torch.nn.functional.one_hot(labels, num_classes=3)
            labels = labels.cuda(device_number)
            text_ids = text_ids.cuda(device_number)
            text_masks = text_masks.cuda(device_number)
            logits = model(text_ids, text_masks).logits
            loss_func = BCEWithLogitsLoss()
            loss = loss_func(logits, labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            writer.add_scalar('train_loss', loss.item(), global_step)
            running_loss += loss.item()
            global_step += 1
			
            if global_step % eval_every == 0:
                model.eval()
                val_iter = 1
                with torch.no_grad():  
                    for val_labels, val_text_ids, val_text_masks, in batch(validation, validation_labels):
                        val_labels = torch.nn.functional.one_hot(val_labels, num_classes=3)
                        val_labels = val_labels.cuda(device_number)
                        val_text_ids = val_text_ids.cuda(device_number)
                        val_text_masks = val_text_masks.cuda(device_number)
                        output = model(val_text_ids, val_text_masks).logits
                        valid_loss = loss_func(output, val_labels.float())
                        valid_loss = valid_loss.item()
                        valid_running_loss += valid_loss
                        val_iter += 1
                        val_step += 1
                        writer.add_scalar('valid_loss', valid_loss, val_step)	

                # evaluation
                average_train_loss = running_loss / eval_every
                average_valid_loss = valid_running_loss / val_iter
                train_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                global_steps_list.append(global_step)
				
                # resetting running values
                running_loss = 0.0                
                valid_running_loss = 0.0
                model.train()
				
                # print progress
                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'.format(epoch+1, num_epochs, global_step, int(num_epochs*len(train_labels)/64), average_train_loss, average_valid_loss))
							  
                # checkpoint
                if best_valid_loss > average_valid_loss:
                    best_valid_loss = average_valid_loss
                    save_checkpoint(file_path + '/' + 'model.pt', model, best_valid_loss)
					
    print('Finished Training!')
	
model = RobertaForSequenceClassification.from_pretrained("mukund/privbert", num_labels=3)
#model = BERTClass()
model.cuda(device_number)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

train(model=model, optimizer=optimizer)

# Evaluation Function

def evaluate(model, test, test_labels):
    y_pred = []
    y_true = []
    test_texts = []
    i = 0

    model.eval()
    with torch.no_grad():
        for labels, text_ids, text_masks in batch(test, test_labels):
            labels = labels.cuda(device_number)
            text_ids = text_ids.cuda(device_number)
            text_masks = text_masks.cuda(device_number)
            output = model(text_ids, text_masks).logits
            y_pred.extend(torch.argmax(output, 1).cpu().detach().numpy().tolist())
            y_true.extend(labels.cpu().detach().numpy().tolist())
            test_texts.extend(x[558+186+BATCH_SIZE*i:558+186+BATCH_SIZE*(i+1)])
            i += 1
    print('Classification Report:')
    print(classification_report(y_true, y_pred, digits=4))


    for i in range(len(y_pred)):
        if y_pred[i] != y_true[i]:
            print(test_texts[i]+', '+str(y_pred[i])+', '+str(y_true[i])+'\n')
	
best_model = RobertaForSequenceClassification.from_pretrained("mukund/privbert", num_labels=3)
best_model.cuda(device_number)
load_checkpoint(destination_folder + '/model.pt', best_model)
evaluate(best_model, test, test_labels)
