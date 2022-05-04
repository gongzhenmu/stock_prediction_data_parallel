# Stock Predication
There are two py files for his project: ui.py and torch_model.py
Torch.model.py is the file to perform  stock prediction
ui.py is the file to see all the plots and prediction. 
Usage: 
Execute ui.py and select open cvs to pick csv files of price history of companies. You can find cvs files under the data folder.
Once the file is selected you can choose days to be trained/ predicted before pressing "start prediction" button.
The ui.py code will choose to use multiple gpus if available gpus are more than 1 (datasets will be spilted into groups based on the count of gpus). Other wise, cpu will be used for training the data.


Packages:
Sklearn, blitz, matplotlib, tkinter(ui) , pytorch. 
