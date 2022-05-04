# Stock Predication
There are two py files for his project: ui.py and torch_model.py
Torch.model.py is the file to perform  stock prediction
ui.py is the file to see all the plots and prediction. 
Usage: 
python3 torch_model.py {filename} {day_in} {day_out}
defaut value for filename is "data/AMZN.csv" , day_in = 2000, day_out=30

The Torch.model.py code will choose to use multiple gpus if available gpus are more than 1 (datasets will be spilted into groups based on the count of gpus). Other wise, cpu will be used for training the data.
The code might need to run on  machines with 1 or more gpus to see the difference. It will be reflected on the size printed. 

UI.py(Output prediction as plots and buying signals)
Execute ui.py and select open cvs to pick csv files of price history of companies. You can find cvs files under the data folder.
Once the file is selected you can choose days to be trained/ predicted before pressing "start prediction" button.


Packages:
Sklearn, blitz, matplotlib, tkinter(ui) , pytorch. 
