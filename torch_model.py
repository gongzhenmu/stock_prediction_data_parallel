import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from blitz.modules import BayesianLSTM
from blitz.utils import variational_estimator

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



from collections import deque
def convertToDf(stockList):
        output=[]
        for i in range(len(stockList)):
            sub = []
            sub.append("day "+str(i+1))
            sub.append(stockList[i])
            output.append(sub)
        return pd.DataFrame(output,columns=["Day","Close"])





def getSingals(df):
    count = int(np.ceil(len(df) * 0.1))
    signals = pd.DataFrame(index=df.index)
    signals['signal'] = 0.0
    signals['trend'] = df['Close']
    signals['RollingMax'] = (signals.trend.shift(1).rolling(count).max())
    signals['RollingMin'] = (signals.trend.shift(1).rolling(count).min())
    signals.loc[signals['RollingMax'] < signals.trend, 'signal'] = -1
    signals.loc[signals['RollingMin'] > signals.trend, 'signal'] = 1
    return signals


def buy_stock(
    df,
    real_movement,
    signal,
    initial_money = 10000,
    max_buy = 1,
    max_sell = 1,):
    """
    real_movement = actual movement in the real world
    delay = how much interval you want to delay to change our decision from buy to sell, vice versa
    initial_state = 1 is buy, 0 is sell
    initial_money = 1000, ignore what kind of currency
    max_buy = max quantity for share to buy
    max_sell = max quantity for share to sell
    """
    starting_money = initial_money
    states_sell = []
    states_buy = []
    current_inventory = 0

    def buy(i, initial_money, current_inventory):
        shares = initial_money // real_movement[i]
        if shares < 1:
            print(
                'day %d: total balances %f, not enough money to buy a unit price %f'
                % (i, initial_money, real_movement[i])
            )
        else:
            if shares > max_buy:
                buy_units = max_buy
            else:
                buy_units = shares
            initial_money -= buy_units * real_movement[i]
            current_inventory += buy_units
            print(
                'day %d: buy %d units at price %f, total balance %f'
                % (i, buy_units, buy_units * real_movement[i], initial_money)
            )
            states_buy.append(0)
        return initial_money, current_inventory

    for i in range(real_movement.shape[0] - int(0.025 * len(df))):
        state = signal[i]
        if state == 1:
            initial_money, current_inventory = buy(
                i, initial_money, current_inventory
            )
            states_buy.append(i)
        elif state == -1:
            if current_inventory == 0:
                    print('day %d: cannot sell anything, inventory 0' % (i))
            else:
                if current_inventory > max_sell:
                    sell_units = max_sell
                else:
                    sell_units = current_inventory
                current_inventory -= sell_units
                total_sell = sell_units * real_movement[i]
                initial_money += total_sell
                try:
                    invest = (
                        (real_movement[i] - real_movement[states_buy[-1]])
                        / real_movement[states_buy[-1]]
                    ) * 100
                except:
                    invest = 0
                print(
                    'day %d, sell %d units at price %f, investment %f %%, total balance %f,'
                    % (i, sell_units, total_sell, invest, initial_money)
                )
            states_sell.append(i)
            
    invest = ((initial_money - starting_money) / starting_money) * 100
    total_gains = initial_money - starting_money
    return states_buy, states_sell, total_gains, invest
    
@variational_estimator
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.lstm_1 = BayesianLSTM(1, 10, prior_sigma_1=1, prior_pi=1, posterior_rho_init=-3.0)
        self.linear = nn.Linear(10, 1)
            
    def forward(self, x):
        # print("\tIn Model: input size", x.size())
        x_, _ = self.lstm_1(x)
        
        #gathering only the latent end-of-sequence for the linear layer
        x_ = x_[:, -1, :]
        x_ = self.linear(x_)
        # print("output size", x_.size())
        return x_
    


WINDOW_SIZE = 21
SCALER = StandardScaler()
X_train, X_test, y_train, y_test = [], [], [], []
Xs, ys = [], []
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = NN()
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  net = nn.DataParallel(net)
else:
  device = torch.device("cpu")
net = net.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

def create_timestamps_ds(series, 
                         timestep_size=WINDOW_SIZE):
    time_stamps = []
    labels = []
    aux_deque = deque(maxlen=timestep_size)
    
    #starting the timestep deque
    for i in range(timestep_size):
        aux_deque.append(0)
    
    #feed the timestamps list
    for i in range(len(series)-1):
        aux_deque.append(series[i])
        time_stamps.append(list(aux_deque))
    
    #feed the labels lsit
    for i in range(len(series)-1):
        labels.append(series[i + 1])
    
    assert len(time_stamps) == len(labels), "Something went wrong"
    
    #torch-tensoring it
    features = torch.tensor(time_stamps[timestep_size:]).float()
    labels = torch.tensor(labels[timestep_size:]).float()
    
    return features, labels

def pred_stock_future(X_test,future_length,sample_nbr=10):
    
    #sorry for that, window_size is a global variable, and so are X_train and Xs
    global WINDOW_SIZE
    global X_train
    global Xs
    global SCALER
    
    #creating auxiliar variables for future prediction
    preds_test = []
    test_begin = X_test[0:1, :, :]
    test_deque = deque(test_begin[0,:,0].tolist(), maxlen=WINDOW_SIZE)

    idx_pred = np.arange(len(X_train), len(Xs))
    
    #predict it and append to list
    for i in range(len(X_test)):
        #print(i)
        as_net_input = torch.tensor(test_deque).unsqueeze(0).unsqueeze(2)
        pred = [net(as_net_input.to(device)).cpu().item() for i in range(sample_nbr)]
        
        
        test_deque.append(torch.tensor(pred).mean().cpu().item())
        preds_test.append(pred)
        
        if i % future_length == 0:
            #our inptus become the i index of our X_test
            #That tweak just helps us with shape issues
            test_begin = X_test[i:i+1, :, :]
            test_deque = deque(test_begin[0,:,0].tolist(), maxlen=WINDOW_SIZE)

    #preds_test = np.array(preds_test).reshape(-1, 1)
    #preds_test_unscaled = scaler.inverse_transform(preds_test)
    
    return idx_pred, preds_test

def get_confidence_intervals(preds_test, ci_multiplier):
    global SCALER
    
    preds_test = torch.tensor(preds_test)
    
    pred_mean = preds_test.mean(1)
    pred_std = preds_test.std(1).detach().cpu().numpy()

    pred_std = torch.tensor((pred_std))
    
    upper_bound = pred_mean + (pred_std * ci_multiplier)
    lower_bound = pred_mean - (pred_std * ci_multiplier)
    #gather unscaled confidence intervals

    pred_mean_final = pred_mean.unsqueeze(1).detach().cpu().numpy()
    pred_mean_unscaled = SCALER.inverse_transform(pred_mean_final)

    upper_bound_unscaled = upper_bound.unsqueeze(1).detach().cpu().numpy()
    upper_bound_unscaled = SCALER.inverse_transform(upper_bound_unscaled)
    
    lower_bound_unscaled = lower_bound.unsqueeze(1).detach().cpu().numpy()
    lower_bound_unscaled = SCALER.inverse_transform(lower_bound_unscaled)
    
    return pred_mean_unscaled, upper_bound_unscaled, lower_bound_unscaled

    
def predict_file(file_name, day_in, day_out):
    print("----------------------------------------------------")
    if torch.cuda.device_count() > 1 :
        print("Using GPU, gpu count: ",torch.cuda.device_count())
    else:
        print("Using CPU")
    print("----------------------------------------------------")
    df = pd.read_csv(file_name)

    close_prices = df["Close"]

    close_prices_arr = np.array(close_prices).reshape(-1, 1)
    close_prices = SCALER.fit_transform(close_prices_arr)

    close_prices_unscaled = df["Close"]

    Xs, ys = create_timestamps_ds(close_prices)

    Xs = Xs[-(day_in+day_out):]
    ys = ys[-(day_in+day_out):]

    X_train, X_test, y_train, y_test = train_test_split(Xs,
                                                        ys,
                                                        test_size=day_out,
                                                        random_state=42,
                                                        shuffle=False)

    ds = torch.utils.data.TensorDataset(X_train, y_train)
    dataloader_train = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True)

   

    

    # Train the model
    iteration = 0
    for epoch in range(10):
        for i, (datapoints, labels) in enumerate(dataloader_train):
            datapoints = datapoints.to(device)
            labels = labels.to(device)

            # output = net(datapoints)
            # print("Outside: input size", input.size(),
            #     "output_size", output.size())
            
            
            optimizer.zero_grad()
            if torch.cuda.device_count() > 1 :
                loss = net.module.sample_elbo(inputs=datapoints,
                                    labels=labels,
                                    criterion=criterion,
                                    sample_nbr=3,
                                    complexity_cost_weight=1/X_train.shape[0])
            else:
                loss = net.sample_elbo(inputs=datapoints,
                                    labels=labels,
                                    criterion=criterion,
                                    sample_nbr=3,
                                    complexity_cost_weight=1/X_train.shape[0])
            loss.backward()
            optimizer.step()
            
            iteration += 1
            
            if iteration%250==0:
                preds_test = net(X_test.to(device))[:,0].unsqueeze(1)
                loss_test = criterion(preds_test, y_test.to(device))
                print("Iteration: {} Val-loss: {:.4f}".format(str(iteration), loss_test))

    future_length=7
    sample_nbr=4
    ci_multiplier=5
    idx_pred, preds_test = pred_stock_future(X_test, future_length, sample_nbr)
    pred_mean_unscaled, upper_bound_unscaled, lower_bound_unscaled = get_confidence_intervals(preds_test,
                                                                                            ci_multiplier)


    def convertToList(npArray):
        result = []
        for i in npArray:
            result.append(i[0])
        return result
    return convertToList(pred_mean_unscaled), convertToList(upper_bound_unscaled), convertToList(lower_bound_unscaled)

#-----------------buy sell stock--------------

if __name__ == "__main__":
    price, price_high, price_low = [], [], []

    price= predict_file('data/AMZN.csv', 2000, 10)
    print(price)
    