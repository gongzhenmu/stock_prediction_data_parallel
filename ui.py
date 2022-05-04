from matplotlib import figure
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import *
from tkinter import *
from tkinter import filedialog as fd
from tkinter import messagebox as ms
import pandas as pd
import numpy as np
from functools import partial
#import lstm_model
import torch_model

class Gui():
    
    def __init__(self):
        self.root = Tk()
        self.bg = "black"
        #self.root.configure(background="grey")
        self.buy = [ 8,  9,  10,  11,  12, 13,  22,  23,  24,  25,  26]
        self.sell = [4, 5, 15, 16, 17, 18, 19, 20, 28]
        self.fileName = ""
        self.exportPath = ""
        self.price = [925.1021463947068, 925.1641946541879, 925.0090783022085, 922.5089139177896, 923.2960570105328, 918.6180579214854, 919.230609635996, 921.1268561332228, 919.4274360297235, 918.2837768488432, 914.6917302361862, 914.2001906316663, 907.6425115475544, 903.3261525414711, 902.1945061741297, 902.6570697962052, 903.469352265285, 905.1864101720107, 905.5161438157006, 909.0729679544716, 913.131139146762, 915.0882657842167, 913.7043612767538, 911.08076641152,
                    911.147996356588, 911.0288256553002, 912.6658389685335, 913.0121692835742, 913.6540700227051, 914.5590610248089]
        self.price_high = []
        self.price_low = []
        self.resultDF = pd.DataFrame()
        self.total_gains=0
        self.invest=0
        self.buttonList = []
        self.maxMin=[0,0]
        self.day_in_default =180
        self.day_out_default = 30
        self.day_in = self.day_in_default
        self.day_out = self.day_out_default
        

    def open_csv(self):
        self.fileName = fd.askopenfilename()
        self.updateMaxAndMin()

    def export_csv(self):
        self.exportPath = fd.asksaveasfilename()
        #print(self.exportPath)
        if not self.exportPath.endswith(".csv"):
            self.exportPath += ".csv"
        if self.resultDF.empty:
            ms.showinfo(title="No prediction results",message="Please start predicting first.")
            return
        self.resultDF.to_csv(self.exportPath)
    def updateMaxAndMin(self):
        self.t.delete(1.0,END)
        text = "File selected: "+ self.fileName+"\n"
        text += "Maximum: "+str(self.maxMin[1])+"\n"+"Minimum: "+str(self.maxMin[0])+"\n"
        self.t.insert(END, text)

    def updateDayInOut(self):
        try:
            self.day_in=int(self.textField1.get())
        except:
            self.day_in = self.day_in_default
        try:
            self.day_out=int(self.textField2.get())
        except:
            self.day_out = self.day_out_default
    def start_prediction(self):
        if self.fileName == "":
            ms.showinfo(title="Start predicting",message="No selected files")
            return
        ms.showinfo(title="Start predicting",message="It takes up to 10min")
        #Tensorflow:
        # self.price= lstm_model.forecastFromData(self.fileName)
        
        self.updateDayInOut()
        #Pytorch:
        self.price,self.price_high,self.price_low= torch_model.predict_file(self.fileName,self.day_in,self.day_out)
        self.price = self.price[:self.day_out]
        #print(self.price)
        self.price_high = self.price_high[:self.day_out]
        self.price_low = self.price_low[:self.day_out]
        self.resultDF = torch_model.convertToDf(self.price)
        signal = torch_model.getSingals(self.resultDF)['signal']
        self.buy,self.sell,self.total_gains,self.invest = torch_model.buy_stock(self.resultDF,self.resultDF.Close,signal)
        self.buy = [value for value in self.buy if value != 0]
        print(self.buy)
        ## add plot here (call plot)
        # As an example, all you need to do to update the graph is call plot with the new X/Y.
        #Clear max and min from prediction
        self.maxMin.clear()
        self.maxMin.append(min(self.price))
        self.maxMin.append(max(self.price))
        #Calculate selling and buying signals
        self.calculateSellAndBuy()

        #plot main chart
        self.plot(self.plot_frame, self.buy,self.sell,deleteChildren=True,mainPlot=True)
        #plot confidence chart
        confidence_interval = [self.price,self.price_high,self.price_low]
        self.plot(self.info_frame_plot, confidence_interval, [2, 1, 0], figsize=(4, 3), dpi=97, includeToolbar=False, deleteChildren=True,confidencePlot=True)

    def buySingalInfo(self, buySingal):
        #print(buySingal)
        self.updateMaxAndMin()
        self.t.insert(END,"Price: "+str(round(self.price[buySingal],3))+"\n")
        local_buy = [buySingal]
        local_sell = []
        for day in self.sell:
            if day > buySingal:
                gain = round(self.price[day]-self.price[buySingal],3)
                text = "Sell on day "+str(day)+" | Gain: "+str(gain)+"\n"
                self.t.insert(END,text)
                if gain > 0:
                    local_sell.append(day)
        self.plot(self.plot_frame, local_buy,local_sell,deleteChildren=True,mainPlot=True,buttonPress = True)
        self.root.update()

    # TODO:
    def sellSingalInfo(self,sellSignal):
        #print(buySingal)
        self.updateMaxAndMin()
        self.t.insert(END,"Price: "+str(round(self.price[sellSignal],3))+"\n")
        local_buy = []
        local_sell = [sellSignal]
        for day in self.buy:
            if day < sellSignal:
                gain = round(self.price[sellSignal]-self.price[day],3)
                text = "Buy on day "+str(day)+" | Gain: "+str(gain)+"\n"
                self.t.insert(END,text)
                if gain > 0:
                    local_buy.append(day)
        self.plot(self.plot_frame, local_buy,local_sell,deleteChildren=True,mainPlot=True,buttonPress = True)
        self.root.update()

    
    def plot(self, root, x,y,figsize=(9, 5), dpi=100, includeToolbar=True, deleteChildren=False,mainPlot = False,confidencePlot = False,buttonPress = False):
        # Delete all children if there are any, so we can update the plot.
        if deleteChildren:
            for widget in root.winfo_children():
                widget.destroy()

        fig = Figure(figsize = figsize, dpi = dpi)
        #fig.patch.set_facecolor(self.bg)
        plot1 = fig.add_subplot(111)
        # Change background color to black
        # plot1.xaxis.set_color("white")
        # plot1.yaxis.set_color("white")
        # plot1.grid(color=self.bg)
        # plot1.tick_params(axis='x', colors='white')
        # plot1.tick_params(axis='y', colors='white')

        if buttonPress and mainPlot:
            
            close = self.resultDF['Close']
            plot1.plot(close, color='black', lw=2.)
            plot1.plot(close, '^', markersize=10, color='r', label = 'buying signal', markevery = x)
            plot1.plot(close, 'v', markersize=10, color='g', label = 'selling signal', markevery = y)
           
            plot1.legend()
            plot1.set_xlabel("Day")
            plot1.set_ylabel("Price")

        elif mainPlot:
            #Start plotting
            close = self.resultDF['Close']
            plot1.plot(close, color='black', lw=2.)
            plot1.plot(close, '^', markersize=10, color='r', label = 'buying signal', markevery = self.buy)
            plot1.plot(close, 'v', markersize=10, color='g', label = 'selling signal', markevery = self.sell)
            # plot1.xaxis.label.set_color("white")
            # plot1.yaxis.label.set_color("white")
            
            #plot1.set_title('total gains %f, total investment %f%%'%(self.total_gains, self.invest))
            #plot1.plot(close,self.buy[1],self.result[self.buy[1]],marker="^",color='m')
            plot1.legend()
            plot1.set_xlabel("Day")
            plot1.set_ylabel("Price")
            
        
        elif confidencePlot:
            print(x)
            plot1.plot(x[0])
            # plot1.plot(x[1], label ="High" )
            # plot1.plot(x[2],label = "Low")
            x_index = [index for index, value in enumerate(x[0])]
            print(x_index)
            plot1.fill_between(x=x_index,
                    y1=x[1],
                    y2=x[2],
                    facecolor='lightseagreen',
                    label="Confidence interval",
                    alpha=0.5)
            
            plot1.legend()
            plot1.set_title("Confidence interval")
        else:
            plot1.plot(x,y)

        

        canvas = FigureCanvasTkAgg(fig, master = root)
        canvas.draw()
        canvas.get_tk_widget().pack()

        if includeToolbar:
            toolbar = NavigationToolbar2Tk(canvas, root)
            #toolbar.config(background=self.bg)
            #toolbar._message_label.config(background=self.bg)
            toolbar.update()
            canvas.get_tk_widget().pack()


    def build_gui(self):
        self.root.title("Stock Prediction")

        self.plot_frame = Frame(self.root, highlightbackground=self.bg, highlightthickness=1)
        self.plot_frame.grid(row=0, column=0, padx=(10, 10), pady=(10, 10))
        

        self.info_frame = Frame(self.root, highlightbackground=self.bg, highlightthickness=1)
        self.info_frame.grid(row=1, column=0, padx=(10, 10), pady=(10, 10))
        # self.info_frame.pack_propagate(0)

        self.buy_sell_frame = Frame(self.root, highlightbackground=self.bg, highlightthickness=0)
        self.buy_sell_frame.grid(row=0, rowspan=2, column=1, padx=(10, 10), pady=(10, 10))

        self.buttons_frame = Frame(self.root)
        
        self.buttons_frame.grid(row=3, column=0, columnspan=2, padx=(5, 5), pady=(5, 5))
        #self.buttons_frame.configure(background=self.bg)

        

        self.open_csv_button = Button(self.buttons_frame, command = self.open_csv, height = 2, width = 11, text = "Open CSV")
        self.open_csv_button.pack(side=LEFT, padx=(15, 15))
        #self.open_csv_button.configure(background=self.bg,foreground="white")
     

        self.start_prediction_button = Button(self.buttons_frame, command = self.start_prediction, height=2, width=11, text="Start Prediction")
        self.start_prediction_button.pack(side=RIGHT, padx=(15, 15))
        self.start_prediction_button.configure(background="#856ff8",foreground="white")

        self.export_button = Button(self.buttons_frame, command = self.export_csv, height=2, width=11, text="Export")
        self.export_button.pack(side=LEFT, padx=(15, 15))

        self.plot(self.plot_frame, x = [-3, -2, -1, 0, 1, 2, 3], y = [3, 2, 1, 0, 1, 2, 3])

        self.info_frame_text = Frame(self.info_frame)
        self.info_frame_text.grid(row=0, column=0)

        self.info_frame_plot = Frame(self.info_frame,highlightbackground=self.bg, highlightthickness=1)
        self.info_frame_plot.grid(row=0, column=1)

        self.inputDays = Frame(self.root)
        self.inputDays.grid(row=2, column=0, columnspan=2, padx=(5, 5), pady=(5, 5))

        self.daysLabelbefore =Label(self.inputDays, text="Days will be trained：")
        self.daysLabelafter = Label(self.inputDays, text="Days will be predicted：")
        self.daysLabelbefore.grid(row=0, column=0)
        self.daysLabelafter.grid(row=1, column=0)

        self.textField1 = Entry(self.inputDays)
        self.textField2  = Entry(self.inputDays)
        self.textField1.grid(row=0, column=1)
        self.textField2.grid(row=1, column=1)

        self.t = Text(self.info_frame_text, width=60, height=18)
        self.t.pack()
        #self.t.configure(background=self.bg,foreground="white")
        self.updateMaxAndMin()

        self.plot(self.info_frame_plot, x=[0, 1], y=[0.3, 0.4], figsize=(4, 3), dpi=97, includeToolbar=False)
        
        

    def calculateSellAndBuy(self):
        #Clear all buttons in buttonList
        if len(self.buttonList)>0:
            for button in self.buttonList:
                button.destroy()
        self.buttonList.append(Button(self.buy_sell_frame, text ="Show all signals", height=4, width=25, command=partial(self.plot, self.plot_frame, self.buy,self.sell,deleteChildren=True,mainPlot=True)))
        self.buttonList[-1].pack()
        for i in range(len(self.buy)):
            # if i%2 != 0:
            self.buttonList.append(Button(self.buy_sell_frame, text ="Buying Signal : " + "day " + str(self.buy[i]), height=2, width=25, command=partial(self.buySingalInfo, self.buy[i])))
            self.buttonList[-1].pack()
                #self.buttonList[-1].configure(background=self.bg,foreground="white")
        for i in range(len(self.sell)):
            self.buttonList.append(Button(self.buy_sell_frame, text ="Selling Signal : " + "day " + str(self.sell[i]), height=2, width=25, command=partial(self.sellSingalInfo, self.sell[i])))
            self.buttonList[-1].pack()
            #self.buttonList[-1].configure(background=self.bg,foreground="white")
        
    def start_gui_loop(self):
        self.root.mainloop()
    
            
    
    def update_gui(self):
        return 0


if __name__ == '__main__':
    ui = Gui()
    ui.build_gui()
    ui.start_gui_loop()