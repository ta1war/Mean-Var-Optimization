"""
Created on Sun Feb 20 21:29:38 2022

@author: udaytalwar
"""

import pandas_datareader as pdr
import numpy as np
import scipy.optimize 

from datetime import datetime 


def optimize(weights,tickers,start,end,calc_window,rf_rate,short=False,
             price_type='Close',clean='True'):
          
        '''
        weights; *type = list, initial guess weights
        
        start; input as str, data start date format 'YYYY-MM-DD' 
        end; input as str, data end date format 'YYYY-MM-DD'
        
        calc_window; input as str, calculation window for returns, 
        
        rf_rate; desired risk-free rate to benchmark against
        
        price_type; *input as str, default = 'Close', options = 'Adj Close', 
        'Close', 'High','Low','Open', 'Volume'
        
        short = bool, default = 'False', if true weight bounds -1 to 1
        
        clean; bool, default = 'False', if true, the function will drop NA rows
        '''
        start = datetime.strptime(start,'%Y-%m-%d')
        end = datetime.strptime(end,'%Y-%m-%d')
        
        prices = pdr.get_data_yahoo(tickers, start, end)[price_type]
        
        op = prices.pct_change(periods=int(calc_window))
        
        if clean == False:
            op
            
        else:
            op = op.dropna()
        
        vcv_matrix = np.cov(op.T)
        
        avg_return = op.mean()
        
        def sharpe(weights,vcv_matrix,avg_return):        
            
            var = np.dot(np.dot(weights,vcv_matrix),weights)
            
            ret =  np.dot(weights,avg_return)
            
            s_ratio = -(ret-rf_rate)/np.sqrt(var)
            
            return s_ratio
           
        
        sharpe_cons = ({'type':'eq','fun': lambda x: sum(x)-1})
        
        
        if short == False:
            sharpe_bnds = ((0,1),)*len(tickers)
        
        if short == True:
            sharpe_bnds = ((-1,1),)*len(tickers)
        
        optimal = scipy.optimize.minimize(sharpe,weights,bounds = sharpe_bnds,
                                 args = (vcv_matrix,avg_return),
                                 constraints = sharpe_cons)
        
        return optimal
