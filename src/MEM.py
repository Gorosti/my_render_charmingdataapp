#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 19:49:41 2023

@author: gorosti
"""

# Import libraries

import pandas as pd
import numpy as np
import plotly.graph_objs as go

import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
# install('openpyxl')

import matplotlib.pyplot as plt

# Linear model
# from scipy.stats import linregress

# minimizer to check the age back calculation
# from scipy.optimize import minimize

from scipy.optimize import curve_fit
from scipy import stats

# from math import log
from math import log10
# from math import isnan

# pip install uncertainties, if needed
try:
    import uncertainties.unumpy as unp
    import uncertainties as unc
except:
    try:
        from pip import main as pipmain
    except:
        from pip._internal import main as pipmain
    pipmain(['install','uncertainties'])
    import uncertainties.unumpy as unp
    import uncertainties as unc


def n_FOM1(x,*p):
    mu=p[0]
    sigma_phi_t = p[1]
    n0=1;
    return n0*np.exp(-sigma_phi_t*np.exp(-mu*x))
def n_FOM1_unp(x,*p):
    mu=p[0]
    sigma_phi_t = p[1]
    n0=1;
    return n0*unp.exp(-sigma_phi_t*unp.exp(-mu*x))

# FOM (burial No. 1): 
def n_FOM2(x, *p):
    f = p[2]  # f = Dose rate / D_c*tB
    n_i = n_FOM1(x, *p)
    return (n_i-1)*np.exp(-f)+1
def n_FOM2_unp(x, *p):
    f = p[2]  # f = Dose rate / D_c*tB
    n_i = n_FOM1_unp(x, *p)
    return (n_i-1)*unp.exp(-f)+1

# FOM, exposure No. 2: 
def n_FOM3(x, *p):
    mu=p[0]
    sigma_phi_t_2 = p[3]
    n_ii = n_FOM2(x, *p)
    return n_ii*np.exp(-sigma_phi_t_2*np.exp(-mu*x))
def n_FOM3_unp(x, *p):
    mu=p[0]
    sigma_phi_t_2 = p[3]
    n_ii = n_FOM2_unp(x, *p)
    return n_ii*unp.exp(-sigma_phi_t_2*unp.exp(-mu*x))

# FOM, Burial No. 2: 
def n_FOM4(x, *p):
    f_2=p[4]
    n_iii = n_FOM3(x, *p)
    return (n_iii-1)*np.exp(-f_2)+1
def n_FOM4_unp(x, *p):
    f_2=p[4]
    n_iii = n_FOM3_unp(x, *p)
    return (n_iii-1)*unp.exp(-f_2)+1

# FOM, exposure No. 3: 
def n_FOM5(x, *p):
    mu=p[0]
    sigma_phi_t_3 = p[5]
    n_ii = n_FOM4(x, *p)
    return n_ii*np.exp(-sigma_phi_t_3*np.exp(-mu*x))
def n_FOM5_unp(x, *p):
    mu=p[0]
    sigma_phi_t_3 = p[5]
    n_ii = n_FOM4_unp(x, *p)
    return n_ii*unp.exp(-sigma_phi_t_3*unp.exp(-mu*x))

#GOM, exp no 1
def n_GOM1(x, *p):
    r=p[0]
    mu=p[1]
    sigma_phi_t = p[2]
    # n0=1;
    return ((r-1)*sigma_phi_t*np.exp(-mu*x)+1)**(1/(1-r))     
def n_GOM1_unp(x, *p):
    r=p[0]
    mu=p[1]
    sigma_phi_t = p[2]
    # n0=1;
    return ((r-1)*sigma_phi_t*unp.exp(-mu*x)+1)**(1/(1-r))     

# multiple event model (burial No. 1): 
def n_GOM2(x,*p):  # f = Dose rate / D_c*tB
    f=p[3]
    n_i = n_GOM1(x,*p)
    return (n_i-1)*np.exp(-f)+1
def n_GOM2_unp(x,*p):  # f = Dose rate / D_c*tB
    f=p[3]
    n_i = n_GOM1_unp(x,*p)
    return (n_i-1)*unp.exp(-f)+1

# multiple event model, exposure No. 2: 
def n_GOM3(x, *p):
    r=p[0]
    mu=p[1]
    sigma_phi_t_2=p[4]
    n_ii = n_GOM2(x, *p)
    return ((r-1)*sigma_phi_t_2*np.exp(-mu*x)+n_ii**(1-r))**(1/(1-r))     
def n_GOM3_unp(x, *p):
    r=p[0]
    mu=p[1]
    sigma_phi_t_2=p[4]
    n_ii = n_GOM2_unp(x, *p)
    return ((r-1)*sigma_phi_t_2*unp.exp(-mu*x)+n_ii**(1-r))**(1/(1-r))     

# multiple event model, Burial No. 2: 
def n_GOM4(x, *p): 
    f_2 = p[5]
    n_iii = n_GOM3(x, *p)
    return (n_iii-1)*np.exp(-f_2)+1
def n_GOM4_unp(x, *p): 
    f_2 = p[5]
    n_iii = n_GOM3_unp(x, *p)
    return (n_iii-1)*unp.exp(-f_2)+1

# GOM, exposure No. 3: 
def n_GOM5(x, *p):
    r=p[0]
    mu=p[1]
    sigma_phi_t_3=p[6]
    n_ii = n_GOM4(x, *p)
    return ((r-1)*sigma_phi_t_3*np.exp(-mu*x)+n_ii**(1-r))**(1/(1-r))     
def n_GOM5_unp(x, *p):
    r=p[0]
    mu=p[1]
    sigma_phi_t_3=p[6]
    n_ii = n_GOM4_unp(x, *p)
    return ((r-1)*sigma_phi_t_3*unp.exp(-mu*x)+n_ii**(1-r))**(1/(1-r)) 
    
    
class MEM():
    
    def __init__(self, filepath): 
        self.filepath = filepath
        self.x = None
        self.n = None
        self.n_err = None
    
    
    def read_data_array(self, sheet_name, weight):
        df_dict = pd.read_excel(self.filepath,
                       #engine ='openpyxl', 
                       usecols = [0,1,2],
                       header = None, 
                       sheet_name = sheet_name)
        self.idx = ()
        for i in range(len(sheet_name)):
            x_temp = np.array(df_dict[sheet_name[i]][0])
            lenght = len(x_temp)
            self.idx+= (lenght,)
        
        self.t = pd.concat(df_dict.values())
        self.comboX=np.array(self.t[0])
        self.comboY=np.array(self.t[1])
        self.comboSE=np.array(self.t[2])
        xall = []          
        yall = []
        errall = []   
        for sheet in sheet_name:

                    # Read data for each sheet and sort values by increasing depth
                    x_sheet = df_dict[sheet][0].sort_values()
                    y_sheet = df_dict[sheet][1][x_sheet.index].values
                    err_sheet = df_dict[sheet][2][x_sheet.index].values

                    # Convert into lists

                    x_sheet = list(x_sheet)
                    y_sheet = list(y_sheet)
                    err_sheet = list(err_sheet)

                    xall.append(x_sheet)
                    yall.append(y_sheet)
                    errall.append(err_sheet)
        self.xall = xall
        self.yall = yall
        self.errall = errall

        
        if weight == 'y':
            self.sigma=self.comboY   # uncertainties on y
            self.absolute_sigma=False
        if weight == 'se':
            self.sigma=self.comboSE   # uncertainties on y
            self.absolute_sigma=False
        if weight =='none':
            self.sigma=None   # uncertainties on y
            self.absolute_sigma=False
                     
    
    def define_model(self, material, guess, sheet_name, model_idx,residual):
        n_sample = len(sheet_name)
        
        def idP0(e,i): # index of parameters for sample i for event no. e
                ini = np.arange(1,(n_sample-1)*6+2,6)
                #fin = np.arange(e+1,(n_sample-1)*6+6,6)
                #return np.append([0],np.arange(ini[i],fin[i]+1,1))
                return np.append([0],np.arange(ini[i],ini[i]+e+1,1))
        self.idP0 = idP0
        
         
       
        #Number of prameters used in total and in individual fits
        # All index of parameter used (P)
        num_par = []
        P = np.array([])
        for i in range(0,len(sheet_name)):
            p_used = self.idP0(model_idx[i],i)
            if material == 'F':
                num_par.append(len(p_used)) #number of parameter used (used for uncertainty calc.)
                P=np.append(P,list(p_used[1:])) 
            if material == 'Q':
                num_par.append(len(p_used[1:])) #not counting in the order as it is not used
                P=np.append(P,list(p_used[1:])) 
        self.num_par = num_par
        P = P.astype(np.int32) 
        P = np.append([0],P) 
        if material == 'F':
            self.P = P[0:]
        if material == 'Q':
            self.P = P[1:]
        #print(self.num_par)
        print(self.P)
        
        
        def IDX(e,i,j): 
                ind = np.searchsorted(self.P,self.idP0(e,i)[j])
                return ind
        self.IDX = IDX
        
        # self.P is a list of index for parameters to include from the Guess values.
        r = residual
        if material == 'F':
            def function(data,i,e, *q):# not all parameters are used here, r is shared
                if e==  1:   
                    return r+(1-r)*n_GOM1(data,*np.array([q[0],q[self.IDX(e,i,1)],q[self.IDX(e,i,2)]]))
                if e==  2:
                    return r+(1-r)*n_GOM2(data,*np.array([q[0],q[self.IDX(e,i,1)],q[self.IDX(e,i,2)],q[self.IDX(e,i,3)]]))
                if e==  3:      
                    return r+(1-r)*n_GOM3(data,*np.array([q[0],q[self.IDX(e,i,1)],q[self.IDX(e,i,2)],q[self.IDX(e,i,3)],q[self.IDX(e,i,4)]]))
                if e==  4:      
                    return r+(1-r)*n_GOM4(data,*np.array([q[0],q[self.IDX(e,i,1)],q[self.IDX(e,i,2)],q[self.IDX(e,i,3)],q[self.IDX(e,i,4)],q[self.IDX(e,i,5)]]))
                if e==  5:      
                    return r+(1-r)*n_GOM5(data,*np.array([q[0],q[self.IDX(e,i,1)],q[self.IDX(e,i,2)],q[self.IDX(e,i,3)],q[self.IDX(e,i,4)],q[self.IDX(e,i,5)],q[self.IDX(e,i,6)]]))
            
            self.fun = function

            def function_unp(data,i,e, *q):# not all parameters are used here, r is shared
                if e==  1:   
                    return r+(1-r)*n_GOM1_unp(data,*np.array([q[0],q[self.IDX(e,i,1)],q[self.IDX(e,i,2)]]))
                if e==  2:
                    return r+(1-r)*n_GOM2_unp(data,*np.array([q[0],q[self.IDX(e,i,1)],q[self.IDX(e,i,2)],q[self.IDX(e,i,3)]]))
                if e==  3:      
                    return r+(1-r)*n_GOM3_unp(data,*np.array([q[0],q[self.IDX(e,i,1)],q[self.IDX(e,i,2)],q[self.IDX(e,i,3)],q[self.IDX(e,i,4)]]))
                if e==  4:      
                    return r+(1-r)*n_GOM4_unp(data,*np.array([q[0],q[self.IDX(e,i,1)],q[self.IDX(e,i,2)],q[self.IDX(e,i,3)],q[self.IDX(e,i,4)],q[self.IDX(e,i,5)]]))
                if e==  5:      
                    return r+(1-r)*n_GOM5_unp(data,*np.array([q[0],q[self.IDX(e,i,1)],q[self.IDX(e,i,2)],q[self.IDX(e,i,3)],q[self.IDX(e,i,4)],q[self.IDX(e,i,5)],q[self.IDX(e,i,6)]]))

            self.fun_unp = function_unp
            self.Bounds = (np.concatenate([[1], guess[self.P[1:]]*0]),  np.inf)
        
        if material == 'Q':
            def function(data,i,e, *q):# not all parameters are used here, r is shared
                if e==  1:   
                    return r+(1-r)*n_FOM1(data,*np.array([q[self.IDX(e,i,1)],q[self.IDX(e,i,2)]]))
                if e==  2:
                    return r+(1-r)*n_FOM2(data,*np.array([q[self.IDX(e,i,1)],q[self.IDX(e,i,2)],q[self.IDX(e,i,3)]]))
                if e==  3:      
                    return r+(1-r)*n_FOM3(data,*np.array([q[self.IDX(e,i,1)],q[self.IDX(e,i,2)],q[self.IDX(e,i,3)],q[self.IDX(e,i,4)]]))
                if e==  4:      
                    return r+(1-r)*n_FOM4(data,*np.array([q[self.IDX(e,i,1)],q[self.IDX(e,i,2)],q[self.IDX(e,i,3)],q[self.IDX(e,i,4)],q[self.IDX(e,i,5)]]))
                if e==  5:      
                    return r+(1-r)*n_FOM5(data,*np.array([q[self.IDX(e,i,1)],q[self.IDX(e,i,2)],q[self.IDX(e,i,3)],q[self.IDX(e,i,4)],q[self.IDX(e,i,5)],q[self.IDX(e,i,6)]]))

            self.fun = function
            
            def function_unp(data,i,e, *q):# not all parameters are used here, r is shared
                if e==  1:   
                    return r+(1-r)*n_FOM1_unp(data,*np.array([q[self.IDX(e,i,1)],q[self.IDX(e,i,2)]]))
                if e==  2:
                    return r+(1-r)*n_FOM2_unp(data,*np.array([q[self.IDX(e,i,1)],q[self.IDX(e,i,2)],q[self.IDX(e,i,3)]]))
                if e==  3:      
                    return r+(1-r)*n_FOM3_unp(data,*np.array([q[self.IDX(e,i,1)],q[self.IDX(e,i,2)],q[self.IDX(e,i,3)],q[self.IDX(e,i,4)]]))
                if e==  4:      
                    return r+(1-r)*n_FOM4_unp(data,*np.array([q[self.IDX(e,i,1)],q[self.IDX(e,i,2)],q[self.IDX(e,i,3)],q[self.IDX(e,i,4)],q[self.IDX(e,i,5)]]))
                if e==  5:      
                    return r+(1-r)*n_FOM5_unp(data,*np.array([q[self.IDX(e,i,1)],q[self.IDX(e,i,2)],q[self.IDX(e,i,3)],q[self.IDX(e,i,4)],q[self.IDX(e,i,5)],q[self.IDX(e,i,6)]]))

            self.fun_unp = function_unp
            self.Bounds = (np.concatenate([[0], guess[self.P[1:]]*0]),  np.inf)

        self.P0 = guess[self.P]
     
            
    def multiple_fiting_array(self,sheet_name,model_idx):
    
            def DATA(comboData,i, *p):
                if i==0:
                    initial=i
                    final = np.sum(self.idx[:1])
                if i>0:
                    initial=np.sum(self.idx[:i]) 
                    final = np.sum(self.idx[:i+1])
                return comboData[initial:final]
    
            def result_i(comboData,i, *p):
                data = DATA(comboData,i, *p)
                return self.fun(data,i,model_idx[i], *p)

            def g(comboData,*p):
                R= result_i(comboData,0, *p)
                for i in range(1,len(sheet_name)):
                    R = np.append(R,result_i(comboData,i, *p))
                return R
                       
            self.model_new = g
            
            
    
    def plot_figure_guess(self, sheet_name,logID,model_idx,Site_idx,Thickness):
         # define a figure environment
        self.fig, (ax1) = plt.subplots()
        colors = []     
        for i in range(len(sheet_name)):
            xmax = max(self.xall[i])+0.5
            xi = np.linspace(0,xmax,100)
            y_guess_temp = self.fun(xi,i,model_idx[i],*self.P0)
            if Site_idx[i] == 'End':
                xfit = Thickness[i]-xi
                xdata =Thickness[i]-self.xall[i]
            if Site_idx[i] == 'Front':
                xfit = xi
                xdata =self.xall[i] 
            plt.plot(xfit, y_guess_temp,color = 'green',label='Guess') # plot the equation using the fitted parameters
            plt.errorbar(xdata, self.yall[i], self.errall[i], ls='',
                     marker = 'o', c = 'black',
                     markerfacecolor  = 'white', zorder = 100)              
        ax1.legend()
        # label the axes of axes[0]
        ax1.set_xlabel("Depth [mm]")
        ax1.set_ylabel("Normalised luminescence signal")
        ax1.set_title("Guess")
        if logID == 'y':
                ax1.set_yscale('log')
        #add legend to plot
        #plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='lower right') 


    
    def run_model_array(self):
            self.popt, self.pcov = curve_fit(
                f=self.model_new,       # model function
                xdata=self.comboX,   # x data
                ydata=self.comboY,   # y data
                p0=self.P0,      # initial value of the parameters
                sigma=self.sigma,   # 
                absolute_sigma=self.absolute_sigma,
                check_finite=True,
                bounds=self.Bounds,
                method='trf',
                jac=None,
                full_output=False
                )
            self.perr = np.sqrt(np.diag(self.pcov))  # standart deviation on parameters

            print('sd of parameters' + str(self.perr))
            
            
    #def run_model_array_MonteCarlo(self):
    #    self.popt_MC = list()
    #    self.pcov_MC = list()
    #    for j in range(0,1000):
    #        popt_MC, pcov_MC = curve_fit(
    #            f=self.model_new,       # model function
    #            xdata=self.comboX,   # x data
    #            ydata=np.random.normal(self.comboY,self.comboSE),   # y data
    #            p0=self.P0,      # initial value of the parameters
    #            check_finite=True,
    #            bounds=self.Bounds,
    #            method='trf',
    #            jac=None,
    #            full_output=False
    #            )
    #        self.popt_MC.append(list(popt_MC))
        


    
    
    def Well_bleached_depth(self,WBtolerance,model_idx):
        x_WB = []
        for i in range(len(self.xall)):
            event = model_idx[i]
            # calculate parameter confidence interval
            self.poptCI = unc.correlated_values(self.popt, self.pcov)

            # depth values in mm
            xmax = max(self.xall[i])+0.5
            xi = np.linspace(0,xmax,1000)
            yi = self.fun_unp(xi,i,model_idx[i],*self.poptCI)

            if event >=2:
                yi_prev1 = self.fun_unp(xi,i,1,*self.poptCI) #exposure #1
                if event==2: #exposure + burial
                    idx_1 = np.argmin(np.abs(yi_prev1/yi-WBtolerance)) #find index for exp1 < x% of burial 1
                    x_wb1 = xi[idx_1]
                    x_wb2 =np.nan
                    x_wb3 =np.nan
                    x_wb4 =np.nan
                if event>=3:
                    yi_prev2 = self.fun_unp(xi,i,2,*self.poptCI) #burial #1
                    idx_1 = np.argmin(np.abs(yi_prev1/yi_prev2-WBtolerance)) #find index for exp1 < x% of burial 1
                    x_wb1 = xi[idx_1]
                    if event ==3: #exposure #2
                        idx_2 = np.argmin(np.abs(yi/yi_prev2-(1-WBtolerance)))  #find index for burial1 > (1-x)% of exposure 2
                        x_wb2 = xi[idx_2]
                        x_wb3 =np.nan
                        x_wb4 =np.nan
                    if event >=4:
                        yi_prev3 = self.fun_unp(xi,i,3,*self.poptCI) #exposure # 2
                        idx_2 = np.argmin(np.abs(yi_prev3/yi_prev2-(1-WBtolerance)))  #find index for burial1 > (1-x)% of exposure 2
                        x_wb2 = xi[idx_2]
                        if event ==4: #burial #2
                            idx_3 = np.argmin(np.abs(yi_prev3/yi-WBtolerance)) #find index for exp2 < x% of burial 2
                            x_wb3 = xi[idx_3]
                            x_wb4 =np.nan
                        if event ==5: #exposure #3
                            yi_prev4 = self.fun_unp(xi,i,4,*self.poptCI) #burial # 2
                            idx_3 = np.argmin(np.abs(yi_prev3/yi_prev4-WBtolerance)) #find index for exp2 < x% of burial 2
                            x_wb3 = xi[idx_3]
                            idx_4 = np.argmin(np.abs(yi/yi_prev4-(1-WBtolerance)))  #find index for burial2 > (1-x)% of exposure 3
                            x_wb4 = xi[idx_4]
                                
                            print("well bleach depth 4=" , x_wb4)             
                        print("well bleach depth 3=" , x_wb3)             
                    print("well bleach depth 2=" , x_wb2)           
                print("well bleach depth 1=" , x_wb1)    
            if event==1:
                x_wb1 =np.nan
                x_wb2 =np.nan
                x_wb3 =np.nan
                x_wb4 =np.nan


            x_WB.append([x_wb1, x_wb2, x_wb3,x_wb4])
            #return x_WB
        self.x_WB = x_WB

    def Parameterrresults(self,sheet_name,model_idx):
        Par_fit = []
        Par_fit_err = []
        Parameters = ('Kinetic order','Attenuation coeff (mu)','sigma*phi*te1', 'F*tb1','sigma*phi*te2', 'F*tb2' )
        
        for i in range(0,len(sheet_name)):
            ind = np.searchsorted(self.P,self.idP0(model_idx[i],i))
            fit_list = list(self.popt[ind])
            Par_fit.append(fit_list)
            fit_err_list = list(self.perr[ind]) #standard deviation of parameters
            Par_fit_err.append(fit_err_list)
        self.Par_fit = Par_fit
        self.Par_fit_err = Par_fit_err
        
        for i in range(0,len(sheet_name)): #different samples
            for j in range(0,len(self.Par_fit[i])): #different fitting parameters
                print('Sample' + str(i+1) + ':' + Parameters[j] + '=' + str(self.Par_fit[i][j]))
        
    def SingleCal(self,sheet_name,model_idx,Known_exp_time1,Known_exp_time2, Known_exp_time3):     
        sf_recent = []
        sf_cal1 = []
        sf_cal2 = []
        sf_cal3 = []
        rsd_sf_cal1 = []
        rsd_sf_cal2 = []
        rsd_sf_cal3 = []
        t_e1_cal1 = [] #exposure time 1
        t_e1_cal2 = [] #exposure time 1
        t_e1_cal3 = [] #exposure time 1
        t_e2_cal1 = [] #exposure time 2
        t_e2_cal2 = [] #exposure time 2
        t_e2_cal3 = [] #exposure time 2
        t_e3_cal1 = [] #exposure time 3
        t_e3_cal2 = [] #exposure time 3
        t_e3_cal3 = [] #exposure time 3
        sd_t_e1_cal1 = [] #sd on exposure time 1
        sd_t_e1_cal2 = [] #sd on exposure time 1
        sd_t_e1_cal3 = [] #sd on exposure time 1
        sd_t_e2_cal1 = [] #sd on exposure time 2
        sd_t_e2_cal2 = [] #sd on exposure time 2
        sd_t_e2_cal3 = [] #sd on exposure time 2
        sd_t_e3_cal1 = [] #sd on exposure time 3
        sd_t_e3_cal2 = [] #sd on exposure time 3
        sd_t_e3_cal3 = [] #sd on exposure time 3

       
        #find sigma_phi from known age samples
        for i in range(0,len(sheet_name)): #different samples 
            nan_idx = Known_exp_time1[i]/Known_exp_time1[i]
            sf_cal1_list = self.Par_fit[i][2]/Known_exp_time1[i]
            sf_cal1.append(sf_cal1_list)
            rsd_sf_cal1_list = self.Par_fit_err[i][2]/self.Par_fit[i][2]*nan_idx #relative standard deviation
            rsd_sf_cal1.append(rsd_sf_cal1_list)
            if model_idx[i]>2: #
                nan_idx = Known_exp_time2[i]/Known_exp_time2[i]
                sf_cal2_list = self.Par_fit[i][4]/Known_exp_time2[i]
                sf_cal2.append(sf_cal2_list)
                rsd_sf_cal2_list = self.Par_fit_err[i][4]/self.Par_fit[i][4]*nan_idx #relative standard deviation
                rsd_sf_cal2.append(rsd_sf_cal2_list)
            if model_idx[i]<3: #if only one exposure
                sf_cal2_list = [np.nan]
                sf_cal2.append(sf_cal2_list)
                rsd_sf_cal2_list = [np.nan]
                rsd_sf_cal2.append(rsd_sf_cal2_list)
            if model_idx[i]>4: # exposure no. 3
                nan_idx = Known_exp_time3[i]/Known_exp_time3[i]
                sf_cal3_list = self.Par_fit[i][6]/Known_exp_time3[i]
                sf_cal3.append(sf_cal3_list)
                rsd_sf_cal3_list = self.Par_fit_err[i][6]/self.Par_fit[i][6]*nan_idx #relative standard deviation
                rsd_sf_cal3.append(rsd_sf_cal3_list)
            if model_idx[i]<5: #if only one or two exposures
                sf_cal3_list = [np.nan]
                sf_cal3.append(sf_cal3_list)
                rsd_sf_cal3_list = [np.nan]
                rsd_sf_cal3.append(rsd_sf_cal3_list)

        rsd_sf_cal1 = np.array(rsd_sf_cal1)
        sf_cal1 = np.array(sf_cal1)
        rsd_sf_cal2 = np.array(rsd_sf_cal2)
        sf_cal2 = np.array(sf_cal2)
        rsd_sf_cal3 = np.array(rsd_sf_cal3)
        sf_cal3 = np.array(sf_cal3)
        
        
        w1 = 1/(rsd_sf_cal1*sf_cal1)**2
        if np.nansum(w1)>0:
            self.sf_cal1 = np.nansum(sf_cal1*w1)/np.nansum(w1)
            self.sd_sf_cal1 = np.sqrt(1/np.nansum(w1))*np.sqrt(np.count_nonzero(np.isfinite(w1)))
        if np.nansum(w1)==0:
            self.sf_cal1 = np.array(np.nan)
            self.sd_sf_cal1 = np.array(np.nan)
        w2 = 1/(rsd_sf_cal2*sf_cal2)**2
        if np.nansum(w2)>0:
            self.sf_cal2 = np.nansum(sf_cal2*w2)/np.nansum(w2)
            self.sd_sf_cal2 = np.sqrt(1/np.nansum(w2))*np.sqrt(np.count_nonzero(np.isfinite(w2)))
        if np.nansum(w2)==0:
            self.sf_cal2 = np.array(np.nan)
            self.sd_sf_cal2 = np.array(np.nan)

        w3 = 1/(rsd_sf_cal3*sf_cal3)**2
        if np.nansum(w3)>0:
            self.sf_cal3 = np.nansum(sf_cal2*w3)/np.nansum(w3)
            self.sd_sf_cal3 = np.sqrt(1/np.nansum(w3))*np.sqrt(np.count_nonzero(np.isfinite(w3)))
        if np.nansum(w3)==0:
            self.sf_cal3 = np.array(np.nan)
            self.sd_sf_cal3 = np.array(np.nan)
      
            
        #find exposure time using single Cal (SC) or multi cal (MC)
        def sd_ratio(A,B,a,b):
            return A/B*np.sqrt((a/A)**2+(b/B)**2)
        
        for i in range(0,len(sheet_name)): #different samples
                sft = self.Par_fit[i][2]
                sd_sft = self.Par_fit_err[i][2]
                t_e1_cal1_list = (sft/self.sf_cal1)
                t_e1_cal2_list = (sft/self.sf_cal2)
                t_e1_cal3_list = (sft/self.sf_cal3)
                sd_t_e1_cal1_list = sd_ratio(sft,self.sf_cal1,sd_sft,self.sd_sf_cal1)
                sd_t_e1_cal2_list = sd_ratio(sft,self.sf_cal2,sd_sft,self.sd_sf_cal2)
                sd_t_e1_cal3_list = sd_ratio(sft,self.sf_cal3,sd_sft,self.sd_sf_cal3)
                if np.isfinite(Known_exp_time1[i]): #if the exposure time is known write nan
                    t_e1_cal1_list = [np.nan]
                    sd_t_e1_cal1_list = [np.nan]
                t_e1_cal1.append(t_e1_cal1_list)
                t_e1_cal2.append(t_e1_cal2_list)
                t_e1_cal3.append(t_e1_cal3_list)
                sd_t_e1_cal1.append(sd_t_e1_cal1_list)
                sd_t_e1_cal2.append(sd_t_e1_cal2_list)
                sd_t_e1_cal3.append(sd_t_e1_cal3_list)
              
                if model_idx[i]>2: #two exposures
                    sft = self.Par_fit[i][4]
                    sd_sft = self.Par_fit_err[i][4]
                    t_e2_cal1_list = (sft/self.sf_cal1)
                    t_e2_cal2_list = (sft/self.sf_cal2)
                    t_e2_cal3_list = (sft/self.sf_cal3)
                    sd_t_e2_cal1_list = sd_ratio(sft,self.sf_cal1,sd_sft,self.sd_sf_cal1)
                    sd_t_e2_cal2_list = sd_ratio(sft,self.sf_cal2,sd_sft,self.sd_sf_cal2)
                    sd_t_e2_cal3_list = sd_ratio(sft,self.sf_cal3,sd_sft,self.sd_sf_cal3)
                    if np.isfinite(Known_exp_time2[i]): #if the exposure time 2 is known write nan
                        t_e2_cal2_list = [np.nan]
                        sd_t_e2_cal2_list = [np.nan]
                    t_e2_cal1.append(t_e2_cal1_list)
                    t_e2_cal2.append(t_e2_cal2_list)
                    t_e2_cal3.append(t_e2_cal3_list)
                    sd_t_e2_cal1.append(sd_t_e2_cal1_list)
                    sd_t_e2_cal2.append(sd_t_e2_cal2_list)
                    sd_t_e2_cal3.append(sd_t_e2_cal3_list)

                if model_idx[i]>4: #Three exposures
                    sft = self.Par_fit[i][6]
                    sd_sft = self.Par_fit_err[i][6]
                    t_e3_cal1_list = (sft/self.sf_cal1)
                    t_e3_cal2_list = (sft/self.sf_cal2)
                    t_e3_cal3_list = (sft/self.sf_cal3)
                    sd_t_e3_cal1_list = sd_ratio(sft,self.sf_cal1,sd_sft,self.sd_sf_cal1)
                    sd_t_e3_cal2_list = sd_ratio(sft,self.sf_cal2,sd_sft,self.sd_sf_cal2)
                    sd_t_e3_cal3_list = sd_ratio(sft,self.sf_cal3,sd_sft,self.sd_sf_cal3)
                    if np.isfinite(Known_exp_time3[i]): #if the exposure time 3 is known write nan
                        t_e3_cal3_list = [np.nan]
                        sd_t_e3_cal3_list = [np.nan]
                    t_e3_cal1.append(t_e3_cal1_list)
                    t_e3_cal2.append(t_e3_cal2_list)
                    t_e3_cal3.append(t_e3_cal3_list)

        self.t_e1_cal1 = [t_e1_cal1,sd_t_e1_cal1]
        self.t_e1_cal2 = [t_e1_cal2,sd_t_e1_cal2]
        self.t_e1_cal3 = [t_e1_cal3,sd_t_e1_cal3]
        
        self.t_e2_cal1 = [t_e2_cal1,sd_t_e2_cal1]
        self.t_e2_cal2 = [t_e2_cal2,sd_t_e2_cal2]
        self.t_e2_cal3 = [t_e2_cal3,sd_t_e2_cal3]

        self.t_e3_cal1 = [t_e3_cal1,sd_t_e3_cal1]
        self.t_e3_cal2 = [t_e3_cal2,sd_t_e3_cal2]
        self.t_e3_cal3 = [t_e3_cal3,sd_t_e3_cal3]
        
        
    def xp_depth(self, material,sheet_name,model_idx):
       
        if material == 'Q':
            def function_EXP1(data,i, *q):# not all parameters are used here, r is shared
                    return self.fun_unp(data,i,1,*q)
            def function_EXP2(data,i, *q):# not all parameters are used here, r is shared
                    return self.fun_unp(data,i,3,*q)
            def function_EXP3(data,i, *q):# not all parameters are used here, r is shared
                    return self.fun_unp(data,i,5,*q)
            def n_p(i,parameter):
                    #return np.exp(-1) 
                    return 0.5

        if material == 'F':
            def function_EXP1(data,i, *q):# not all parameters are used here, r is shared
                    return self.fun_unp(data,i,1,*q)
            def function_EXP2(data,i, *q):# not all parameters are used here, r is shared
                    return self.fun_unp(data,i,3,*q)
            def function_EXP3(data,i, *q):# not all parameters are used here, r is shared
                    return self.fun_unp(data,i,5,*q)
            def n_p(i,parameter):
                    r = parameter[i][0]
                    #return np.power((1/r),(1/(r-1))) 
                    return  0.5
            
            
        print('Fitting parameters: ' + str(self.popt))
        self.xp = list()
        self.xp_upper = list()
        self.xp_lower = list()
        self.xp2 = list()
        self.xp2_upper = list()
        self.xp2_lower = list()
        self.xp3 = list()
        self.xp3_upper = list()
        self.xp3_lower = list()
        for i in range(len(sheet_name)):
                self.poptCI = unc.correlated_values(self.popt, self.pcov)

                xmax = max(self.xall[i])+0.5
                xi = np.linspace(0,xmax,1000)
                
                yi = function_EXP1(xi,i,*self.popt) # 1 exposure
                py = function_EXP1(xi,i,*self.poptCI)
                nom = unp.nominal_values(py)
                std = unp.std_devs(py)
                
                if model_idx[i]>2: #find xp_2
                    yi2 = function_EXP2(xi,i,*self.popt)
                    py2 = function_EXP2(xi,i,*self.poptCI) #2 exposures
                    nom2 = unp.nominal_values(py2)
                    std2 = unp.std_devs(py2)
                    K1 = 1-self.Par_fit[i][3] #burial plateau #1
                    
                if model_idx[i]>4: #find xp_3
                    yi3 = function_EXP3(xi,i,*self.popt)
                    py3 = function_EXP3(xi,i,*self.poptCI) #3 exposures 
                    nom3 = unp.nominal_values(py3)
                    std3 = unp.std_devs(py3)
                    K2 = 1-self.Par_fit[i][5] #burial plateau #2
                
                # find index of value where yi=n_p            
                idx = np.argmin(np.abs(n_p(i,self.Par_fit)-yi))
                idx_upper = np.argmin(np.abs(n_p(i,self.Par_fit)-(nom - 1.96 * std)))
                idx_lower = np.argmin(np.abs(n_p(i,self.Par_fit)-(nom + 1.96 * std)))
                # find the sample inflection point
                xpi = list([xi[idx]])
                xpi_upper = list([xi[idx_upper]])
                xpi_lower = list([xi[idx_lower]])
                self.xp.append(xpi)
                self.xp_upper.append(xpi_upper)
                self.xp_lower.append(xpi_lower)
                
                if model_idx[i]>2: #find xp_2
                    # find index of value where yi=n_p            
                    idx2 = np.argmin(np.abs(n_p(i,self.Par_fit)/K1-yi2))
                    idx_upper2 = np.argmin(np.abs(n_p(i,self.Par_fit)/K1-(nom2 - 1.96 * std)))
                    idx_lower2 = np.argmin(np.abs(n_p(i,self.Par_fit)/K1-(nom2 + 1.96 * std)))
                    # find the sample inflection point
                    xpi2 = list([xi[idx2]])
                    xpi_upper2 = list([xi[idx_upper2]])
                    xpi_lower2 = list([xi[idx_lower2]])
                if model_idx[i]<3:
                    xpi2 = list([np.nan])
                    xpi_upper2 = list([np.nan])
                    xpi_lower2 = list([np.nan])
                self.xp2.append(xpi2)
                self.xp2_upper.append(xpi_upper2)
                self.xp2_lower.append(xpi_lower2)
                    
                if model_idx[i]>4: #find xp_3
                    # find index of value where yi=n_p            
                    idx3 = np.argmin(np.abs(n_p(i,self.Par_fit)/K2-yi3))
                    idx_upper3 = np.argmin(np.abs(n_p(i,self.Par_fit)/K2-(nom3 - 1.96 * std)))
                    idx_lower3 = np.argmin(np.abs(n_p(i,self.Par_fit)/K2-(nom3 + 1.96 * std)))
                    # find the sample inflection point
                    xpi3 = list([xi[idx3]])
                    xpi_upper3 = list([xi[idx_upper3]])
                    xpi_lower3 = list([xi[idx_lower3]])
                if model_idx[i]<5:
                    xpi3 = list([np.nan])
                    xpi_upper3 = list([np.nan])
                    xpi_lower3 = list([np.nan])
                self.xp3.append(xpi3)
                self.xp3_upper.append(xpi_upper3)
                self.xp3_lower.append(xpi_lower3)

                
        return self.xp
    
    
    
        
        
    def confidence_bands_array(self,logID, model_idx,PlotPredictionBands,PlotPrev,Site_idx,Thickness):
        
        plotly_content_fit = []

        for i in range(len(self.xall)):
            event = model_idx[i]
            x_temp = np.array(self.xall[i])
            y_temp = np.array(self.yall[i])


            # calculate parameter confidence interval
            self.poptCI = unc.correlated_values(self.popt, self.pcov)

            # plot data
            #plt.scatter(self.x, self.n, s=3, label='Data')

            # calculate regression confidence interval
            xmax = max(self.xall[i])+0.5
            px = np.linspace(0, xmax, 100)
            py = self.fun_unp(px,i,model_idx[i],*self.poptCI)
            nom = unp.nominal_values(py)
            std = unp.std_devs(py)
            
            #Confidence intervals for previous profiles
            if event >=2:
                py_prev1  = self.fun_unp(px,i,1,*self.poptCI)
                nom_prev1 = unp.nominal_values(py_prev1)
                std_prev1 = unp.std_devs(py_prev1)
            if event >=3:
                py_prev2  = self.fun_unp(px,i,2,*self.poptCI)
                nom_prev2 = unp.nominal_values(py_prev2)
                std_prev2 = unp.std_devs(py_prev2)
            if event >=4:
                py_prev3  = self.fun_unp(px,i,3,*self.poptCI)
                nom_prev3 = unp.nominal_values(py_prev3)
                std_prev3 = unp.std_devs(py_prev3)

            def predband(p_used, i, x, xd, yd, p, func, conf=0.95):
                # x = requested points
                # xd = x data
                # yd = y data
                # p = parameters
                # func = function name
                alpha = 1.0 - conf    # significance
                N = np.size(xd)          # data sample size
                var_n = p_used #len(p)  # number of parameters
                # Quantile of Student's t distribution for p=(1-alpha/2)
                q = stats.t.ppf(1.0 - alpha / 2.0, N - var_n)
                # Stdev of an individual measurement
                se = np.sqrt(1. / (N - var_n) * \
                         np.sum((yd - func(xd,i,model_idx[i], *p)) ** 2))
                # Auxiliary definitions
                sx = (x - xd.mean()) ** 2
                sxd = np.sum((xd - xd.mean()) ** 2)
                # Predicted values (best-fit model)
                yp = func(x,i,model_idx[i], *p)
                # Prediction band
                dy = q * se * np.sqrt(1.0+ (1.0/N) + (sx/sxd))
                # Upper & lower prediction bands.
                lpb, upb = yp - dy, yp + dy
                return lpb, upb

            lpb, upb = predband(self.num_par[i],i, px, x_temp, y_temp, self.popt, self.fun_unp, conf=0.95)

            # plot the regression
            if Site_idx[i] == 'End':
                xfit = Thickness[i]-px
                xdata =Thickness[i]-self.xall[i]
                x_wellBleach = Thickness[i]-self.x_WB[i]          
            if Site_idx[i] == 'Front':
                xfit = px
                xdata =self.xall[i]
                x_wellBleach = self.x_WB[i]
                
                
            ### Start Matplotlib ###
            
            # fig, (ax2) = plt.subplots()
           
            # plt.plot(xfit, nom, c='black', label='fit')
            # plt.errorbar(xdata, self.yall[i], self.errall[i], ls='',
            #              marker = 'o', c = 'black',
            #              markerfacecolor  = 'white', zorder = 100)

            # # uncertainty lines (95% confidence)
            # plt.plot(xfit, nom - 1.96 * std, c='orange',\
            #      label='95% Confidence Region')
            # plt.plot(xfit, nom + 1.96 * std, c='orange')
            
            ### End Matplotlib ###
            
            
            ### For Plotly ###
            
            if len(plotly_content_fit) == 0:
                
                plotly_content_fit.append(go.Scatter(
                                                     x= xfit,
                                                     y= nom,
                                                     line=dict(color='black'),
                                                     mode='lines',
                                                     showlegend = True,
                                                     legendgroup="Fit",
                                                     name = 'Fitted Model'))
                
                plotly_content_fit.append(go.Scatter(
                                                     x= xfit,
                                                     y= nom - 1.96 * std,
                                                     line=dict(color='orange'),
                                                     mode='lines',
                                                     showlegend = True,
                                                     legendgroup="FIT+-",
                                                     name = '95% Confidence Region'))
                
                plotly_content_fit.append(go.Scatter(
                                                     x= xfit,
                                                     y= nom + 1.96 * std,
                                                     line=dict(color='orange'),
                                                     mode='lines',
                                                     showlegend = False,
                                                     legendgroup="FIT+-",
                                                     name = '95% Confidence Region'))
                
                plotly_content_fit.append(go.Scatter(
                                               x=xdata,
                                               y=self.yall[i],
                                               error_y=dict(
                                                    type='data', # value of error bar given in data coordinates
                                                    array=self.errall[i],
                                                    visible=True),
                                               line=dict(color='black'),
                                               mode='markers',
                                               marker_symbol='circle',
                                               marker_color = 'black',
                                               showlegend = True,
                                               legendgroup="Data",
                                               name = 'Data'))
                
            else:
                
            
                plotly_content_fit.append(go.Scatter(
                                                     x= xfit,
                                                     y= nom,
                                                     line=dict(color='black'),
                                                     mode='lines',
                                                     showlegend = False,
                                                     legendgroup="Fit",))
                
                plotly_content_fit.append(go.Scatter(
                                                     x= xfit,
                                                     y= nom - 1.96 * std,
                                                     line=dict(color='orange'),
                                                     mode='lines',
                                                     showlegend = False,
                                                     legendgroup="FIT+-",))
                
                plotly_content_fit.append(go.Scatter(
                                                     x= xfit,
                                                     y= nom + 1.96 * std,
                                                     line=dict(color='orange'),
                                                     mode='lines',
                                                     showlegend = False,
                                                     legendgroup="FIT+-"))
                
                plotly_content_fit.append(go.Scatter(
                                               x=xdata,
                                               y=self.yall[i],
                                               error_y=dict(
                                                    type='data', # value of error bar given in data coordinates
                                                    array=self.errall[i],
                                                    visible=True),
                                               line=dict(color='black'),
                                               mode='markers',
                                               marker_symbol='circle',
                                               marker_color = 'black',
                                               showlegend = False,
                                               legendgroup="Data"))
                
            layout = go.Layout(paper_bgcolor='rgba(0,0,0,0)',
                               plot_bgcolor='rgba(0,0,0,0)',
                               title="Fitted Curves",
                               xaxis_title="Depth [mm]",
                               yaxis_title="Normalised luminescence signal",
                               height=500,
                               width=500)
            
            self.plotly_fig_fit = go.Figure(data=plotly_content_fit,
                                            layout=layout)
            
            ### End Plotly ###

            # if PlotPredictionBands =='y':
            #     # prediction band (95% confidence)
            #     plt.plot(xfit, lpb, 'k--',label='95% Prediction Band')
            #     plt.plot(xfit, upb, 'k--')
            
            # plt.ylabel('y')
            # plt.xlabel('x')
            # #plt.legend(loc='best')
            
            # if PlotPrev =='yes':
            #     if event >=2:
            #         plt.plot(xfit, nom_prev1, c='red', label='fit')
            #     if event >=3:
            #         plt.plot(xfit, nom_prev2, c='blue', label='fit')    
            #     if event >=4:
            #         plt.plot(xfit, nom_prev3, c='green', label='fit')

            # #Plot well bleached depth
            # if event==2:
            #     plt.plot((x_wellBleach[0], x_wellBleach[0]), (nom[0]-0.2,nom[0]+0.2), 'k-')
            # if event>2:
            #     plt.plot((x_wellBleach[0], x_wellBleach[0]), (nom_prev2[0]-0.2,nom_prev2[0]+0.2), 'k-')
            # if event>=3:
            #     plt.plot((x_wellBleach[1], x_wellBleach[1]), (nom_prev2[0]-0.2,nom_prev2[0]+0.2), 'k-')
            # if event>=4:
            #     plt.plot((x_wellBleach[2], x_wellBleach[2]), (nom[0]-0.2,nom[0]+0.2), 'k-')
            # if event>=5:
            #     plt.plot((x_wellBleach[3], x_wellBleach[3]), (nom_prev2[0]-0.2,nom_prev2[0]+0.2), 'k-')
                       
            # if event>= 2:
            #     print("F*tb 1 =",self.popt[3])
            #     print('Well bleached depths sample ' + str(i) + ':' + str(self.x_WB[i]))
            # if event>= 3:
            #     print("sfte 2 =",self.popt[4])
            # if event>= 4:
            #     print("F*tb 2 =",self.popt[5])
            # if event>= 5:
            #     print("sfte 3 =",self.popt[6])


        # if logID == 'yes':
        #         ax2.set_yscale('log')
        
        # # label the axes of axes[0]
        # ax2.set_xlabel("Depth [mm]")
        # ax2.set_ylabel("Normalised luminescence signal")
        # ax2.set_title("Non-linear Least square fit")

        # # save and show figure
        # plt.savefig('regression.png')
        # plt.show()
        # print('xp: ' + str(self.xp))
        # #print('xp lower: ' + str(self.xp_lower))
        # #print('xp upper: ' + str(self.xp_upper))
       

    
    def ERC(self,KNOWN_EXP_TIME1,KNOWN_EXP_TIME2,KNOWN_EXP_TIME3,sheet_name,model_idx):
    
        # Create ERC from profiles wiht known exposure time 1        
        xp_all = np.array([item for sublist in self.xp for item in sublist])
        xp_lower_all = np.array([item for sublist in self.xp_lower for item in sublist])
        xp_upper_all = np.array([item for sublist in self.xp_upper for item in sublist])
        xp_err_all = (xp_all-xp_lower_all)/1.96  #to give one standard deviation
        
        xp2_all = np.array([item for sublist in self.xp for item in sublist])
        xp2_lower_all = np.array([item for sublist in self.xp2_lower for item in sublist])
        xp2_upper_all = np.array([item for sublist in self.xp2_upper for item in sublist])
        xp2_err_all = (xp2_all-xp2_lower_all)/1.96 #to give one standard deviation
        
        xp3_all = np.array([item for sublist in self.xp3 for item in sublist])
        xp3_lower_all = np.array([item for sublist in self.xp3_lower for item in sublist])
        xp3_upper_all = np.array([item for sublist in self.xp3_upper for item in sublist])
        xp3_err_all = (xp3_all-xp3_lower_all)/1.96  #to give one standard deviation

        index = np.isfinite(KNOWN_EXP_TIME1)
        Time = KNOWN_EXP_TIME1[index]
        know_Age1_log = [log10(x) for x in Time]
        
        index2 = np.isfinite(KNOWN_EXP_TIME2)
        Time2 = KNOWN_EXP_TIME2[index2]
        know_Age2_log = [log10(x) for x in Time2]
        
        index3 = np.isfinite(KNOWN_EXP_TIME3)
        Time3 = KNOWN_EXP_TIME3[index3]
        know_Age3_log = [log10(x) for x in Time3]
        
        def ERCfun(logt,*p): #function to fit ERC
            return p[0]*logt + p[1]
        
        def ERC_solve(XP,*p): #solve unknown t from ERC
            return 10**((XP-p[1])/p[0])
        
        xxpp = [xp_all[index],xp2_all[index2],xp3_all[index3]]
        xxpp_err = [xp_err_all[index],xp2_err_all[index2],xp3_err_all[index3]]
        KNOWN_AGE_LOG = [know_Age1_log,know_Age2_log,know_Age3_log]
        TIME_ERC = []
        
        figure_plotly = []
        
        for k in range(0,3): 
            xp = xxpp[k]
            xp_err = xxpp_err[k]
            know_Age_log = KNOWN_AGE_LOG[k]
            
        
            if len(xp)>1: #If more than one known exposure profile, do ERC
                self.poptERC, self.pcovERC = curve_fit(
                        f=ERCfun,       # model function
                        xdata=know_Age_log,   # x data
                        ydata=xp,   # y data
                        p0=np.array([1, 1]),      # initial value of the parameters
                        sigma=xp_err,   # 
                        absolute_sigma=True,
                        check_finite=True,
                        method='trf',
                        jac=None,
                        full_output=False
                        )

                #Estimate exposure times from ERC
                t_ERC = []
                t_ERC_all = []
                for i in range(0,len(sheet_name)): #different profiles
                    t_erc = [ERC_solve(xp_all[i],*self.poptERC)]  #estimated exp. times from ERC
                    t_ERC_all.append(list(t_erc))
                    if np.isfinite(KNOWN_EXP_TIME1[i]): #if the exposure time is known write nan
                        t_erc = [np.nan]
                    t_ERC.append(list(t_erc))
                self.t_ERC = t_ERC
                TIME_ERC.append(t_ERC)

                # calculate regression confidence interval
                self.poptCI_ERC = unc.correlated_values(self.poptERC, self.pcovERC)
                tmax = np.array(max(t_ERC_all))
                tmin = np.array(min(t_ERC_all))
                tt = np.linspace(log10(tmin/2), log10(tmax*2), 100)
                ERC_fit = ERCfun(tt,*self.poptCI_ERC)
                nom = unp.nominal_values(ERC_fit)
                std = unp.std_devs(ERC_fit)
            
  #####GOROS: This figure was moved to here and included in the loop of different exposure events##########
                # Create the ERC figure
                
                ### For Matplot ###
                
                # fig, ax1 = plt.subplots(figsize=(8, 8))
                
                # ax1.plot(10**tt, nom - 1.96 * std, c='orange',\
                #          label='95% Confidence Region')
                # ax1.plot(10**tt, nom + 1.96 * std, c='orange')
                # ax1.plot(10**tt, nom, c='red')
                # ax1.errorbar(Time, xp, xp_err, ls='',
                #                  marker = 'o', c = 'black',
                #                  markerfacecolor  = 'white', zorder = 100)
    
                # for i in range(0,len(sheet_name)): #different profiles
                #         if np.isfinite(self.t_ERC[i]):
                #             x_points = np.array([tt[0], self.t_ERC[i][0], self.t_ERC[i][0]])
                #             y_points = np.array([xp_all[i], xp_all[i], 0])
                #             ax1.plot(x_points, y_points,'--')
    
    
                # ax1.set_xlabel('Exposure time')
                # ax1.set_ylabel('inflection point [mm]')
                # ax1.set_xscale('log')        
                # ax1.set_xlim([10**tt[0]/5,10**tt[-1]*5])
                # ax1.set_ylim(bottom=0)
                # plt.title('ERC curve from exposure no. {}'.format(k+1))
                
                
                ### End Matplotlib ###
                
                
                ### For Plotly ###
                
                plotly_content_ERC = []
                
                plotly_content_ERC.append(go.Scatter(
                                                     x= 10**tt,
                                                     y= nom + 1.96 * std,
                                                     line=dict(color='orange'),
                                                     mode='lines',
                                                     showlegend = False,
                                                     name='95% Confidence Region'))
                
                plotly_content_ERC.append(go.Scatter(
                                                     x= 10**tt,
                                                     y= nom - 1.96 * std,
                                                     line=dict(color='orange'),
                                                     mode='lines',
                                                     showlegend = False))
                
                plotly_content_ERC.append(go.Scatter(
                                                     x= 10**tt,
                                                     y= nom,
                                                     line=dict(color='red'),
                                                     mode='lines',
                                                     showlegend = False))
                
                plotly_content_ERC.append(go.Scatter(
                                               x=Time,
                                               y=xp,
                                               error_y=dict(
                                                    type='data', 
                                                    array=xp_err,
                                                    visible=True),
                                               line=dict(color='black'),
                                               mode='markers',
                                               marker_symbol='circle',
                                               marker_color = 'black',
                                               showlegend = True,
                                               name = 'Data'))

                for i, name in enumerate(sheet_name): #different profiles
                    if np.isfinite(self.t_ERC[i]):
                            
                        x_points = [tt[0], self.t_ERC[i][0], self.t_ERC[i][0]]
                        y_points = [xp_all[i], xp_all[i], 0]
                        
                        plotly_content_ERC.append(go.Scatter(
                                                             x= x_points,
                                                             y= y_points,
                                                             mode='lines',
                                                             line=dict(dash='dot'),
                                                             showlegend = True,
                                                             name=name))
                                    
                layout = go.Layout(paper_bgcolor='rgba(0,0,0,0)',
                                   plot_bgcolor='rgba(0,0,0,0)',
                                   title='ERC curve from exposure no. {}'.format(k+1),
                                   xaxis_title='Exposure time',
                                   yaxis_title='Inflection point [mm]',
                                   height=500,
                                   width=500)
                
                plotly_fig_ERC = go.Figure(data=plotly_content_ERC,
                                                layout=layout)
                
                # plotly_fig_ERC.update_yaxes(range=[0, ax1.get_ylim()])
                # self.plotly_fig_ERC.update_xaxes(range=[10**tt[0]/5, 10**tt[-1]*5])
                plotly_fig_ERC.update_xaxes(type="log")
                
                figure_plotly.append(plotly_fig_ERC)
                
            self.plotly_fig_ERC = figure_plotly


                ### End Plotly ###
                
        self.TIME_ERC = TIME_ERC

        
#Make Gausiam distribution of all xp,
#fit to new values, estimate t_exp, 
#calc. sd from all cycles- use original fit as best fit       
        #Create Gausian distribution of xp:
        XP_all = []
        XP2_all = []
        XP3_all = []

        for i in range(0,len(sheet_name)): #different profiles
            #For exposure #1
            mu, sigma = xp_all[i], xp_err_all[i] # mean and standard deviation
            xp_s = np.random.normal(mu, sigma, 1000)
            XP_all.append(list(xp_s))
            #For exposure #2
            mu2, sigma2 = xp2_all[i], xp2_err_all[i] # mean and standard deviation
            xp2_s = np.random.normal(mu2, sigma2, 1000)
            XP2_all.append(list(xp2_s))
            #For exposure #3
            mu3, sigma3 = xp3_all[i], xp3_err_all[i] # mean and standard deviation
            xp3_s = np.random.normal(mu3, sigma3, 1000)
            XP3_all.append(list(xp3_s))
        
        def Extract(lst,j):
            return [item[j] for item in lst]               
                
        XXPP = [XP_all,XP2_all,XP3_all]
        KNOWN_AGE_LOG = [know_Age1_log,know_Age2_log,know_Age3_log]
        INDEX = [index,index2,index3] 
        SD_t_ERC = []
        SE_t_ERC = []
        MEAN_t_ERC = []
        for k in range(0,3): #for exopsure 1, 2, 3
            XP_ALL = XXPP[k]
            know_Age_log = KNOWN_AGE_LOG[k]
        
            if len(know_Age_log)>1: # Only run ERC fit if more than one exp. is known
                par_ERC_moteCarlo = []

                for j in range(0,1000): #fit each of the MC generated data sets of xp and time
                    XP = np.array(Extract(XP_ALL,j))
                    poptERC, pcovERC = curve_fit(
                        f=ERCfun,       # model function
                        xdata=know_Age_log,   # x data
                        ydata=XP[INDEX[k]],   # y data
                        p0=np.array([1, 1]),      # initial value of the parameters
                        check_finite=True,
                        method='trf',
                        jac=None,
                        full_output=False
                        )
                    par_ERC_moteCarlo.append(list(poptERC))

                #Estimate exposure times from ERC
                sd_t_ERC = []
                mean_t_ERC = []
                t_ERC_monteCarlo = []
                no_it = len(XP_all[0])
                for i in range(0,len(sheet_name)): #different profiles
                    t_ERC_Gaus = np.array([])
                    for j in range(0,len(XP_all[0])): #monte carlo runs
                            t_erc = [ERC_solve(XP_all[i][j],*par_ERC_moteCarlo[j])]  #estimated exp. times from ERC            
                            if np.isfinite(KNOWN_EXP_TIME1[i]): #if the exposure time is known write nan
                                t_erc = [np.nan]
                            t_ERC_Gaus = np.append(t_ERC_Gaus,t_erc)
                    t_ERC_monteCarlo.append(list(t_ERC_Gaus))
                    sd_t_erc = [np.std(t_ERC_Gaus)]
                    mean_t_erc = [np.mean(t_ERC_Gaus)]
                    sd_t_ERC.append(list(sd_t_erc))
                    mean_t_ERC.append(list(mean_t_erc))


                for i in range(0,len(sheet_name)): #different profiles
                        if np.isfinite(self.t_ERC[i]):
                                plt.figure(figsize=(10, 7))
                                fig, ax1 = plt.subplots(figsize=(8, 8))
                                ax1.set_ylabel('Frequency')
                                ax1.set_xlabel('Exposure time')
                                #ax1.set_ylim([0,0.6])
                                ax1.set_xlim([0,max(t_ERC_monteCarlo[i])*1.1])
                                ax1.hist(t_ERC_monteCarlo[i], bins=20, edgecolor='black', weights=np.ones_like(t_ERC_monteCarlo[i]) / len(t_ERC_monteCarlo[i]))
            
                MEAN_t_ERC.append(mean_t_ERC)
                SD_t_ERC.append(sd_t_ERC)
        self.SD_t_ERC = SD_t_ERC
        print(self.TIME_ERC)
        print(self.SD_t_ERC)
        print(self.t_e1_cal1)
        

   ############ TO DO: #################

   ##### All profiles in one table including:
    
   # Model used in fitting: given as model_idx
   # r (order)          : saved as self.Par_fit[i][0] with i being sample no.
   # u (attenuation)    : saved as self.Par_fit[i][1] with i being sample no.
   # sfte1              : saved as self.Par_fit[i][2] with i being sample no.
   # F*tb1 = DR/Dc*tb1  : saved as self.Par_fit[i][3] with i being sample no.
   # sfte2              : saved as self.Par_fit[i][4] with i being sample no.
   # F*tb2 = DR/Dc*tb2  : saved as self.Par_fit[i][5] with i being sample no.
   # sfte3              : will be saved as self.Par_fit[i][6] with i being sample no.
   # F*tb3 = DR/Dc*tb3  : will be saved as self.Par_fit[i][7] with i being sample no.
   # Exposure time from ERC with sd of values : saved as self.TIME_ERC and self.SD_t_ERC
   # Exposure time from single/multiple calibration and sd of the values: saved as self.t_e1_cal1, self.t_e1_cal2, self.t_e2_cal1, self.t_e2_cal2
   # Well bleached depth (aliqouts to use in burial age estimation): saved as self.x_WB
    
   ####### Graphs and data:########
   #     Profiles: done (change colors)
   #     Fits: done (change colors)
   #     ERC: done  (change colors)
   #     guess graph (to help in choosing P0 values)
    
    

    def fitting_array(self, sheet_name, material,guess,logID, weight,model_idx,WBtolerance,PlotPredictionBands,PlotPrev,Site_idx,Thickness,Known_exp_time1,Known_exp_time2,Known_exp_time3,residual): 
        self.read_data_array(sheet_name,weight)
        self.define_model(material,guess,sheet_name,model_idx,residual)
        self.multiple_fiting_array(sheet_name,model_idx)
        #self.plot_figure_guess(xmax, sheet_name,logID,Site_idx,Thickness)
        self.run_model_array()
        self.Parameterrresults(sheet_name,model_idx)
        self.xp_depth(material,sheet_name,model_idx)
        self.Well_bleached_depth(WBtolerance,model_idx)
        self.confidence_bands_array(logID,model_idx,PlotPredictionBands,PlotPrev,Site_idx,Thickness)
        self.SingleCal(sheet_name,model_idx,Known_exp_time1,Known_exp_time2, Known_exp_time3)
        self.ERC(Known_exp_time1,Known_exp_time2,Known_exp_time3,sheet_name,model_idx)
        
