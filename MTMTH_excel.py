from typing import Text
from ipywidgets.widgets.interaction import interactive
from ipywidgets.widgets.widget_float import FloatText
from ipywidgets.widgets.widget_selection import Dropdown
from numpy.core.fromnumeric import size
from numpy.core.function_base import linspace
import pandas as pd
import matplotlib.pyplot as plt
import os
import scipy.signal as spsign

import matplotlib.colors as colors
import numpy as np
import scipy.integrate as sp    #trapz, Simps, cumtrapz, romb
import ipywidgets as widgets

from IPython.display import display
from scipy import sparse
from scipy.sparse.linalg import spsolve
from random import randint

class excel_MTMTH_Calc():




    def __init__(self, *args,**kwargs):
        directory = kwargs.get('Directory', r'C:\Users\bjorngso\OneDrive - Universitetet i Oslo\01 Results\Testing\MTMTH\GC\Old_method')
     
        together = list()
        for root, dirs, files in os.walk(directory, topdown=False):
            for name in files:
                joint = ('{}'.format(name), os.path.join(root, name))
                together.append(joint)        
        self.direct= widgets.Dropdown(options=together, description='Eksperiment:', disabled=False,layout=widgets.Layout(width='30%'),style = {'description_width': 'initial'})
        self.help()
        display(self.direct)

    def directory(self):
        return print(self.directory)

    def help(self):
        return print("""
        Here are the fucntions that are included in the class. This is meant to be used together with Jupyter LAB:

        obj.Quantify()                                    Quantify and integrate the results from GC

        obj.plot(form="total over time")                 Plot total Yield produced over time
        obj.plot(form="")                                Plot  Yield produced vs time

               
        

        """)


    def Quantify(self):
        file = self.direct.value
        try:
            df = pd.read_excel(io=file,sheet_name='Sheet2',header= 1, nrows=10 ,usecols='B:X',dtype=np.float64)
            first_row_with_all_NaN = df[df.isnull().all(axis=1) == True].index.tolist()[0]
            df = df.loc[0:first_row_with_all_NaN-1]
            sheet_type = 1
        except:
            df = pd.read_excel(io=file,sheet_name='Sheet1',header= 0, nrows=10 ,usecols='B:X',dtype=np.float64)
            sheet_type = 2

        time = df['Time (min)'].to_numpy()     
        Methane = df['Methane'].to_numpy()
        Ethylene = df['Ethylene'].to_numpy()
        Ethane = df['Ethane'].to_numpy()
        Propene = df['Propene'].to_numpy()

        if sheet_type == 1:
            C8 = df['C8'].to_numpy()
            CO = df['CO'].to_numpy()
            H2O = df['H2O'].to_numpy()
        if sheet_type == 2:
            DME = df['DME'].to_numpy()

        C4 = np.array([])
        C5 = np.array([])
        C6 = np.array([])
        C7 = np.array([])

        if sheet_type == 1:
            for index, row in  df.iterrows():
                C4 = np.append(C4, (row['1-Butene & 2-trans-Butene']+row['Isobutene']+row['2-cis-Butene']))
                C5 = np.append(C5, (row['C5_1']+row['C5_2']+row['C5_3']+row['C5_4']))
                C6 = np.append(C6, (row['C6_1']+row['C6_2']+row['Heksene']+row['C6_4']))
                C7 = np.append(C7, (row['C7_1']+row['C7_2']+row['C7_3']+row['C7_4']))

        if sheet_type == 2:
            for index, row in  df.iterrows():
                C4 = np.append(C4, (row['1-Butene & 2-trans-Butene']+row['Isobutene']+row['2-cis-Butene']))
                C5 = np.append(C5, (row['C5_1']+row['C5_2']+row['C5_3']))
                C6 = np.append(C6, (row['C6_1']+row['C6_2']+row['C6_3']))
                C7 = np.append(C7, (row['C7_1']+row['C7_2']+row['C7_3']))
            

    ########################################################################################################################################################
    ########################################################################################################################################################
    ########################################################################################################################################################
    ########################################################################################################################################################

        if sheet_type == 1:
            fig, (ax1,ax3,ax5) = plt.subplots(1,3,figsize=(14,6))      
            fig.suptitle(file[79:-5], fontsize = 18)
            
            print("POPYLENE EXPERIMENT")
            ax1.plot(time,Methane, label='Methane',marker='o')
            ax1.plot(time, CO, label='CO',marker='o')
            ax1.plot(time,Ethane, label='Ethane',marker='o')
            ax1.plot(time,Ethylene, label='Ethylene',marker='o')
            ax1.plot(time,H2O, label='H2O',marker='o')
            ax1.plot(time,C4, label='C4',marker='o')
            ax1.plot(time,C5, label='C5',marker='o')
            ax1.plot(time,C6, label='C6',marker='o')
            ax1.plot(time,C7, label='C7',marker='o')
            ax1.plot(time,C8, label='C8',marker='o')
            ax1.legend()
            ax1.set_xlabel('Time (min)')
            ax1.set_ylabel('GC Area (counts)')
            ax1.set_title('All compounds')
            ax1.grid(True)

            ax4 = ax1.twinx()
            ax4.plot(time,Propene, label='Propene',color='darkgreen',marker='s', linestyle='dotted')
            ax4.set_ylabel('Propene', color='darkgreen')
            ax4.spines['right'].set_color('darkgreen')
            ax4.tick_params(axis='y', colors='darkgreen')






            #Carbons of interest
            list_of_comp = [C4/4,C5/5,C7/7,C8/8]
            list_of_name = ["C4","C5","C7","C8"]

            cmap = plt.get_cmap('Set2')
            list_of_colors = cmap(np.linspace(0,1,len(list_of_comp)))
            cmap = plt.get_cmap('Pastel2')
            list_of_colors_fill = cmap(np.linspace(0,1,len(list_of_comp)))
            
            i = 0
            time = np.insert(time,0,time[0]-time[0])
            time = np.append(time,time[-1]+0.00000000000000000000000000000000000001)
            list_of_area = np.array([])
            
            
            
            C6_ = C6/6
            comp_new = np.insert(C6_,0,0)
            comp_new = np.append(comp_new,0)
            bkg = np.linspace(comp_new[-1],comp_new[-1],len(comp_new))
            C6_area = np.trapz(comp_new-bkg, time, 0.00001)
            
            ax5.plot(time,comp_new, label='{}'.format("C6"),color="silver")
            ax5.fill_between(time,comp_new,bkg,color="silver")
            ax5.legend()
            ax5.set_xlabel('Time (min)')
            ax5.set_ylabel('GC Area (counts)')
            ax5.set_title('Carbons of interest')
            ax5.grid(True)

            for comp in list_of_comp:
                comp_new = np.insert(comp,0,0)
                comp_new = np.append(comp_new,0)
                bkg = np.linspace(comp_new[-1],comp_new[-1],len(comp_new))
                area = np.trapz(comp_new-bkg, time, 0.00001)
                ax3.plot(time,comp_new, label='{}'.format(list_of_name[i]),color=list_of_colors[i])
                ax3.fill_between(time,comp_new,bkg,color=list_of_colors_fill[i])
                list_of_area = np.append(list_of_area, area)
                i= i+1
            list_of_area = np.insert(list_of_area,2,C6_area)  
            list_of_name = np.insert(list_of_name,2,"C6") 
            
            
            
            ax3.legend()
            ax3.set_xlabel('Time (min)')
            ax3.set_ylabel('GC Area (counts)')
            ax3.set_title('Carbons of interest')
            ax3.grid(True)
            
            


    ########################################################################################################################################################
        if sheet_type == 2:
            fig, (ax1,ax3) = plt.subplots(1,2,figsize=(10,6))      
            fig.suptitle(file[79:-5], fontsize = 18)
            print("ETHYLENE EXPERIMENT")
            
            ax1.plot(time,Methane, label='Methane',marker='o')
            ax1.plot(time,DME, label='DME',marker='o')
            ax1.plot(time,Ethane, label='Ethane',marker='o')
            ax1.plot(time,Propene, label='Propene',marker='o')
            ax1.plot(time,C4, label='C4',marker='o')
            ax1.plot(time,C5, label='C5',marker='o')
            ax1.plot(time,C6, label='C6',marker='o')
            ax1.plot(time,C7, label='C7',marker='o')
            ax1.legend()
            ax1.set_xlabel('Time (min)')
            ax1.set_ylabel('GC Area (counts)')
            ax1.set_title('All compounds')
            ax1.grid(True)

            ax4 = ax1.twinx()
            ax4.plot(time,Ethylene, label='Ethylene',color='darkgreen',marker='s', linestyle='dotted')
            ax4.set_ylabel('Ethylene', color='darkgreen')
            ax4.spines['right'].set_color('darkgreen')
            ax4.tick_params(axis='y', colors='darkgreen')






            #Carbons of interest
            list_of_comp = [Propene/3,C4/4,C5/5,C6/6,C7/7]
            list_of_name = ["C3","C4","C5","C6","C7"]

            cmap = plt.get_cmap('Set2')
            list_of_colors = cmap(np.linspace(0,1,len(list_of_comp)))
            cmap = plt.get_cmap('Pastel2')
            list_of_colors_fill = cmap(np.linspace(0,1,len(list_of_comp)))
            
            i = 0
            time = np.insert(time,0,time[0]-time[0])
            time = np.append(time,time[-1]+0.00000000000000000000000000000000000001)
            list_of_area = np.array([])

            for comp in list_of_comp:
                comp_new = np.insert(comp,0,0)
                comp_new = np.append(comp_new,0)
                bkg = np.linspace(0,0,len(comp_new))
                area = np.trapz(comp_new-bkg, time, 0.00001)
                ax3.plot(time,comp_new, label='{}'.format(list_of_name[i]),color=list_of_colors[i])
                ax3.fill_between(time,comp_new,bkg,color=list_of_colors_fill[i]) 
                list_of_area = np.append(list_of_area, area)
                i= i+1

            ax3.legend()
            ax3.set_xlabel('Time (min)')
            ax3.set_ylabel('GC Area (counts)')
            ax3.set_title('Carbons of interest')
            ax3.grid(True)


        print("")
        print("")
        print("")
        print("")
        list_of_cons = np.array([]) 
        for area in list_of_area:
            dry_weight = 0.086
            ppm = area/0.5529 #calib factor for method ethylene FID
            percent = ppm*(10**(-6))*100 #ppm to percentage
            mL = percent*15.0/100
            mol = (1*(mL*(10**(-3))))/(298.15*0.082057)
            Yield = mol*(10**(6))/dry_weight   #(micromol/gram)
            list_of_cons = np.append(list_of_cons, "{:.5}".format(Yield))

        dataset = pd.DataFrame({"Species":list_of_name,"Yield (micromol/gram)":list_of_cons})
        T_dataset = dataset.T
        display(T_dataset)


        fig.tight_layout()
        
        
        
    ########################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################
    ################################################################################################################################################################################################################################################################################################################
    ################################################################################################################################################################################################################################################################################################################
    ################################################################################################################################################################################################################################################################################################################
    ################################################################################################################################################################################################################################################################################################################
    ################################################################################################################################################################################################################################################################################################################
    ################################################################################################################################################################################################################################################################################################################

    def plot(self, form):
        file = self.direct.value
        fig, (ax1) = plt.subplots(1,figsize=(6,8)) 
        try:
            df = pd.read_excel(io=file,sheet_name='Sheet2',header= 1, nrows=10 ,usecols='B:X',dtype=np.float64)
            first_row_with_all_NaN = df[df.isnull().all(axis=1) == True].index.tolist()[0]
            df = df.loc[0:first_row_with_all_NaN-1]
            sheet_type = 1
        except:
            df = pd.read_excel(io=file,sheet_name='Sheet1',header= 0, nrows=10 ,usecols='B:X',dtype=np.float64)
            sheet_type = 2
        
        time = df['Time (min)'].to_numpy()     
        Methane = df['Methane'].to_numpy()
        Ethylene = df['Ethylene'].to_numpy()
        Ethane = df['Ethane'].to_numpy()
        Propene = df['Propene'].to_numpy()
        one_butene_trans = df['1-Butene & 2-trans-Butene'].to_numpy()
        Isobutene = df['Isobutene'].to_numpy()
        cis_Butene = df['2-cis-Butene'].to_numpy()
        
        if sheet_type == 1:
            C8 = df['C8'].to_numpy()
            CO = df['CO'].to_numpy()
            H2O = df['H2O'].to_numpy()
        if sheet_type == 2:
            DME = df['DME'].to_numpy()

        C4 = np.array([])
        C5 = np.array([])
        C6 = np.array([])
        C7 = np.array([])

        if sheet_type == 1:
            for index, row in  df.iterrows():
                C4 = np.append(C4, (row['1-Butene & 2-trans-Butene']+row['Isobutene']+row['2-cis-Butene']))
                C5 = np.append(C5, (row['C5_1']+row['C5_2']+row['C5_3']+row['C5_4']))
                C6 = np.append(C6, (row['C6_1']+row['C6_2']+row['Heksene']+row['C6_4']))
                C7 = np.append(C7, (row['C7_1']+row['C7_2']+row['C7_3']+row['C7_4']))

        if sheet_type == 2:
            for index, row in  df.iterrows():
                C4 = np.append(C4, (row['1-Butene & 2-trans-Butene']+row['Isobutene']+row['2-cis-Butene']))
                C5 = np.append(C5, (row['C5_1']+row['C5_2']+row['C5_3']))
                C6 = np.append(C6, (row['C6_1']+row['C6_2']+row['C6_3']))
                C7 = np.append(C7, (row['C7_1']+row['C7_2']+row['C7_3']))
            
                
                
        time = np.insert(time,0,time[0]-time[0])
        
        
        if form == "total over time":
            try:        
                list_of_comp = [C4/4,C5/5,C6/6,C7/7,C8/8]
                list_of_comp_new = []

                for entry in list_of_comp:
                    total = 0
                    result = np.array([])
                    result= np.append(result,total)
                    for element in entry:
                        total += element
                        result= np.append(result,total)
                    list_of_comp_new.append(result)

                list_of_name = ["C4","C5","C6","C7","C8"]
                list_of_symbol = ['D','p','h','*','8']
                cmap = plt.get_cmap('Set2')
                list_of_colors = cmap(np.linspace(0,1,len(list_of_name)+1))

                
                for i in range(len(list_of_comp)):
                    species = list_of_comp_new[i]
                    dry_weight = 0.086
                    ppm = species/0.5529 #calib factor for method ethylene FID
                    percent = ppm*(10**(-6))*100 #ppm to percentage
                    mL = percent*15.0/100
                    mol = (1*(mL*(10**(-3))))/(298.15*0.082057)
                    Yield = mol*(10**(6))/dry_weight   #(micromol/gram)         
                    ax1.plot(time,Yield, label=list_of_name[i],marker=list_of_symbol[i],color=list_of_colors[i+1],markersize=12,linewidth=3)  
                    
            except: 
                list_of_comp = [Propene/3,C4/4,C5/5,C6/6,C7/7]
                list_of_comp_new = []
                
                
                for entry in list_of_comp:
                    total = 0
                    result = np.array([])
                    result= np.append(result,total)
                    for element in entry:
                        total += element
                        result= np.append(result,total)
                    list_of_comp_new.append(result)
                
                list_of_name = ["C3","C4","C5","C6","C7"]
                list_of_symbol = ['^','D','p','h','*']
                cmap = plt.get_cmap('Set2')
                list_of_colors = cmap(np.linspace(0,1,len(list_of_name)+1))


                
                for i in range(len(list_of_comp)):
                    species = list_of_comp_new[i]
                    dry_weight = 0.086
                    ppm = species/0.5529 #calib factor for method ethylene FID
                    percent = ppm*(10**(-6))*100 #ppm to percentage
                    mL = percent*15.0/100
                    mol = (1*(mL*(10**(-3))))/(298.15*0.082057)
                    Yield = mol*(10**(6))/dry_weight   #(micromol/gram)         
                    ax1.plot(time,Yield, label=list_of_name[i],marker=list_of_symbol[i],color=list_of_colors[i],markersize=12,linewidth=3)  
                    
            ax1.set_ylabel(r'Total Yield [$\mu$mol/$g_{zeolite}$]',fontsize = 16)
                    
                    
        else:
            try:      
                list_of_comp = [C4/4,C5/5,C6/6,C7/7,C8/8]
                list_of_name = ["C4","C5","C6","C7","C8"]
                list_of_symbol = ['D','p','h','*','8']
                cmap = plt.get_cmap('Set2')
                list_of_colors = cmap(np.linspace(0,1,len(list_of_name)+1))

                
                for i in range(len(list_of_comp)):
                    species = np.insert(list_of_comp[i],0,0)
                    dry_weight = 0.086
                    ppm = species/0.5529 #calib factor for method ethylene FID
                    percent = ppm*(10**(-6))*100 #ppm to percentage
                    mL = percent*15.0/100
                    mol = (1*(mL*(10**(-3))))/(298.15*0.082057)
                    Yield = mol*(10**(6))/dry_weight   #(micromol/gram)
                    ax1.plot(time,Yield, label=list_of_name[i],marker=list_of_symbol[i],color=list_of_colors[i+1],markersize=12,linewidth=3)
            except: 
                list_of_comp = [Propene/3,C4/4,C5/5,C6/6,C7/7]
                list_of_name = ["C3","C4","C5","C6","C7"]
                list_of_symbol = ['^','D','p','h','*']
                cmap = plt.get_cmap('Set2')
                list_of_colors = cmap(np.linspace(0,1,len(list_of_name)+1))

                for i in range(len(list_of_comp)):
                    species = np.insert(list_of_comp[i],0,0)
                    dry_weight = 0.086
                    ppm = species/0.5529 #calib factor for method ethylene FID
                    percent = ppm*(10**(-6))*100 #ppm to percentage
                    mL = percent*15.0/100
                    mol = (1*(mL*(10**(-3))))/(298.15*0.082057)
                    Yield = mol*(10**(6))/dry_weight   #(micromol/gram)         
                    ax1.plot(time,Yield, label=list_of_name[i],marker=list_of_symbol[i],color=list_of_colors[i],markersize=12,linewidth=3)
            
            ax1.set_ylabel(r'Yield [$\mu$mol/$g_{zeolite}$]',fontsize = 16)
            
                    
        ax1.legend(prop={'size': 12})
        ax1.set_xlabel('Time [min]', fontsize = 16)

        ax1.set_title('Cu-MOR(6.5)', fontsize = 16,fontweight ='bold')
        ax1.grid(True)
        ax1.tick_params(axis='both', which='major', labelsize=12)
        
        
        
        
