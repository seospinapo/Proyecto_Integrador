#Función del modelo

#Inputs
import pandas as pd
import numpy as np

#score_modelo (int_rate,inq_last_6mths,revol_bal,open_acc_6m,total_cu_tl,avg_cur_bal,bc_open_to_buy,mort_acc,num_accts_ever_120_pd,num_tl_90g_dpd_24m,revol_bal_joint,mths_since_recent_revol_delinq,mths_since_recent_inq, mths_since_recent_bc_dlq,disbursement_method,verification_status,term,purpose,emp_length,annual_inc,addr_state,revol_util,tot_cur_bal,mths_since_rcnt_il,total_rev_hi_lim)

def score_modelo (int_rate,inq_last_6mths,revol_bal,open_acc_6m,total_cu_tl,avg_cur_bal,bc_open_to_buy,mort_acc,num_accts_ever_120_pd,num_tl_90g_dpd_24m,revol_bal_joint,mths_since_recent_revol_delinq,mths_since_recent_inq, mths_since_recent_bc_dlq,disbursement_method,verification_status,term,purpose,emp_length,annual_inc,addr_state,revol_util,tot_cur_bal,mths_since_rcnt_il,total_rev_hi_lim):
    
    if pd.isna(mths_since_recent_revol_delinq):
        v1=-0.1045
    else:
        if mths_since_recent_revol_delinq <= 17:
            v1=-0.0438
        else:
            if mths_since_recent_revol_delinq <= 33:
                v1=-0.111
            else:
                if mths_since_recent_revol_delinq <=52:
                    v1=-0.1497
                else:
                    v1=-0.1466
    
    if pd.isna(mths_since_recent_inq):
        v2=-0.3153
    else:
        if mths_since_recent_inq <= 2:
            v2=0.0304
        else:
            if mths_since_recent_inq <= 5:
                v2=-0.0703
            else:
                if mths_since_recent_inq <=10:
                    v2=-0.0647
                else:
                    v2=-0.1358
    
    if pd.isna(mths_since_recent_bc_dlq):
        v3=-0.1551
    else:
        if mths_since_recent_bc_dlq <= 21:
            v3=-0.0569
        else:
            if mths_since_recent_bc_dlq <= 38:
                v3=-0.1101
            else:
                if mths_since_recent_bc_dlq <=59:
                    v3=-0.1338
                else:
                    v3=-0.0997
                    
    if disbursement_method== "Cash":
        v4=-0.159
    else:
        v4=-0.3966
    
    if verification_status== "Verified":
        v5=-0.132
    else:
        if verification_status== "Source Verified" :
            v5=-0.1117
        else:
            v5=-0.3119
    
    if term == "36 months":
        v6=-0.6129
    else:
        v6=0.0573
    
    purpose= pd.DataFrame([purpose],columns=["purpose"])
    purpose.replace({'purpose': {'renewable_energy': 'renewable_energy-moving', 'moving': 'renewable_energy-moving', 'medical': 'medical-house-debt_consolidation-other', 'house': 'medical-house-debt_consolidation-other', 'debt_consolidation': 'medical-house-debt_consolidation-other', 'other': 'medical-house-debt_consolidation-other', 'vacation': 'vacation-major_purchase', 'major_purchase': 'vacation-major_purchase', 'home_improvement': 'home_improvement-educational-credit_card', 'educational': 'home_improvement-educational-credit_card', 'credit_card': 'home_improvement-educational-credit_card'}},inplace=True)

    if purpose.iloc[0,0]== "car":
        v7=-0.2694
    else:
        if purpose.iloc[0,0]== "home_improvement-educational-credit_card" :
            v7=-0.0611
        else:
            if purpose.iloc[0,0] == "medical-house-debt_consolidation-other":
                v7=-0.0137
            else:
                if purpose.iloc[0,0]=="renewable_energy-moving":
                    v7= 0.0225
                else:
                    if purpose.iloc[0,0]== "small_business" :
                        v7=0.3608                       
                    else:
                        if purpose.iloc[0,0]== "vacation-major_purchase" :
                            v7=-0.0288
                        else:
                            v7=-0.566
    
    addr_state= pd.DataFrame([addr_state],columns=["addr_state"])                
    addr_state.replace({'addr_state': {'AK': 'Region Oeste', 'AL': 'Region Sur', 'AR': 'Region Sur', 'AZ': 'Region Oeste', 'CA': 'Region Oeste', 'CO': 'Region Oeste', 'CT': 'Region Noreste', 'DC': 'Region Sur', 'DE': 'Region Sur', 'FL': 'Region Sur', 'GA': 'Region Sur', 'HI': 'Region Oeste', 'IA': 'Region Medio Oeste', 'ID': 'Region Oeste', 'IL': 'Region Medio Oeste', 'IN': 'Region Medio Oeste', 'KS': 'Region Medio Oeste', 'KY': 'Region Sur', 'LA': 'Region Sur', 'MA': 'Region Noreste', 'MD': 'Region Sur', 'ME': 'Region Noreste', 'MI': 'Region Medio Oeste', 'MN': 'Region Medio Oeste', 'MO': 'Region Medio Oeste', 'MS': 'Region Sur', 'MT': 'Region Oeste', 'NC': 'Region Sur', 'ND': 'Region Medio Oeste', 'NE': 'Region Medio Oeste', 'NH': 'Region Noreste', 'NJ': 'Region Noreste', 'NM': 'Region Oeste', 'NV': 'Region Oeste', 'NY': 'Region Noreste', 'OH': 'Region Medio Oeste', 'OK': 'Region Sur', 'OR': 'Region Oeste', 'PA': 'Region Noreste', 'RI': 'Region Noreste', 'SC': 'Region Sur', 'SD': 'Region Medio Oeste', 'TN': 'Region Sur', 'TX': 'Region Sur', 'UT': 'Region Oeste', 'VA': 'Region Sur', 'VT': 'Region Noreste', 'WA': 'Region Oeste', 'WI': 'Region Medio Oeste', 'WV': 'Region Sur', 'WY': 'Region Oeste'}},  inplace = True)
   
    if addr_state.iloc[0,0]== "Region Medio Oeste":
        v8=-0.1755
    else:
        if addr_state.iloc[0,0]== "Region Noreste" :
            v8=-0.0971
        else:
            if addr_state.iloc[0,0] == "Region Oeste":
                v8=-0.1796
            else:
                v8=-0.1034    
    
    
    
    emp_length=pd.DataFrame([emp_length],columns=["emp_length"])
    emp_length.replace({'emp_length': {'1 year': "1-5 years", '2 years': "1-5 years",'3 years': "1-5 years",'4 years': "1-5 years",'5 years': "1-5 years",'6 years': "6-9 years",'7 years': "6-9 years",'8 years': "6-9 years",'9 years': "6-9 years"}},  inplace = True)
    
    if emp_length.iloc[0,0]== "1-5 years":
        v9=-0.4279
    else:
        if emp_length.iloc[0,0]== "10+ years" :
            v9=-0.446
        else:
            if emp_length.iloc[0,0] == "6-9 years":
                v9=-0.4185
            else:
                v9=-0.3847
    
    if annual_inc <= 45000:
        v10=0.0163
    else:
        if annual_inc <= 63250:
            v10=-0.1098
        else:
            if annual_inc <= 90000:
                v10=-0.1877
            else:
                v10=-0.2745
    
    if revol_util == 0 or pd.isna(revol_util) :
        v11=-0.2855
    else:
        if revol_util <= 50:
            v11=-0.2382
        else:
            if revol_util <= 100:
                v11=-0.1389
            else:
                v11=0.1069
               
    if pd.isna(tot_cur_bal):
        v12=-0.2183
    else:
        if tot_cur_bal <= 28210:
            v12=-0.1465
        else:
            if tot_cur_bal <= 73471:
                v12=-0.0492
            else:
                if tot_cur_bal <=199810:
                    v12=-0.0919
                else:
                    v12=-0.0498
                    
    if pd.isna(mths_since_rcnt_il):
        v13=-0.3091
    else:
        if mths_since_rcnt_il <= 7:
            v13=-0.0536
        else:
            if mths_since_rcnt_il <= 13:
                v13=-0.0739
            else:
                if mths_since_rcnt_il <=23:
                    v13=-0.0602
                else:
                    v13=-0.0587
    
    if pd.isna(total_rev_hi_lim):
        v14=-0.2183
    else:
        if total_rev_hi_lim <= 13900:
            v14=-0.0966
        else:
            if total_rev_hi_lim <= 23500:
                v14=-0.0829
            else:
                if total_rev_hi_lim <=39200:
                    v14=-0.0738
                else:
                    v14=-0.0841
    
    v15=float(int_rate)*(0.0851)
    if pd.isna(inq_last_6mths):
        inq_last_6mths=0
    else:
        inq_last_6mths=inq_last_6mths
    v16=float(inq_last_6mths)*0.0357
    if pd.isna(revol_bal):
        revol_bal=0
    else:
        revol_bal=revol_bal
    v17=float(revol_bal)*2.18E-06
    if pd.isna(open_acc_6m):
        open_acc_6m=0
    else:
        open_acc_6m=open_acc_6m
    v18=float(open_acc_6m)*0.0602
    if pd.isna(total_cu_tl):
        total_cu_tl=0
    else:
        total_cu_tl=total_cu_tl
    v19=float(total_cu_tl)*-0.0223
    if pd.isna(avg_cur_bal):
        avg_cur_bal=0
    else:
        avg_cur_bal=avg_cur_bal
    v20=float(avg_cur_bal)*-1.33E-05
    if pd.isna(bc_open_to_buy):
        bc_open_to_buy=0
    else:
        bc_open_to_buy=bc_open_to_buy
    v21=float(bc_open_to_buy)*-9.33E-06
    if pd.isna(mort_acc):
        mort_acc=0
    else:
        mort_acc=mort_acc
    v22=float(mort_acc)*-0.0587
    if pd.isna(num_accts_ever_120_pd):
        num_accts_ever_120_pd=0
    else:
        num_accts_ever_120_pd=num_accts_ever_120_pd
    v23=float(num_accts_ever_120_pd)*0.0123
    if pd.isna(num_tl_90g_dpd_24m):
        num_tl_90g_dpd_24m=0
    else:
        num_tl_90g_dpd_24m=num_tl_90g_dpd_24m
    v24=float(num_tl_90g_dpd_24m)*0.0728
    if pd.isna(revol_bal_joint):
        revol_bal_joint=0
    else:
        revol_bal_joint=revol_bal_joint
    v25=float(revol_bal_joint)*-6.23E-06
    
    
    z=v1+v2+v3+v4+v5+v6+v7+v8+v9+v10+v11+v12+v13+v14+v15+v16+v17+v18+v19+v20+v21+v22+v23+v24+v25
    sigmoide= 1/(1+np.exp(-(z)))
    if sigmoide>0.2:
        decision="Negar"
    else:
        decision="Aceptar" 

    return(decision, sigmoide)