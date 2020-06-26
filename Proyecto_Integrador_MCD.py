# -*- coding: utf-8 -*-

"Created on Sat May 16 15:36:59 2020"

#Proyecto Integrador Semestre 1


# In[]
#Sección de librerias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from scipy.spatial import distance
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split
from sklearn import decomposition

# In[Lectura del todo el dataset y seleccipón de la muestra]

#Ingesta desde AWS
base_completa = pd.read_csv('https://cda-proyectointegrador.s3.amazonaws.com/BD/loan.csv')

#Extracción de la base de créditos clasificados como Charged Off y Fully Paid (Únicos a usar en el modelo)
chargedoff = base_completa[base_completa.loan_status == 'Charged Off']
fullypaid =base_completa[base_completa.loan_status == 'Fully Paid']

#Creación del nuevo dataset con los registros Charged Off y Fully Paid
df_v1 = pd.concat([chargedoff,fullypaid])


# In[]
#Limpieza de variables

#Eliminación de variables que estan asociadas al comportamiento del credito pues el modelo que se desea realizar es para ser
#utilizado en la generación del crédito
df_v2 = df_v1.drop(['id','member_id','url','mths_since_last_record','annual_inc_joint','all_util','dti_joint','mo_sin_old_il_acct',
                    'sec_app_earliest_cr_line','sec_app_inq_last_6mths','sec_app_mort_acc','sec_app_open_acc','sec_app_revol_util',
                    'sec_app_open_act_il','sec_app_num_rev_accts','sec_app_chargeoff_within_12_mths','sec_app_collections_12_mths_ex_med',
                    'sec_app_mths_since_last_major_derog','hardship_flag','hardship_type','hardship_reason','hardship_status','deferral_term',
                    'hardship_amount','hardship_start_date','hardship_end_date','payment_plan_start_date','hardship_length','hardship_dpd',
                    'hardship_loan_status','orig_projected_additional_accrued_interest','hardship_payoff_balance_amount',
                    'hardship_last_payment_amount','debt_settlement_flag_date','settlement_status','settlement_date','settlement_amount',
                    'settlement_percentage','settlement_term','debt_settlement_flag','pymnt_plan','desc','title','zip_code',
                    'last_pymnt_d','last_pymnt_amnt','next_pymnt_d','last_credit_pull_d','collections_12_mths_ex_med',
                    'mths_since_last_major_derog','policy_code','application_type','verification_status_joint','acc_now_delinq',
                    'tot_coll_amt','dti','out_prncp','out_prncp_inv','total_pymnt','total_pymnt_inv','total_rec_prncp','total_rec_int',
                    'total_rec_late_fee','recoveries','collection_recovery_fee'],axis=1)

#Reemplazo de nulos por ceros en la variables donde los nulos se puden interpretar como este valor
df_v2 = df_v2.fillna({'mths_since_last_major_derog':0,'open_acc_6m':0,'open_act_il':0,'open_acc_6m':0,'open_il_12m':0,'open_il_24m':0,
                      'total_bal_il':0,'open_rv_12m':0,'open_rv_24m':0,'max_bal_bc':0,'inq_last_12m':0,'inq_fi':0,'inq_fi':0,
                      'total_cu_tl':0,'inq_last_12m':0,'acc_open_past_24mths':0,'avg_cur_bal':0,'bc_open_to_buy':0,'bc_util':0,
                      'chargeoff_within_12_mths':0,'mort_acc':0,'num_actv_bc_tl':0,'num_actv_rev_tl':0,'num_bc_sats':0,'num_bc_tl':0,
                      'num_il_tl':0,'num_op_rev_tl':0,'num_rev_accts':0,'num_rev_tl_bal_gt_0':0,'num_sats':0,'num_tl_120dpd_2m':0,
                      'num_tl_30dpd':0,'num_tl_90g_dpd_24m':0,'num_tl_op_past_12m':0,'pct_tl_nvr_dlq':0,'percent_bc_gt_75':0,
                      'pub_rec_bankruptcies':0,'revol_bal_joint':0,'num_accts_ever_120_pd':0,'tax_liens':0,'total_bal_ex_mort':0})


#Conversión de variables numerícas a categoricas y objetos a categoricas
df_v2['annual_inc']=df_v2['annual_inc'].astype('category')
df_v2['delinq_2yrs']=df_v2['delinq_2yrs'].astype('category')
df_v2['mths_since_last_delinq']=df_v2['mths_since_last_delinq'].astype('category')
df_v2['revol_util']=df_v2['revol_util'].astype('category')
df_v2['tot_cur_bal']=df_v2['tot_cur_bal'].astype('category')
df_v2['mths_since_rcnt_il']=df_v2['mths_since_rcnt_il'].astype('category')
df_v2['total_rev_hi_lim']=df_v2['total_rev_hi_lim'].astype('category')
df_v2['loan_status']=df_v2['loan_status'].astype('category')
df_v2['purpose']=df_v2['purpose'].astype('category')
df_v2['addr_state']=df_v2['addr_state'].astype('category')
df_v2['emp_length']=df_v2['emp_length'].astype('category')
df_v2['term']=df_v2['term'].astype('category')
df_v2['home_ownership']=df_v2['home_ownership'].astype('category')
df_v2['verification_status']=df_v2['verification_status'].astype('category')
df_v2['disbursement_method']=df_v2['disbursement_method'].astype('category')
df_v2['mths_since_recent_bc_dlq']=df_v2['mths_since_recent_bc_dlq'].astype('category')
df_v2['mths_since_recent_inq']=df_v2['mths_since_recent_inq'].astype('category')
df_v2['mths_since_recent_revol_delinq']=df_v2['mths_since_recent_revol_delinq'].astype('category')


df_v2['mo_sin_old_rev_tl_op']=df_v2['mo_sin_old_rev_tl_op'].astype('category')#Esta no
df_v2['il_util']=df_v2['il_util'].astype('category') #Esta no
df_v2['emp_title']=df_v2['emp_title'].astype('category') #Esta no
df_v2['grade']=df_v2['grade'].astype('category') #Esta no
df_v2['sub_grade']=df_v2['sub_grade'].astype('category') #Esta no
df_v2['mo_sin_rcnt_rev_tl_op']=df_v2['mo_sin_rcnt_rev_tl_op'].astype('category')#Esta no
df_v2['mths_since_recent_bc']=df_v2['mths_since_recent_bc'].astype('category') #Esta no
df_v2['total_bc_limit']=df_v2['total_bc_limit'].astype('category') #Esta no
df_v2['total_il_high_credit_limit']=df_v2['total_il_high_credit_limit'].astype('category') #Esta no
df_v2['issue_d']=df_v2['issue_d'].astype('category') #Esta no
df_v2['initial_list_status']=df_v2['initial_list_status'].astype('category')#Esta no
df_v2['mo_sin_rcnt_tl']=df_v2['mo_sin_rcnt_tl'].astype('category') #Esta no
df_v2['tot_hi_cred_lim']=df_v2['tot_hi_cred_lim'].astype('category') #Esta no
df_v2['earliest_cr_line']=df_v2['earliest_cr_line'].astype('category') #Esta no

#Visualización de tipos de variables en el data frame
df_v2_var_types = df_v2.dtypes

# In[]

#Dimensión del data frame
[mb,nb] = np.shape(df_v2)

#Transformar a variable binaria la variable respuesta (1 para los créditos Charged Off y 0 para créditos Fully Paid)
marca_co = []
for i in range(mb):
    if df_v2.iloc[i,14] == 'Charged Off':
        marca_co.append(1)
    else:
        marca_co.append(0)

#Agregar variable respuesta al Data Frame como una variable categórica
df_v2['marca_co'] = marca_co 
plt.hist(df_v2['marca_co'])
plt.title("Distribución del set de datos por variable respuesta")


df_v2.pivot_table('marca_co', 'purpose', aggfunc=np.mean).sort_values(by='marca_co', ascending=False)
df_v2.pivot_table('marca_co', 'emp_length', aggfunc=np.mean).sort_values(by='marca_co', ascending=False)
df_v2.pivot_table('marca_co', 'term', aggfunc=np.mean).sort_values(by='marca_co', ascending=False)
df_v2.pivot_table('marca_co', 'grade', aggfunc=np.mean).sort_values(by='marca_co', ascending=False)
df_v2.pivot_table('marca_co', 'home_ownership', aggfunc=np.mean).sort_values(by='marca_co', ascending=False)
df_v2.pivot_table('marca_co', 'addr_state', aggfunc=np.mean).sort_values(by='marca_co', ascending=False)
df_v2.pivot_table('marca_co', 'mo_sin_rcnt_rev_tl_op', aggfunc=np.mean).sort_values(by='marca_co', ascending=False)



df_v2['marca_co']=df_v2['marca_co'].astype('category')

#Actualizar dimensión de la muestra usada
[mb,nb] = np.shape(df_v2)

#Reseteo del indice
df_v2 = df_v2.reset_index()
df_v2 = df_v2.drop(['index'], axis=1)

#Separar datos numericos
df_num = pd.DataFrame([])
df_str = pd.DataFrame([])

tipos = df_v2.dtypes

tipos2 = tipos
tipos2 = tipos2.to_frame()
tipos2 = tipos2.drop(tipos2[tipos2.iloc[:,0]=='category'].index)
tipos2t = np.transpose(tipos2)

df_num = df_v2[list(tipos2t.head(0))]


tipos3 = tipos
tipos3 = tipos3.to_frame()
tipos3 = tipos3.drop(tipos3[tipos3.iloc[:,0]!='category'].index)
tipos3t = np.transpose(tipos3)

df_str = df_v2[list(tipos3t.head(0))]

# In[]

#Normalización de datos para PCA
df_num_norm = StandardScaler().fit_transform(df_num)
df_num_norm = pd.DataFrame(df_num_norm,columns=list(tipos2t.head()))

# In[]

#Matriz de covarianza, correlaciones, gráfica de dependencia líneal y número de condición
cov_df = df_num_norm.cov()
var_global = sum(np.diag(cov_df))
det=np.linalg.det(cov_df)
corr_df = df_num_norm.corr()
sns.heatmap(corr_df, center=0, cmap='Blues_r')
cond_cov = np.linalg.cond(cov_df)

# In[]

#Identificación de outliers y Eliminación del 10%
#a=[]
a_rob = []
media_num_norm = np.array(df_num_norm.mean())
mediana_num_norm = np.array(df_num_norm.median())
inv_cov = np.linalg.inv(np.array(cov_df))
for i in range(len(df_num_norm.index)):
    #b = distance.mahalanobis(np.array(df_num_norm.iloc[i,:]),media_num_norm,inv_cov)
    b_rob = distance.mahalanobis(np.array(df_num_norm.iloc[i,:]),mediana_num_norm,inv_cov)
    #a.append(b)
    a_rob.append(b_rob)
    
#df_num_norm['mahal_normal'] = a
df_num_norm['mahal_rob'] = a_rob

#df_v2['mahal_normal'] = a
df_v2['mahal_rob'] = a_rob 

#a = pd.DataFrame(a)
a_rob = pd.DataFrame(a_rob)
#p = a.quantile(0.9)
p_rob = a_rob.quantile(0.9)
df_num_norm = df_num_norm[df_num_norm.mahal_rob < p_rob.iloc[0]]
df_v2 = df_v2[df_v2.mahal_rob < p_rob.iloc[0]]

df_num_norm = df_num_norm.drop(['mahal_rob'],axis=1)

# In[]

#Vectores y valores propios de la matriz de covarianza de la muestra
eig_vals, eig_vecs = np.linalg.eig(np.array(cov_df))
print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)

#Hacemos una lista de parejas (autovector, autovalor) 
eig_pares = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

#Ordenamos estas parejas den orden descendiente con la función sort
eig_pares.sort(key=lambda x: x[0], reverse=True)

#Visualizamos la lista de autovalores en orden desdenciente
print('Autovalores en orden descendiente:')
for i in eig_pares:
    print(i[0])
    
#A partir de los autovalores, calculamos la varianza explicada
tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

#Gráfico de variabilidad explicada
plt.plot(cum_var_exp, color="green")
plt.xlabel("Autovalores")
plt.ylabel("Variabilidad explicada")
plt.title("Variabilidad explicada por los autovalores")
#Se puede correr para identificar cuales son las variables que más peso tienen, pero en realidad no vale la penas utilizarlo para
#reducir dimensionalidad

# In[]

#Vectores y valores singulares
df_num_norm = np.array(df_num_norm)
df_num_norm = np.asmatrix(df_num_norm)
U, s, Vh = np.linalg.svd(df_num_norm, full_matrices=False)
variabilidad = np.cumsum(s/sum(s))

# In[]

pca = decomposition.PCA(n_components=None)

pca.fit(df_num_norm)

X = pca.transform(df_num_norm)
plt.plot(np.cumsum(pca.explained_variance_ratio_), color="red")
plt.xlabel('numero de componentes')
plt.ylabel('Variabilidad explicada acumulada')
plt.title("Variabilidad explicada por componentes principales")

# In[]

#Análisis de dependencia entre variables numéricas
df_num_norm = np.array(df_num_norm)
DD = np.diag(inv_cov)*np.diag(cov_df)
print(DD)
DD_inverso = DD**-1
print(DD_inverso)
UNO_menos_DD_inverso = 1 - DD_inverso
UNO_menos_DD_inverso = pd.DataFrame(UNO_menos_DD_inverso)
UNO_menos_DD_inverso['indice_original']=UNO_menos_DD_inverso.index
tipos2_2=tipos2.reset_index()
UNO_menos_DD_inverso['label']=list(tipos2_2.iloc[:,0])
UNO_menos_DD_inverso=UNO_menos_DD_inverso.sort_values(0, ascending=False)
print(UNO_menos_DD_inverso.round(3))

# In[]
#Recategorizar las variables categoricas

#Variable annual_inc a Rango_Ing
percentiles=[]
P25=df_v2['annual_inc'].astype('int').quantile(0.25)
P50=df_v2['annual_inc'].astype('int').quantile(0.5)
P75=df_v2['annual_inc'].astype('int').quantile(0.75)

for lab, row in df_v2.iterrows():
    if df_v2.loc[lab,'annual_inc'].astype('int') <= P25:
        percentiles.append("<="+str(P25))
    else:
        if df_v2.loc[lab,'annual_inc'] <= P50:
            percentiles.append("<="+str(P50))
        else:
            if df_v2.loc[lab,'annual_inc'] <= P75:
                percentiles.append("<="+str(P75))
            else:
                percentiles.append(">"+str(P75))

df_v2['Rango_Ing']=percentiles

#Variable revol_util a Rango_Uso_CredRot
Rango_Uso=[]
for lab, row in df_v2.iterrows():
    if df_v2.loc[lab,'revol_util'] == 0:
        Rango_Uso.append("0")
    else:
        if df_v2.loc[lab,'revol_util'] <= 50:
            Rango_Uso.append("Entre 1 y 50%")
        else:
            if df_v2.loc[lab,'revol_util']<= 100:
                Rango_Uso.append("Entre 51 y 100%")
            else:
                Rango_Uso.append("Mayor a 100%")
            
df_v2['Rango_Uso_CredRot']=Rango_Uso


#Variable tot_cur_bal a Rango_Saldo_Actual
percentiles=[]
P25=df_v2['tot_cur_bal'].astype('float').quantile(0.25)
P50=df_v2['tot_cur_bal'].astype('float').quantile(0.5)
P75=df_v2['tot_cur_bal'].astype('float').quantile(0.75)

for lab, row in df_v2.iterrows():
    if pd.isna(df_v2.loc[lab,'tot_cur_bal']):
        percentiles.append('No aplica')
    else:
        if df_v2.loc[lab,'tot_cur_bal'] <= P25:
            percentiles.append("<="+str(P25))
        else:
            if df_v2.loc[lab,'tot_cur_bal']<= P50:
                percentiles.append("<="+str(P50))
            else:
                if df_v2.loc[lab,'tot_cur_bal'] <= P75:
                    percentiles.append("<="+str(P75))
                else:
                    percentiles.append(">"+str(P75))

df_v2['Rango_Saldo_Actual']=percentiles


#Variable mths_since_rcnt_il a Rango_Meses_Apertura_CPP
percentiles=[]
P25=df_v2['mths_since_rcnt_il'].astype('float').quantile(0.25)
P50=df_v2['mths_since_rcnt_il'].astype('float').quantile(0.5)
P75=df_v2['mths_since_rcnt_il'].astype('float').quantile(0.75)

for lab, row in df_v2.iterrows():
    if pd.isna(df_v2.loc[lab,'mths_since_rcnt_il']):
        percentiles.append('No aplica')
    else:
        if df_v2.loc[lab,'mths_since_rcnt_il'] <= P25:
            percentiles.append("<="+str(P25))
        else:
            if df_v2.loc[lab,'mths_since_rcnt_il']<= P50:
                percentiles.append("<="+str(P50))
            else:
                if df_v2.loc[lab,'mths_since_rcnt_il'] <= P75:
                    percentiles.append("<="+str(P75))
                else:
                    percentiles.append(">"+str(P75))

df_v2['Rango_Meses_Apertura_CPP']=percentiles


#Variable total_rev_hi_lim a Rango_Limite_Total_Rot 
percentiles=[]
P25=df_v2['total_rev_hi_lim'].astype('float').quantile(0.25)
P50=df_v2['total_rev_hi_lim'].astype('float').quantile(0.5)
P75=df_v2['total_rev_hi_lim'].astype('float').quantile(0.75)

for lab, row in df_v2.iterrows():
    if pd.isna(df_v2.loc[lab,'total_rev_hi_lim']):
        percentiles.append('No aplica')
    else:
        if df_v2.loc[lab,'total_rev_hi_lim'] <= P25:
            percentiles.append("<="+str(P25))
        else:
            if df_v2.loc[lab,'total_rev_hi_lim']<= P50:
                percentiles.append("<="+str(P50))
            else:
                if df_v2.loc[lab,'total_rev_hi_lim'] <= P75:
                    percentiles.append("<="+str(P75))
                else:
                    percentiles.append(">"+str(P75))

df_v2['Rango_Limite_Total_Rot']=percentiles


#Variable delinq_2yrs a Rango_delinq_2yrs 
Rango_Uso=[]
for lab, row in df_v2.iterrows():
    if df_v2.loc[lab,'delinq_2yrs'] == 0:
        Rango_Uso.append("0")
    else:
        if df_v2.loc[lab,'delinq_2yrs'] <= 5:
            Rango_Uso.append("Entre 1 y 5")
        else:
            if df_v2.loc[lab,'delinq_2yrs']<= 10:
                Rango_Uso.append("Entre 6 y 10")
            else:
                if df_v2.loc[lab,'delinq_2yrs']<= 20:
                    Rango_Uso.append("Entre 11 y 20")
                else:
                    Rango_Uso.append("Mas de 20")
            
df_v2['Rango_delinq_2yrs']=Rango_Uso


#Variable mths_since_recent_bc_dlq a Rango_Meses_Mora_TC
percentiles=[]
P25=df_v2['mths_since_recent_bc_dlq'].astype('float').quantile(0.25)
P50=df_v2['mths_since_recent_bc_dlq'].astype('float').quantile(0.5)
P75=df_v2['mths_since_recent_bc_dlq'].astype('float').quantile(0.75)

for lab, row in df_v2.iterrows():
    if pd.isna(df_v2.loc[lab,'mths_since_recent_bc_dlq']):
        percentiles.append('No aplica')
    else:
        if df_v2.loc[lab,'mths_since_recent_bc_dlq'] <= P25:
            percentiles.append("<="+str(P25))
        else:
            if df_v2.loc[lab,'mths_since_recent_bc_dlq']<= P50:
                percentiles.append("<="+str(P50))
            else:
                if df_v2.loc[lab,'mths_since_recent_bc_dlq'] <= P75:
                    percentiles.append("<="+str(P75))
                else:
                    percentiles.append(">"+str(P75))

df_v2['Rango_Meses_Mora_TC']=percentiles

#Variable mths_since_recent_inq a Rango_Meses_Ult_inv
percentiles=[]
P25=df_v2['mths_since_recent_inq'].astype('float').quantile(0.25)
P50=df_v2['mths_since_recent_inq'].astype('float').quantile(0.5)
P75=df_v2['mths_since_recent_inq'].astype('float').quantile(0.75)

for lab, row in df_v2.iterrows():
    if pd.isna(df_v2.loc[lab,'mths_since_recent_inq']):
        percentiles.append('No aplica')
    else:
        if df_v2.loc[lab,'mths_since_recent_inq'] <= P25:
            percentiles.append("<="+str(P25))
        else:
            if df_v2.loc[lab,'mths_since_recent_inq']<= P50:
                percentiles.append("<="+str(P50))
            else:
                if df_v2.loc[lab,'mths_since_recent_inq'] <= P75:
                    percentiles.append("<="+str(P75))
                else:
                    percentiles.append(">"+str(P75))

df_v2['Rango_Meses_Ult_inv']=percentiles

#Variable mths_since_recent_revol_delinq a Rango_Meses_Mora_R
percentiles=[]
P25=df_v2['mths_since_recent_revol_delinq'].astype('float').quantile(0.25)
P50=df_v2['mths_since_recent_revol_delinq'].astype('float').quantile(0.5)
P75=df_v2['mths_since_recent_revol_delinq'].astype('float').quantile(0.75)

for lab, row in df_v2.iterrows():
    if pd.isna(df_v2.loc[lab,'mths_since_recent_revol_delinq']):
        percentiles.append('No aplica')
    else:
        if df_v2.loc[lab,'mths_since_recent_revol_delinq'] <= P25:
            percentiles.append("<="+str(P25))
        else:
            if df_v2.loc[lab,'mths_since_recent_revol_delinq']<= P50:
                percentiles.append("<="+str(P50))
            else:
                if df_v2.loc[lab,'mths_since_recent_revol_delinq'] <= P75:
                    percentiles.append("<="+str(P75))
                else:
                    percentiles.append(">"+str(P75))

df_v2['Rango_Meses_Mora_Rot']=percentiles






#Variable emp_length, purpose, addr_state
df_v2.replace({'emp_length': {'1 year': "1-5 years", '2 years': "1-5 years",'3 years': "1-5 years",'4 years': "1-5 years",'5 years': "1-5 years",'6 years': "6-9 years",'7 years': "6-9 years",'8 years': "6-9 years",'9 years': "6-9 years"}},  inplace = True)
df_v2.replace({'addr_state': {'AK': 'Region Oeste', 'AL': 'Region Sur', 'AR': 'Region Sur', 'AZ': 'Region Oeste', 'CA': 'Region Oeste', 'CO': 'Region Oeste', 'CT': 'Region Noreste', 'DC': 'Region Sur', 'DE': 'Region Sur', 'FL': 'Region Sur', 'GA': 'Region Sur', 'HI': 'Region Oeste', 'IA': 'Region Medio Oeste', 'ID': 'Region Oeste', 'IL': 'Region Medio Oeste', 'IN': 'Region Medio Oeste', 'KS': 'Region Medio Oeste', 'KY': 'Region Sur', 'LA': 'Region Sur', 'MA': 'Region Noreste', 'MD': 'Region Sur', 'ME': 'Region Noreste', 'MI': 'Region Medio Oeste', 'MN': 'Region Medio Oeste', 'MO': 'Region Medio Oeste', 'MS': 'Region Sur', 'MT': 'Region Oeste', 'NC': 'Region Sur', 'ND': 'Region Medio Oeste', 'NE': 'Region Medio Oeste', 'NH': 'Region Noreste', 'NJ': 'Region Noreste', 'NM': 'Region Oeste', 'NV': 'Region Oeste', 'NY': 'Region Noreste', 'OH': 'Region Medio Oeste', 'OK': 'Region Sur', 'OR': 'Region Oeste', 'PA': 'Region Noreste', 'RI': 'Region Noreste', 'SC': 'Region Sur', 'SD': 'Region Medio Oeste', 'TN': 'Region Sur', 'TX': 'Region Sur', 'UT': 'Region Oeste', 'VA': 'Region Sur', 'VT': 'Region Noreste', 'WA': 'Region Oeste', 'WI': 'Region Medio Oeste', 'WV': 'Region Sur', 'WY': 'Region Oeste'}},  inplace = True)
df_v2.replace({'purpose': {'renewable_energy': 'renewable_energy-moving', 'moving': 'renewable_energy-moving', 'medical': 'medical-house-debt_consolidation-other', 'house': 'medical-house-debt_consolidation-other', 'debt_consolidation': 'medical-house-debt_consolidation-other', 'other': 'medical-house-debt_consolidation-other', 'vacation': 'vacation-major_purchase', 'major_purchase': 'vacation-major_purchase', 'home_improvement': 'home_improvement-educational-credit_card', 'educational': 'home_improvement-educational-credit_card', 'credit_card': 'home_improvement-educational-credit_card'}},inplace=True)

# In[]

#Variables dummies
prueba = pd.get_dummies(df_v2[['Rango_Meses_Mora_Rot','Rango_Meses_Ult_inv','Rango_Meses_Mora_TC','disbursement_method','verification_status','home_ownership','term','purpose','emp_length','Rango_Ing','addr_state', 'Rango_Uso_CredRot', 'Rango_Saldo_Actual', 'Rango_Meses_Apertura_CPP', 'Rango_Limite_Total_Rot', 'Rango_delinq_2yrs']])
df_num=df_v2[list(tipos2t.head(0))]
result = pd.concat([df_num, prueba], axis=1, sort=False)
df_x=result

# In[]

#Modelo de regresión Logística utilizando sklearn

df_y=pd.DataFrame(df_v2['marca_co'])

X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, random_state=0)     
        
# In[]

#Modelo de regresión lineal generalizado

import gc
gc.collect()

model3 = sm.GLM(y_train,X_train,family=sm.families.Binomial())

result3 = model3.fit()

print(result3.summary())

result3.aic
#maxiter=1000,wls_method='pinv',cov_type='robust'

#Se calcula la predicción según el score de cada persona (Entrenamiento)
prediccion=[]
entrena=[]
parametros=np.array(result3.params)
for lab, row in X_train.iterrows():
    if 1/(1+np.exp(-(np.dot(parametros,np.array(X_train.loc[lab,:])))))>0.5:
        y=1
    else:
        y=0
    prediccion.append(y)
    if y==y_train.loc[lab,'marca_co']:
        entrena.append(1)
    else:
        entrena.append(0)

1-np.mean(entrena) 

results = confusion_matrix(y_train, pd.DataFrame(prediccion))


prediccion=[]
test=[]
parametros=np.array(result3.params)
for lab, row in X_test.iterrows():
    if 1/(1+np.exp(-(np.dot(parametros,np.array(X_test.loc[lab,:])))))>0.5:
        y=1
    else:
        y=0
    prediccion.append(y)
    if y==y_test.loc[lab,'marca_co']:
        test.append(1)
    else:
        test.append(0)

1-np.mean(test) 

results = confusion_matrix(y_test, pd.DataFrame(prediccion))

# In[]
#Se eliminan variables por correlaciones multiples
Eliminar = UNO_menos_DD_inverso[UNO_menos_DD_inverso.iloc[:,0] > 0.7]
X_train= X_train.drop(Eliminar.iloc[:,2],axis=1)
X_test= X_test.drop(Eliminar.iloc[:,2],axis=1)

gc.collect()
#Segundo modelo
model4 = sm.GLM(y_train,X_train,family=sm.families.Binomial())

result4 = model4.fit()
print(result4.summary())
result4.aic

#Se calcula la predicción según el score de cada persona (Entrenamiento)
prediccion=[]
entrena=[]
parametros=np.array(result4.params)
for lab, row in X_train.iterrows():
    if 1/(1+np.exp(-(np.dot(parametros,np.array(X_train.loc[lab,:])))))>0.5:
        y=1
    else:
        y=0
    prediccion.append(y)
    if y==y_train.loc[lab,'marca_co']:
        entrena.append(1)
    else:
        entrena.append(0)

1-np.mean(entrena) 

results = confusion_matrix(y_train, pd.DataFrame(prediccion))


prediccion=[]
test=[]
parametros=np.array(result4.params)
for lab, row in X_test.iterrows():
    if 1/(1+np.exp(-(np.dot(parametros,np.array(X_test.loc[lab,:])))))>0.5:
        y=1
    else:
        y=0
    prediccion.append(y)
    if y==y_test.loc[lab,'marca_co']:
        test.append(1)
    else:
        test.append(0)

1-np.mean(test) 

results = confusion_matrix(y_test, pd.DataFrame(prediccion))

# In[]
#Se eliminan variables por p-valor
X_train=X_train.drop(['max_bal_bc','inq_fi','inq_last_12m','chargeoff_within_12_mths','delinq_amnt','num_tl_120dpd_2m','num_tl_30dpd','home_ownership_ANY','home_ownership_MORTGAGE','home_ownership_NONE','home_ownership_OTHER','home_ownership_OWN','home_ownership_RENT','Rango_delinq_2yrs_0','Rango_delinq_2yrs_Entre 1 y 5','Rango_delinq_2yrs_Entre 11 y 20','Rango_delinq_2yrs_Entre 6 y 10'],axis=1)
X_test=X_test.drop(['max_bal_bc','inq_fi','inq_last_12m','chargeoff_within_12_mths','delinq_amnt','num_tl_120dpd_2m','num_tl_30dpd','home_ownership_ANY','home_ownership_MORTGAGE','home_ownership_NONE','home_ownership_OTHER','home_ownership_OWN','home_ownership_RENT','Rango_delinq_2yrs_0','Rango_delinq_2yrs_Entre 1 y 5','Rango_delinq_2yrs_Entre 11 y 20','Rango_delinq_2yrs_Entre 6 y 10'],axis=1)

gc.collect()
#Segundo modelo
model5 = sm.GLM(y_train,X_train,family=sm.families.Binomial())

result5 = model5.fit()
print(result5.summary())

#Se calcula la predicción según el score de cada persona (Entrenamiento)
prediccion=[]
entrena=[]
parametros=np.array(result5.params)
for lab, row in X_train.iterrows():
    if 1/(1+np.exp(-(np.dot(parametros,np.array(X_train.loc[lab,:])))))>0.5:
        y=1
    else:
        y=0
    prediccion.append(y)
    if y==y_train.loc[lab,'marca_co']:
        entrena.append(1)
    else:
        entrena.append(0)

1-np.mean(entrena) 

results = confusion_matrix(y_train, pd.DataFrame(prediccion))


prediccion=[]
test=[]
parametros=np.array(result5.params)
for lab, row in X_test.iterrows():
    if 1/(1+np.exp(-(np.dot(parametros,np.array(X_test.loc[lab,:])))))>0.5:
        y=1
    else:
        y=0
    prediccion.append(y)
    if y==y_test.loc[lab,'marca_co']:
        test.append(1)
    else:
        test.append(0)

1-np.mean(test) 

results = confusion_matrix(y_test, pd.DataFrame(prediccion))


# In[]

parametros=np.array(result5.params)
Costo=df_v2.pivot_table('loan_amnt', 'marca_co', aggfunc=np.sum).sort_values(by='loan_amnt', ascending=False)
error=[]
CostoDecDin=[]
PorcCost=[]
Tolerancia=[]
for i in range(10):
    prediccion=[]
    entrena=[]
    param= (i+1)/(20)
    for lab, row in X_train.iterrows():
        if 1/(1+np.exp(-(np.dot(parametros,np.array(X_train.loc[lab,:])))))>param:
            y=1
        else:
            y=0
        prediccion.append(y)
        if y==y_train.loc[lab,'marca_co']:
            entrena.append(1)
        else:
            entrena.append(0)

    error.append(1-np.mean(entrena))
    
    results = confusion_matrix(y_train, pd.DataFrame(prediccion))
    Prob_1_0=results[0,1]/(results[0,1]+results[0,0])
    Prob_0_1=results[1,0]/(results[1,0]+results[1,1])
    Costo_Dec=(Prob_1_0*Costo.iloc[0,0]*Tasa)+(Prob_0_1*Costo.iloc[1,0])
    CostoDecDin.append(Costo_Dec)
    PorcCost.append(Costo_Dec/(Costo.iloc[0,0]+Costo.iloc[1,0]))
    Tolerancia.append(param)
    
plt.plot(Tolerancia,error, color="red")
plt.xlabel('Tolerancia')
plt.ylabel('Error de predicción')
plt.title("Error de predicción Vs Tolerancia")

plt.plot(Tolerancia,CostoDecDin, color="green")
plt.xlabel('Tolerancia')
plt.ylabel('Costo')
plt.title("Costo de decisión (USD) Vs Tolerancia")

plt.plot(Tolerancia,PorcCost, color="blue")
plt.xlabel('Tolerancia')
plt.ylabel('% Costo')
plt.title("% Costo de decisión Vs Tolerancia")


