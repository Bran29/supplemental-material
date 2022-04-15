from scipy.optimize import minimize
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from pyswarm import pso
import matplotlib
sns.set_theme(style="white", palette="muted")
matplotlib.rcParams.update({'font.size': 18})
def convex_WTP(v):
        b=np.multiply(v,v)/111+10
        return b

def concave_WTP(v):
        b=10*np.power(v,0.478)+10
        return b

def logisti_WTP(v):
        b=91/(1+np.exp(-(0.1*(v-50))))+10
        return b

def gauss_dist(v,miu=50,sigma=20):
        d=(1/(sigma*np.sqrt(2*np.pi)))*np.exp(-np.power((v-miu),2)/(2*sigma*sigma))
        return d*4

def gauss_dist2(v,miu1=25,sigma1=10,miu2=75,sigma2=10):
        d1=(1/(sigma1*np.sqrt(2*np.pi)))*np.exp(-np.power((v-miu1),2)/(2*sigma1*sigma1))
        d2=(1/(sigma2*np.sqrt(2*np.pi)))*np.exp(-np.power((v-miu2),2)/(2*sigma2*sigma2))
        return (d1+d2)/2

def gauss_dist3(v,miu=100,sigma=20):
        d=(1/(sigma*np.sqrt(2*np.pi)))*np.exp(-np.power((v-miu),2)/(2*sigma*sigma))
        return d*4

def exp_dist(v,lamd=0.5):
        d=lamd*np.exp(-(lamd*v))
        return d

def object_func(x):
        b = convex_WTP(v)
        d=gauss_dist2(v)
        y= -np.sum(np.multiply(np.multiply(d,x),np.array([x[i] <= b[i] for i in range(len(v) )], dtype=int)))
        return  y

def cons(x):
        cons1=[]
        # cons1.append(x[0])
        # cons1.append(-x[0])
        for i in range(20):
                cons1.append(x[i+1]-x[i])
        for i in range(1,20):
                cons1.append(x[i]/5/i-x[i+1]/5/(i+1))
        return cons1

e = 1e-10 # 非常接近0的值
v= np.arange(0,101,5)
b2=convex_WTP(v)
b1=concave_WTP(v)
b3=logisti_WTP(v)

d1=gauss_dist2(v)
d2=gauss_dist(v)
d3=gauss_dist3(v)
width=0.5
fig, ax = plt.subplots(3, 5, figsize=(55,25))
ax[0][0].plot(v,b1,c='b',marker='o',linewidth=3.0)
ax[0][1].plot(v,b2,c='b',marker='o',linewidth=3.0)
ax[0][2].plot(v,b3,c='b',marker='o',linewidth=3.0)

color = 'tab:blue'

ax[0][0].set_ylabel('WTP',color=color, fontsize=35)
ax[0][1].set_ylabel('WTP',color=color, fontsize=35)
ax[0][2].set_ylabel('WTP',color=color, fontsize=35)
ax[0][0].set_xlabel('value', fontsize=35)
ax[0][1].set_xlabel('value', fontsize=35)
ax[0][2].set_xlabel('value', fontsize=35)

ax[0][0].tick_params(axis='y', labelcolor=color, labelsize=25)
ax[0][1].tick_params(axis='y', labelcolor=color, labelsize=25)
ax[0][2].tick_params(axis='y', labelcolor=color, labelsize=25)
ax[0][0].tick_params(axis='x', labelsize=25)
ax[0][1].tick_params(axis='x',  labelsize=25)
ax[0][2].tick_params(axis='x', labelsize=25)
color = 'tab:red'
ax00_t=ax[0][0].twinx()
ax00_t.plot(v,d1,c='r',marker='*',linewidth=3.0)
ax01_t=ax[0][1].twinx()
ax01_t.plot(v,d1,c='r',marker='*',linewidth=3.0)
ax02_t=ax[0][2].twinx()
ax02_t.plot(v,d1,c='r',marker='*',linewidth=3.0)
ax00_t.set_ylabel('Requirement Distribution',color=color, fontsize=35)
ax01_t.set_ylabel('Requirement Distribution',color=color, fontsize=35)
ax02_t.set_ylabel('Requirement Distribution',color=color, fontsize=35)
ax00_t.set_xlabel('value',  fontsize=35)
ax01_t.set_xlabel('value',fontsize=35)
ax02_t.set_xlabel('value', fontsize=35)
ax00_t.tick_params(axis='y', labelcolor=color, labelsize=25)
ax01_t.tick_params(axis='y', labelcolor=color, labelsize=25)
ax02_t.tick_params(axis='y', labelcolor=color, labelsize=25)

# calculate price function in b1 d1

x0 = np.arange(10,81,3.5)
fun1 = lambda x : -np.sum(np.multiply(np.multiply(d1,x),np.array([x[i] <= b1[i] for i in range(len(v) )], dtype=int)))
cons = (
        {'type': 'ineq', 'fun': lambda x: x[0]},
        {'type': 'ineq', 'fun': lambda x: x[1]},
        {'type': 'ineq', 'fun': lambda x: x[2]},
        {'type': 'ineq', 'fun': lambda x: x[3]},
        {'type': 'ineq', 'fun': lambda x: x[4]},  # x>=e，即 x > 0
        {'type': 'ineq', 'fun': lambda x: x[5]},
        {'type': 'ineq', 'fun': lambda x: x[6]},
        {'type': 'ineq', 'fun': lambda x: x[7]},
        {'type': 'ineq', 'fun': lambda x: x[8]},  # x>=e，即 x > 0
        {'type': 'ineq', 'fun': lambda x: x[9]},
        {'type': 'ineq', 'fun': lambda x: x[10]},
        {'type': 'ineq', 'fun': lambda x: x[11]},
        {'type': 'ineq', 'fun': lambda x: x[12]},
        {'type': 'ineq', 'fun': lambda x: x[13]},  # x>=e，即 x > 0
        {'type': 'ineq', 'fun': lambda x: x[14]},
        {'type': 'ineq', 'fun': lambda x: x[15]},
        {'type': 'ineq', 'fun': lambda x: x[16]},
        {'type': 'ineq', 'fun': lambda x: x[17]},
        {'type': 'ineq', 'fun': lambda x: x[18]},  # x>=e，即 x > 0
        {'type': 'ineq', 'fun': lambda x: x[19]},
        {'type': 'ineq', 'fun': lambda x: x[20]},

        {'type': 'ineq', 'fun': lambda x: -x[0] +100},
        {'type': 'ineq', 'fun': lambda x: -x[1] +100},
        {'type': 'ineq', 'fun': lambda x: -x[2] +100},
        {'type': 'ineq', 'fun': lambda x: -x[3] +100},
        {'type': 'ineq', 'fun': lambda x: -x[4] +100},  # x>=e，即 x > 0
        {'type': 'ineq', 'fun': lambda x: -x[5] +100},
        {'type': 'ineq', 'fun': lambda x: -x[6] +100},
        {'type': 'ineq', 'fun': lambda x: -x[7] +100},
        {'type': 'ineq', 'fun': lambda x: -x[8] +100},  # x>=e，即 x > 0
        {'type': 'ineq', 'fun': lambda x: -x[9] +100},
        {'type': 'ineq', 'fun': lambda x: -x[10] +100},
        {'type': 'ineq', 'fun': lambda x: -x[11] +100},
        {'type': 'ineq', 'fun': lambda x: -x[12] +100},
        {'type': 'ineq', 'fun': lambda x: -x[13] +100},  # x>=e，即 x > 0
        {'type': 'ineq', 'fun': lambda x: -x[14] +100},
        {'type': 'ineq', 'fun': lambda x: -x[15] +100},
        {'type': 'ineq', 'fun': lambda x: -x[16] +100},
        {'type': 'ineq', 'fun': lambda x: -x[17] +100},
        {'type': 'ineq', 'fun': lambda x: -x[18] +100},  # x>=e，即 x > 0
        {'type': 'ineq', 'fun': lambda x: -x[19] +100},
        {'type': 'ineq', 'fun': lambda x: -x[20] +100},


        # {'type': 'ineq', 'fun': lambda x: x[0] - x[1]/2} ,
        {'type': 'ineq', 'fun': lambda x:  x[1]/5/1-x[2]/5/2} ,
        {'type': 'ineq', 'fun': lambda x:  x[2]/5/2-x[3]/5/3},
        {'type': 'ineq', 'fun': lambda x:  x[3]/5/3-x[4]/5/4} ,
        {'type': 'ineq', 'fun': lambda x:  x[4]/5/4-x[5]/5/5},
        {'type': 'ineq', 'fun': lambda x:  x[5]/5/5-x[6]/5/6} ,
        {'type': 'ineq', 'fun': lambda x:  x[6]/5/6-x[7]/5/7},
        {'type': 'ineq', 'fun': lambda x:  x[7]/5/7-x[8]/5/8} ,
        {'type': 'ineq', 'fun': lambda x:  x[8]/5/8-x[9]/5/9},
        {'type': 'ineq', 'fun': lambda x:  x[9]/5/9-x[10]/5/10} ,
        {'type': 'ineq', 'fun': lambda x:  x[10]/5/10-x[11]/5/11},
        {'type': 'ineq', 'fun': lambda x:  x[11]/5/11-x[12]/5/12} ,
        {'type': 'ineq', 'fun': lambda x:  x[12]/5/12-x[13]/5/13},
        {'type': 'ineq', 'fun': lambda x: x[13] /5/ 13 - x[14] /5/ 14},
        {'type': 'ineq', 'fun': lambda x: x[14] /5/ 14 - x[15] /5/ 15},
        {'type': 'ineq', 'fun': lambda x: x[15] /5/ 15 - x[16] /5/ 16},
        {'type': 'ineq', 'fun': lambda x: x[16] /5/ 16 - x[17] /5/ 17},
        {'type': 'ineq', 'fun': lambda x: x[17] /5/ 17 - x[18] /5/ 18},
        {'type': 'ineq', 'fun': lambda x: x[18] /5/ 18 - x[19] /5/ 19},
        {'type': 'ineq', 'fun': lambda x: x[19] /5/ 19 - x[20] /5/ 20},


        {'type': 'ineq', 'fun': lambda x: x[20] - x[19]},
        {'type': 'ineq', 'fun': lambda x: x[19] - x[18]},
        {'type': 'ineq', 'fun': lambda x: x[18] - x[17]},
        {'type': 'ineq', 'fun': lambda x: x[17] - x[16]},
        {'type': 'ineq', 'fun': lambda x: x[16] - x[15]},
        {'type': 'ineq', 'fun': lambda x: x[15] - x[14]},
        {'type': 'ineq', 'fun': lambda x: x[14] - x[13]},
        {'type': 'ineq', 'fun': lambda x: x[13] - x[12]},
        {'type': 'ineq', 'fun': lambda x: x[12] - x[11]},
        {'type': 'ineq', 'fun': lambda x: x[11] - x[10]},
        {'type': 'ineq', 'fun': lambda x: x[10] - x[9]},
        {'type': 'ineq', 'fun': lambda x: x[9] - x[8]},
        {'type': 'ineq', 'fun': lambda x: x[8] - x[7]},
        {'type': 'ineq', 'fun': lambda x: x[7] - x[6]},
        {'type': 'ineq', 'fun': lambda x: x[6] - x[5]},
        {'type': 'ineq', 'fun': lambda x: x[5] - x[4]},
        {'type': 'ineq', 'fun': lambda x: x[4] - x[3]},
        {'type': 'ineq', 'fun': lambda x:  x[3] -x[2] },
        {'type': 'ineq', 'fun': lambda x:  x[2] -x[1] },
        {'type': 'ineq', 'fun': lambda x:  x[1] -x[0] }
       )
 # 设置初始值

res = minimize(fun1, x0, method='SLSQP', constraints=cons)
vbm_x=res.x
print('vbm最小值：',res.fun)


max_x=np.ones(21)*100
max_r=-np.sum(np.multiply(np.multiply(d1,max_x),np.array([max_x[i] <= b1[i] for i in range(len(v) )], dtype=int)))
print('maxc最小值：',max_r)

med_x=np.ones(21)*np.median(b1)
med_r=-np.sum(np.multiply(np.multiply(d1,med_x),np.array([med_x[i] <= b1[i] for i in range(len(v) )], dtype=int)))
print('medr最小值：',med_r)

ax[1][0].plot(v,vbm_x,'o-',linewidth=3.0,label='SVBP')
ax[1][0].plot(v,max_x,'^-',linewidth=3.0,label='MaxW')
ax[1][0].plot(v,med_x,'*-',linewidth=3.0,label='MedW')
ax[1][0].set_xlabel('value', fontsize=35)
ax[1][0].set_ylabel('Price', fontsize=35)
ax[1][0].tick_params(axis='x', labelsize=25)
ax[1][0].tick_params(axis='y',  labelsize=25)
# ax[1][0].legend(loc="lower right", fontsize=35)

data_result=[-res.fun,-max_r,-med_r]
label = ('SVBP', 'MaxW', 'MedW')
bar_x = np.arange(len(label))

bars=ax[2][0].bar(x=bar_x-width/2,height=data_result, width=0.5, color=['b'])

color = 'tab:blue'
ax[2][0].set_yscale('log')
temp_i=0
for b in bars[1:]:
        height = round(bars[0].get_height()/b.get_height(),2)
        ax[2][0].annotate('$\\times${}'.format(height),
                    xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",color='black',
                    ha='center', va='bottom')


ax[2][0].set_ylabel('Revenue',color=color, fontsize=35)
ax[2][0].set_xlabel('(a)', fontsize=45)
ax[2][0].tick_params(axis='y', labelcolor=color, labelsize=25)
ax[2][0].tick_params(axis='x', labelsize=25)
afford_ratio_vbm=sum(np.array([d1[i] for i in range(len(v) )  if vbm_x[i] <= b1[i]]))/sum(d1)
afford_ratio_max=sum(np.array([d1[i] for i in range(len(v) )  if max_x[i] <= b1[i]]))/sum(d1)
afford_ratio_med=sum(np.array([d1[i] for i in range(len(v) )  if med_x[i] <= b1[i]]))/sum(d1)

data_result=[afford_ratio_vbm,afford_ratio_max,afford_ratio_med]
label = ('SVBP', 'MaxW', 'MedW')
bar_x = np.arange(len(label))
color = 'tab:red'
ax20_t=ax[2][0].twinx()
bars=ax20_t.bar(x=bar_x+width/2,height=data_result, width=0.5,color=['red'])

temp_i=0
for b in bars[1:]:
        height = round(bars[0].get_height()/b.get_height(),2)
        # ax20_t.annotate('$\\times${}'.format(height),
        #             xy=(b.get_x() + b.get_width() / 2, b.get_height()),
        #             xytext=(0, 3),  # 3 points vertical offset
        #             textcoords="offset points",color='black',
        #             ha='center', va='bottom')
        if temp_i == 0:
                ax20_t.annotate('$\\times${}'.format(height),
                                  xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                                  xytext=(0, 20),  # 3 points vertical offset
                                  textcoords="offset points", color='black',
                                  ha='center', va='bottom')
        else:
                ax20_t.annotate('$\\times${}'.format(height),
                                  xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                                  xytext=(0, 3),  # 3 points vertical offset
                                  textcoords="offset points", color='black',
                                  ha='center', va='bottom')
ax20_t.set_ylabel('Affordability Ratio',color=color, fontsize=35)
ax20_t.set_yscale('log')
ax20_t.tick_params(axis='y', labelcolor=color, labelsize=25)
ax[2][0].set_xticks(bar_x, label)



ax[2][1].set_yscale('log')
ax[2][2].set_yscale('log')
# calculate price function in b1 d1

x0 =np.arange(10,61,2.5)
fun1 = lambda x : -np.sum(np.multiply(np.multiply(d1,x),np.array([x[i] <= b2[i] for i in range(len(v) )], dtype=int)))
 # 设置初始值

res = minimize(fun1, x0, method='SLSQP', constraints=cons)
vbm_x=res.x
print('vbm最小值：',res.fun)


max_x=np.ones(21)*100
max_r=-np.sum(np.multiply(np.multiply(d1,max_x),np.array([max_x[i] <= b2[i] for i in range(len(v) )], dtype=int)))
print('maxc最小值：',max_r)

med_x=np.ones(21)*np.median(b2)
med_r=-np.sum(np.multiply(np.multiply(d1,med_x),np.array([med_x[i] <= b2[i] for i in range(len(v) )], dtype=int)))
print('medr最小值：',med_r)


ax[1][1].plot(v,vbm_x,'o-',linewidth=3.0,label='SVBP')
ax[1][1].plot(v,max_x,'^-',linewidth=3.0,label='MaxW')
ax[1][1].plot(v,med_x,'*-',linewidth=3.0,label='MedW')
ax[1][1].set_xlabel('value', fontsize=35)
ax[1][1].set_ylabel('Price', fontsize=35)
ax[1][1].tick_params(axis='x', labelsize=25)
ax[1][1].tick_params(axis='y',  labelsize=25)
# ax[1][1].legend(loc="lower right", fontsize=35)
data_result=[-res.fun,-max_r,-med_r]
label = ('SVBP', 'MaxW', 'MedW')
bar_x = np.arange(len(label))

bars=ax[2][1].bar(x=bar_x-width/2,height=data_result, width=0.5, tick_label=label,color=['b'])
ax[2][1].set_yscale('log')
color = 'tab:blue'
temp_i=0
for b in bars[1:]:
        height = round(bars[0].get_height()/b.get_height(),2)
        ax[2][1].annotate('$\\times${}'.format(height),
                    xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",color='black',
                    ha='center', va='bottom')


ax[2][1].set_ylabel('Revenue',color=color, fontsize=35)
ax[2][1].set_xlabel('(b)', fontsize=45)
ax[2][1].tick_params(axis='y', labelcolor=color, labelsize=25)
ax[2][1].tick_params(axis='x', labelsize=25)
afford_ratio_vbm=sum(np.array([d1[i] for i in range(len(v) )  if vbm_x[i] <= b2[i]]))/sum(d1)
afford_ratio_max=sum(np.array([d1[i] for i in range(len(v) )  if max_x[i] <= b2[i]]))/sum(d1)
afford_ratio_med=sum(np.array([d1[i] for i in range(len(v) )  if med_x[i] <= b2[i]]))/sum(d1)

data_result=[afford_ratio_vbm,afford_ratio_max,afford_ratio_med]
label = ('SVBP', 'MaxW', 'MedW')
bar_x = np.arange(len(label))
color = 'tab:red'
ax20_t=ax[2][1].twinx()
bars=ax20_t.bar(x=bar_x+width/2,height=data_result, width=0.5, tick_label=label,color=['red'])
temp_i=0
for b in bars[1:]:
        height = round(bars[0].get_height()/b.get_height(),2)
        # ax20_t.annotate('$\\times${}'.format(height),
        #             xy=(b.get_x() + b.get_width() / 2, b.get_height()),
        #             xytext=(0, 3),  # 3 points vertical offset
        #             textcoords="offset points",color='black',
        #             ha='center', va='bottom')
        if temp_i == 0:
                ax20_t.annotate('$\\times${}'.format(height),
                                  xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                                  xytext=(0, 20),  # 3 points vertical offset
                                  textcoords="offset points", color='black',
                                  ha='center', va='bottom')
        else:
                ax20_t.annotate('$\\times${}'.format(height),
                                  xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                                  xytext=(0, 3),  # 3 points vertical offset
                                  textcoords="offset points", color='black',
                                  ha='center', va='bottom')
        temp_i += 1
ax20_t.set_ylabel('Affordability Ratio',color=color, fontsize=35)
ax20_t.set_yscale('log')
ax20_t.tick_params(axis='y', labelcolor=color, labelsize=25)
ax[2][1].set_xticks(bar_x, label)

# calculate price function in b1 d1
x0 =np.arange(10,61,2.5)
fun1 = lambda x : -np.sum(np.multiply(np.multiply(d1,x),np.array([x[i] <= b3[i] for i in range(len(v) )], dtype=int)))

res = minimize(fun1, x0, method='SLSQP', constraints=cons)
vbm_x=res.x
print('vbm最小值：',res.fun)


max_x=np.ones(21)*100
max_r=-np.sum(np.multiply(np.multiply(d1,max_x),np.array([max_x[i] <= b3[i] for i in range(len(v) )], dtype=int)))
print('maxc最小值：',max_r)

med_x=np.ones(21)*np.median(b3)
med_r=-np.sum(np.multiply(np.multiply(d1,med_x),np.array([med_x[i] <= b3[i] for i in range(len(v) )], dtype=int)))
print('medr最小值：',med_r)


ax[1][2].plot(v,vbm_x,'o-',linewidth=3.0,label='SVBP')
ax[1][2].plot(v,max_x,'^-',linewidth=3.0,label='MaxW')
ax[1][2].plot(v,med_x,'*-',linewidth=3.0,label='MedW')
ax[1][2].set_xlabel('value', fontsize=35)
ax[1][2].set_ylabel('Price', fontsize=35)
ax[1][2].tick_params(axis='x', labelsize=25)
ax[1][2].tick_params(axis='y',  labelsize=25)
# ax[1][2].legend(loc="lower right", fontsize=35)

data_result=[-res.fun,-max_r,-med_r]
label = ('SVBP', 'MaxW', 'MedW')
bar_x = np.arange(len(label))
color = 'tab:blue'
bars=ax[2][2].bar(x=bar_x-width/2,height=data_result, width=0.5, tick_label=label,color=['b'])
ax[2][2].set_yscale('log')
for b in bars[1:]:
        height = round(bars[0].get_height()/b.get_height(),2)
        ax[2][2].annotate('$\\times${}'.format(height),
                    xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",color='black',
                    ha='center', va='bottom')
ax[2][2].set_ylabel('Revenue',color=color, fontsize=35)
ax[2][2].set_xlabel('(c)', fontsize=45)
ax[2][2].tick_params(axis='y', labelcolor=color, labelsize=25)
ax[2][2].tick_params(axis='x', labelsize=25)
afford_ratio_vbm=sum(np.array([d1[i] for i in range(len(v) )  if vbm_x[i] <= b3[i]]))/sum(d1)
afford_ratio_max=sum(np.array([d1[i] for i in range(len(v) )  if max_x[i] <= b3[i]]))/sum(d1)
afford_ratio_med=sum(np.array([d1[i] for i in range(len(v) )  if med_x[i] <= b3[i]]))/sum(d1)
data_result=[afford_ratio_vbm,afford_ratio_max,afford_ratio_med]
label = ('SVBP', 'MaxW', 'MedW')
bar_x = np.arange(len(label))

color = 'tab:red'
ax20_t=ax[2][2].twinx()
bars=ax20_t.bar(x=bar_x+width/2,height=data_result, width=0.5, tick_label=label,color=['red'])
temp_i=0
for b in bars[1:]:
        height = round(bars[0].get_height()/b.get_height(),2)
        # ax20_t.annotate('$\\times${}'.format(height),
        #             xy=(b.get_x() + b.get_width() / 2, b.get_height()),
        #             xytext=(0, 3),  # 3 points vertical offset
        #             textcoords="offset points",color='black',
        #             ha='center', va='bottom')
        if temp_i == 0:
                ax20_t.annotate('$\\times${}'.format(height),
                                  xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                                  xytext=(0, 20),  # 3 points vertical offset
                                  textcoords="offset points", color='black',
                                  ha='center', va='bottom')
        else:
                ax20_t.annotate('$\\times${}'.format(height),
                                  xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                                  xytext=(0, 3),  # 3 points vertical offset
                                  textcoords="offset points", color='black',
                                  ha='center', va='bottom')
        temp_i += 1
ax20_t.set_ylabel('Affordability Ratio',color=color, fontsize=35)
ax20_t.set_yscale('log')
ax20_t.tick_params(axis='y', labelcolor=color, labelsize=25)
ax[2][2].set_xticks(bar_x, label)












ax[0][3].plot(v,b1,c='b',marker='o',linewidth=3.0)
ax[0][4].plot(v,b1,c='b',marker='o',linewidth=3.0)
color = 'tab:blue'

ax[0][3].set_ylabel('WTP',color=color, fontsize=35)
ax[0][4].set_ylabel('WTP',color=color, fontsize=35)
ax[0][3].set_xlabel('value', fontsize=35)
ax[0][4].set_xlabel('value', fontsize=35)


ax[0][3].tick_params(axis='y', labelcolor=color, labelsize=25)
ax[0][4].tick_params(axis='y', labelcolor=color, labelsize=25)
ax[0][3].tick_params(axis='x', labelsize=25)
ax[0][4].tick_params(axis='x',  labelsize=25)

color = 'tab:red'
ax00_t=ax[0][3].twinx()
ax00_t.plot(v,d2,c='r',marker='*',linewidth=3.0)
ax01_t=ax[0][4].twinx()
ax01_t.plot(v,d3,c='r',marker='*',linewidth=3.0)

ax00_t.set_ylabel('Requirement Distribution',color=color, fontsize=35)
ax01_t.set_ylabel('Requirement Distribution',color=color, fontsize=35)

ax00_t.set_xlabel('value', fontsize=35)
ax01_t.set_xlabel('value', fontsize=35)

ax00_t.tick_params(axis='y', labelcolor=color, labelsize=25)
ax01_t.tick_params(axis='y', labelcolor=color, labelsize=25)

# calculate price function in b1 d2
x0 =np.arange(10,61,2.5)

fun1 = lambda x : -np.sum(np.multiply(np.multiply(d2,x),np.array([x[i] <= b1[i] for i in range(len(v) )], dtype=int)))
cons = (
        {'type': 'ineq', 'fun': lambda x: x[0]},
        {'type': 'ineq', 'fun': lambda x: x[1]},
        {'type': 'ineq', 'fun': lambda x: x[2]},
        {'type': 'ineq', 'fun': lambda x: x[3]},
        {'type': 'ineq', 'fun': lambda x: x[4]},  # x>=e，即 x > 0
        {'type': 'ineq', 'fun': lambda x: x[5]},
        {'type': 'ineq', 'fun': lambda x: x[6]},
        {'type': 'ineq', 'fun': lambda x: x[7]},
        {'type': 'ineq', 'fun': lambda x: x[8]},  # x>=e，即 x > 0
        {'type': 'ineq', 'fun': lambda x: x[9]},
        {'type': 'ineq', 'fun': lambda x: x[10]},
        {'type': 'ineq', 'fun': lambda x: x[11]},
        {'type': 'ineq', 'fun': lambda x: x[12]},
        {'type': 'ineq', 'fun': lambda x: x[13]},  # x>=e，即 x > 0
        {'type': 'ineq', 'fun': lambda x: x[14]},
        {'type': 'ineq', 'fun': lambda x: x[15]},
        {'type': 'ineq', 'fun': lambda x: x[16]},
        {'type': 'ineq', 'fun': lambda x: x[17]},
        {'type': 'ineq', 'fun': lambda x: x[18]},  # x>=e，即 x > 0
        {'type': 'ineq', 'fun': lambda x: x[19]},
        {'type': 'ineq', 'fun': lambda x: x[20]},

        {'type': 'ineq', 'fun': lambda x: -x[0] +100},
        {'type': 'ineq', 'fun': lambda x: -x[1] +100},
        {'type': 'ineq', 'fun': lambda x: -x[2] +100},
        {'type': 'ineq', 'fun': lambda x: -x[3] +100},
        {'type': 'ineq', 'fun': lambda x: -x[4] +100},  # x>=e，即 x > 0
        {'type': 'ineq', 'fun': lambda x: -x[5] +100},
        {'type': 'ineq', 'fun': lambda x: -x[6] +100},
        {'type': 'ineq', 'fun': lambda x: -x[7] +100},
        {'type': 'ineq', 'fun': lambda x: -x[8] +100},  # x>=e，即 x > 0
        {'type': 'ineq', 'fun': lambda x: -x[9] +100},
        {'type': 'ineq', 'fun': lambda x: -x[10] +100},
        {'type': 'ineq', 'fun': lambda x: -x[11] +100},
        {'type': 'ineq', 'fun': lambda x: -x[12] +100},
        {'type': 'ineq', 'fun': lambda x: -x[13] +100},  # x>=e，即 x > 0
        {'type': 'ineq', 'fun': lambda x: -x[14] +100},
        {'type': 'ineq', 'fun': lambda x: -x[15] +100},
        {'type': 'ineq', 'fun': lambda x: -x[16] +100},
        {'type': 'ineq', 'fun': lambda x: -x[17] +100},
        {'type': 'ineq', 'fun': lambda x: -x[18] +100},  # x>=e，即 x > 0
        {'type': 'ineq', 'fun': lambda x: -x[19] +100},
        {'type': 'ineq', 'fun': lambda x: -x[20] +100},


        # {'type': 'ineq', 'fun': lambda x: x[0] - x[1]/2} ,
        {'type': 'ineq', 'fun': lambda x:  x[1]/5/1-x[2]/5/2} ,
        {'type': 'ineq', 'fun': lambda x:  x[2]/5/2-x[3]/5/3},
        {'type': 'ineq', 'fun': lambda x:  x[3]/5/3-x[4]/5/4} ,
        {'type': 'ineq', 'fun': lambda x:  x[4]/5/4-x[5]/5/5},
        {'type': 'ineq', 'fun': lambda x:  x[5]/5/5-x[6]/5/6} ,
        {'type': 'ineq', 'fun': lambda x:  x[6]/5/6-x[7]/5/7},
        {'type': 'ineq', 'fun': lambda x:  x[7]/5/7-x[8]/5/8} ,
        {'type': 'ineq', 'fun': lambda x:  x[8]/5/8-x[9]/5/9},
        {'type': 'ineq', 'fun': lambda x:  x[9]/5/9-x[10]/5/10} ,
        {'type': 'ineq', 'fun': lambda x:  x[10]/5/10-x[11]/5/11},
        {'type': 'ineq', 'fun': lambda x:  x[11]/5/11-x[12]/5/12} ,
        {'type': 'ineq', 'fun': lambda x:  x[12]/5/12-x[13]/5/13},
        {'type': 'ineq', 'fun': lambda x: x[13] /5/ 13 - x[14] /5/ 14},
        {'type': 'ineq', 'fun': lambda x: x[14] /5/ 14 - x[15] /5/ 15},
        {'type': 'ineq', 'fun': lambda x: x[15] /5/ 15 - x[16] /5/ 16},
        {'type': 'ineq', 'fun': lambda x: x[16] /5/ 16 - x[17] /5/ 17},
        {'type': 'ineq', 'fun': lambda x: x[17] /5/ 17 - x[18] /5/ 18},
        {'type': 'ineq', 'fun': lambda x: x[18] /5/ 18 - x[19] /5/ 19},
        {'type': 'ineq', 'fun': lambda x: x[19] /5/ 19 - x[20] /5/ 20},


        {'type': 'ineq', 'fun': lambda x: x[20] - x[19]},
        {'type': 'ineq', 'fun': lambda x: x[19] - x[18]},
        {'type': 'ineq', 'fun': lambda x: x[18] - x[17]},
        {'type': 'ineq', 'fun': lambda x: x[17] - x[16]},
        {'type': 'ineq', 'fun': lambda x: x[16] - x[15]},
        {'type': 'ineq', 'fun': lambda x: x[15] - x[14]},
        {'type': 'ineq', 'fun': lambda x: x[14] - x[13]},
        {'type': 'ineq', 'fun': lambda x: x[13] - x[12]},
        {'type': 'ineq', 'fun': lambda x: x[12] - x[11]},
        {'type': 'ineq', 'fun': lambda x: x[11] - x[10]},
        {'type': 'ineq', 'fun': lambda x: x[10] - x[9]},
        {'type': 'ineq', 'fun': lambda x: x[9] - x[8]},
        {'type': 'ineq', 'fun': lambda x: x[8] - x[7]},
        {'type': 'ineq', 'fun': lambda x: x[7] - x[6]},
        {'type': 'ineq', 'fun': lambda x: x[6] - x[5]},
        {'type': 'ineq', 'fun': lambda x: x[5] - x[4]},
        {'type': 'ineq', 'fun': lambda x: x[4] - x[3]},
        {'type': 'ineq', 'fun': lambda x:  x[3] -x[2] },
        {'type': 'ineq', 'fun': lambda x:  x[2] -x[1] },
        {'type': 'ineq', 'fun': lambda x:  x[1] -x[0] }
       )
 # 设置初始值

res = minimize(fun1, x0, method='SLSQP', constraints=cons)
vbm_x=res.x
print('vbm最小值：',res.fun)


max_x=np.ones(21)*100
max_r=-np.sum(np.multiply(np.multiply(d2,max_x),np.array([max_x[i] <= b1[i] for i in range(len(v) )], dtype=int)))
print('maxc最小值：',max_r)

med_x=np.ones(21)*np.median(b1)
med_r=-np.sum(np.multiply(np.multiply(d2,med_x),np.array([med_x[i] <= b1[i] for i in range(len(v) )], dtype=int)))
print('medr最小值：',med_r)

ax[1][3].plot(v,vbm_x,'o-',linewidth=3.0,label='SVBP')
ax[1][3].plot(v,max_x,'^-',linewidth=3.0,label='MaxW')
ax[1][3].plot(v,med_x,'*-',linewidth=3.0,label='MedW')
ax[1][3].set_xlabel('value', fontsize=35)
ax[1][3].set_ylabel('Price', fontsize=35)
ax[1][3].tick_params(axis='x', labelsize=25)
ax[1][3].tick_params(axis='y',  labelsize=25)
# ax[1][3].legend(loc="lower right", fontsize=35)

data_result=[-res.fun,-max_r,-med_r]
label = ('SVBP', 'MaxW', 'MedW')
bar_x = np.arange(len(label))

bars=ax[2][3].bar(x=bar_x-width/2,height=data_result, width=0.5, tick_label=label,color=['b'])

color = 'tab:blue'
ax[2][3].set_yscale('log')
for b in bars[1:]:
        height = round(bars[0].get_height()/b.get_height(),2)
        ax[2][3].annotate('$\\times${}'.format(height),
                    xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",color='black',
                    ha='center', va='bottom')
ax[2][3].set_ylabel('Revenue',color=color, fontsize=35)
ax[2][3].set_xlabel('(d)', fontsize=45)
ax[2][3].tick_params(axis='y', labelcolor=color, labelsize=25)
ax[2][3].tick_params(axis='x', labelsize=25)

afford_ratio_vbm=sum(np.array([d2[i] for i in range(len(v) )  if vbm_x[i] <= b1[i]]))/sum(d2)
afford_ratio_max=sum(np.array([d2[i] for i in range(len(v) )  if max_x[i] <= b1[i]]))/sum(d2)
afford_ratio_med=sum(np.array([d2[i] for i in range(len(v) )  if med_x[i] <= b1[i]]))/sum(d2)

data_result=[afford_ratio_vbm,afford_ratio_max,afford_ratio_med]
label = ('SVBP', 'MaxW', 'MedW')
bar_x = np.arange(len(label))
color = 'tab:red'
ax20_t=ax[2][3].twinx()
bars=ax20_t.bar(x=bar_x+width/2,height=data_result, width=0.5, tick_label=label,color=['red'])

temp_i=0
for b in bars[1:]:
        height = round(bars[0].get_height()/b.get_height(),2)
        if temp_i==0:
                ax20_t.annotate('$\\times${}'.format(height),
                            xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                            xytext=(0,20),  # 3 points vertical offset
                            textcoords="offset points",color='black',
                            ha='center', va='bottom')
        else:
                ax20_t.annotate('$\\times${}'.format(height),
                                xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points", color='black',
                                ha='center', va='bottom')
        temp_i+=1
ax20_t.set_ylabel('Affordability Ratio',color=color, fontsize=35)
ax20_t.set_yscale('log')
ax20_t.tick_params(axis='y', labelcolor=color, labelsize=25)
ax[2][3].set_xticks(bar_x, label)



x0 =np.arange(10,61,2.5)
fun1 = lambda x : -np.sum(np.multiply(np.multiply(d3,x),np.array([x[i] <= b1[i] for i in range(len(v) )], dtype=int)))

res = minimize(fun1, x0, method='SLSQP', constraints=cons)
vbm_x=res.x
print('vbm最小值：',res.fun)


max_x=np.ones(21)*100
max_r=-np.sum(np.multiply(np.multiply(d3,max_x),np.array([max_x[i] <= b1[i] for i in range(len(v) )], dtype=int)))
print('maxc最小值：',max_r)

med_x=np.ones(21)*np.median(b1)
med_r=-np.sum(np.multiply(np.multiply(d3,med_x),np.array([med_x[i] <= b1[i] for i in range(len(v) )], dtype=int)))
print('medr最小值：',med_r)


ax[1][4].plot(v,vbm_x,'o-',linewidth=3.0,label='SVBP')
ax[1][4].plot(v,max_x,'^-',linewidth=3.0,label='MaxW')
ax[1][4].plot(v,med_x,'*-',linewidth=3.0,label='MedW')
ax[1][4].set_xlabel('value', fontsize=35)
ax[1][4].set_ylabel('Price', fontsize=35)
ax[1][4].legend(loc="lower right", fontsize=35)
ax[1][4].tick_params(axis='x', labelsize=25)
ax[1][4].tick_params(axis='y',  labelsize=25)

data_result=[-res.fun,-max_r,-med_r]
label = ('SVBP', 'MaxW', 'MedW')
bar_x = np.arange(len(label))

color = 'tab:blue'
bars=ax[2][4].bar(x=bar_x-width/2,height=data_result, width=0.5, tick_label=label,color=['b'])
ax[2][4].set_yscale('log')
for b in bars[1:]:
        height = round(bars[0].get_height()/b.get_height(),2)
        ax[2][4].annotate('$\\times${}'.format(height),
                    xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",color='black',
                    ha='center', va='bottom')
ax[2][4].set_ylabel('Revenue',color=color, fontsize=35)
ax[2][4].set_xlabel('(e)', fontsize=45)
ax[2][4].tick_params(axis='y', labelcolor=color,which='both', labelsize=25)
ax[2][4].tick_params(axis='x', labelsize=25)

afford_ratio_vbm=sum(np.array([d3[i] for i in range(len(v) )  if vbm_x[i] <= b1[i]]))/sum(d3)
afford_ratio_max=sum(np.array([d3[i] for i in range(len(v) )  if max_x[i] <= b1[i]]))/sum(d3)
afford_ratio_med=sum(np.array([d3[i] for i in range(len(v) )  if med_x[i] <= b1[i]]))/sum(d3)

data_result=[afford_ratio_vbm,afford_ratio_max,afford_ratio_med]
label = ('SVBP', 'MaxW', 'MedW')
bar_x = np.arange(len(label))

color = 'tab:red'
ax20_t=ax[2][4].twinx()
bars=ax20_t.bar(x=bar_x+width/2,height=data_result, width=0.5, tick_label=label,color=['red'])
temp_i=0
for b in bars[1:]:
        height = round(bars[0].get_height()/b.get_height(),2)
        # ax20_t.annotate('$\\times${}'.format(height),
        #             xy=(b.get_x() + b.get_width() / 2, b.get_height()),
        #             xytext=(0, 3),  # 3 points vertical offset
        #             textcoords="offset points",color='black',
        #             ha='center', va='bottom')
        if temp_i == 0:
                ax20_t.annotate('$\\times${}'.format(height),
                                xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                                xytext=(0, 20),  # 3 points vertical offset
                                textcoords="offset points", color='black',
                                ha='center', va='bottom')
        else:
                ax20_t.annotate('$\\times${}'.format(height),
                                xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points", color='black',
                                ha='center', va='bottom')
        temp_i += 1
ax20_t.set_ylabel('Affordability Ratio',color=color, fontsize=35)


ax20_t.tick_params(axis='y', labelcolor=color,which='both', labelsize=25)

ax20_t.set_yscale('log',nonpositive='clip')
ax[2][4].set_xticks(bar_x, label)
fig.tight_layout()

plt.show()
fig.savefig('mk_all.eps', dpi=600, format='eps')