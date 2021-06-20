import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyecharts.charts import Map
from pyecharts.charts import Boxplot
from pyecharts import options as opts
from pyecharts.globals import ThemeType
from collections import Counter
import wordcloud
import jieba
import re
from matplotlib.font_manager import FontProperties
myfont = FontProperties(fname=r"simfang.ttf",size=12)
#plt.rcParams['font.sans-serif'] = ['FangSong']
#plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv('lagou_recruitment.csv')

df.dropna(axis = 0,how = 'any',inplace = True)

gp = df.groupby(["城市"])["岗位名称"].count()
#职业分布柱状图
plt.figure(figsize=(20,10))
flg = plt.bar(gp.index,gp.values,width=0.2,color = 'blue')
for x in flg:
    plt.text(x.get_x()+x.get_width()/2,x.get_height(),'%d'%int(x.get_height()),ha = 'center',va = 'bottom')
plt.title("各城市职业数",fontsize = 20)
plt.xlabel('城市',fontsize = 16)
plt.ylabel("职业数",fontsize = 16)
plt.tick_params(axis='both',which = 'major',labelsize = 16)
plt.savefig('各城市职业数.png')
#plt.show()
#职业分布地理图
city_job = Counter(df["城市"].dropna())
cityj = list(city_job.keys())
jobc = list(city_job.values())
a = (
    Map(init_opts=opts.InitOpts(width="1500px",height="600px"))
    .add("招聘职业数",[(cityj[i],jobc[i])for i in range(len(cityj))],"china-cities",
         label_opts=opts.LabelOpts(is_show=False)
         )
    .set_global_opts(
        title_opts=opts.TitleOpts(title="各城市职业需求数"),
        visualmap_opts=opts.VisualMapOpts(max_=max(jobc)),
    )
    .render("Map.html")
)

assert isinstance(df,object)
df['基本要求'] = df['基本要求'].str.replace('/',' ')
df[['薪水','经验','学历']] = df['基本要求'].str.split(expand = True)
df.drop('基本要求',axis = 1,inplace = True)
assert isinstance(df,object)
df[['公司类别','融资情况','公司人数']] = df['公司状况'].str.split('/',expand = True)
df.drop('公司状况',axis = 1,inplace = True)

#平均工资
def avg_salary(salary):
    min_salary = int(salary.split('-')[0][:-1])
    max_salary = int(salary.split('-')[1][:-1])
    return 1000*(min_salary+max_salary)

dfa = pd.DataFrame(df['城市'])
dfb = df['薪资'].apply(avg_salary)
dfc = dfa.assign(平均薪资 = dfb)
dfc = dfc.groupby('城市').agg({'平均薪资':'mean'})
dfc = dfc.sort_values('平均薪资',ascending = False)

avgs = list(dfc['平均薪资'])
#各城市平均薪资图
plt.figure(figsize=(20,10))
flg = plt.bar(gp.index,avgs,width=0.2,color = 'blue')
flg[0].set_color('red')
for x in flg:
    plt.text(x.get_x()+x.get_width()/2,x.get_height(),'%d'%int(x.get_height()),ha = 'center',va = 'bottom')
plt.title("各城市平均薪资",fontsize = 20)
plt.xlabel('城市',fontsize = 16)
plt.ylabel("平均薪资",fontsize = 16)
plt.tick_params(axis='both',which = 'major',labelsize = 16)
plt.savefig('各城市平均薪资.png')
#plt.show()

dfa = pd.DataFrame(df['公司类别'])
dfb = df['薪资'].apply(avg_salary)
dfc = dfa.assign(平均薪资 = dfb)
dfc = dfc.groupby('公司类别').agg({'平均薪资':'mean'})
dfc = dfc.sort_values('平均薪资',ascending = False)

company = list(dfc.index)[-11:-1]
avgs = list(dfc['平均薪资'])[-11:-1]
#不同类别公司平均薪资图
plt.figure(figsize=(30,10))
flg = plt.barh(company,avgs,color = 'blue')
flg[-10].set_color('red')
plt.title("各类公司企业平均薪资-最高前十",fontsize = 20)
plt.xlabel('平均薪资',fontsize = 16)
plt.ylabel("公司类别",fontsize = 16)
plt.tick_params(axis='both',which = 'major',labelsize = 10)
plt.savefig('各类公司企业平均薪资-前十.png')
#plt.show()

company = list(dfc.index)[0:10]
avgs = list(dfc['平均薪资'])[0:10]
plt.figure(figsize=(30,10))
flg = plt.barh(company,avgs,color = 'blue')
flg[-1].set_color('red')
plt.title("各类公司企业平均薪资-最低前十",fontsize = 20)
plt.xlabel('平均薪资',fontsize = 16)
plt.ylabel("公司类别",fontsize = 16)
plt.tick_params(axis='both',which = 'major',labelsize = 10)
plt.savefig('各类公司企业平均薪资-后十.png')
#plt.show()

dfa = pd.DataFrame(df['学历'])
dfb = df['薪资'].apply(avg_salary)
dfc = dfa.assign(平均薪资 = dfb)
dfc = dfc.groupby('学历').agg({'平均薪资':'mean'})
dfc = dfc.sort_values('平均薪资',ascending = False)
#不同学历薪资图
xueli = list(dfc.index)
avgs = list(dfc['平均薪资'])
plt.figure(figsize=(30,10))
flg = plt.bar(xueli,avgs,width = 0.5,color = 'blue')
flg[0].set_color('red')
for x in flg:
    plt.text(x.get_x()+x.get_width()/2,x.get_height(),'%d'%int(x.get_height()),ha = 'center',va = 'bottom')
plt.title("各学历平均薪资对比",fontsize = 20)
plt.xlabel('学历',fontsize = 16)
plt.ylabel("平均薪资",fontsize = 16)
plt.tick_params(axis='both',which = 'major',labelsize = 20)
plt.savefig('各学历平均薪资对比.png')
#plt.show()

dfa = pd.DataFrame(df['经验'])
dfb = df['薪资'].apply(avg_salary)
dfc = dfa.assign(平均薪资 = dfb)
dfc = dfc.groupby('经验').agg({'平均薪资':'mean'})
dfc = dfc.sort_values('平均薪资',ascending = False)
level = list(dfc.index)
avgs = list(dfc['平均薪资'])
plt.figure(figsize=(30,10))
flg = plt.bar(level,avgs,width = 0.5,color = 'blue')
flg[0].set_color('red')
for x in flg:
    plt.text(x.get_x()+x.get_width()/2,x.get_height(),'%d'%int(x.get_height()),ha = 'center',va = 'bottom')
plt.title("各工作年限平均薪资对比",fontsize = 20)
plt.xlabel('工作年限',fontsize = 16)
plt.ylabel("平均薪资",fontsize = 16)
plt.tick_params(axis='both',which = 'major',labelsize = 20)
plt.savefig('各工作年限平均薪资对比.png')
#plt.show()

level = Counter(df['经验'].dropna())
rank = list(level.keys())
levelc = list(level.values())
plt.figure(figsize=(30,10))
plt.pie(levelc,labels = rank,autopct='%1.1f%%',shadow = False,startangle = 150,radius = 100,
            textprops = {'fontsize':20}
            )
plt.legend(loc="upper right",fontsize=10,bbox_to_anchor=(1.1,1.05),ncol=2,borderaxespad=0.3)
plt.axis('equal')
plt.title("各工作经验需求比例")
plt.savefig('各工作经验需求比例.png')
#plt.show()

#职业岗位词云图
t = open("岗位名称.txt",'a')
for i in range(len(list(df['岗位名称']))):
    t.writelines(list(df['岗位名称'])[i]+" ")
t.close()
t = open("岗位名称.txt",'r')
str1 = t.read()
st = re.sub(r'-（）()/【】／，',' ',str1)
text = ' '.join(jieba.lcut(st))
WC = wordcloud.WordCloud(font_path = 'C:\\Windows\\Fonts\\STFANGSO.TTF',
                         max_words=200,max_font_size = 200,height= 400,width=400,repeat=False,
                         mode='RGBA')

con = WC.generate(text)
plt.imshow(con)
plt.axis("off")
plt.savefig('职业岗位.png')
#plt.show()
#岗位技能词云图
t = open("岗位技能.txt",'a')
for i in range(len(list(df['岗位技能']))):
    t.writelines(list(df['岗位技能'])[i]+" ")
t.close()
t = open("岗位技能.txt",'r')
str1 = t.read()
st = re.sub(r'-（）()/【】／，',' ',str1)
text = ' '.join(jieba.lcut(st))
WC = wordcloud.WordCloud(font_path = 'C:\\Windows\\Fonts\\STFANGSO.TTF',
                         max_words=200,max_font_size = 200,height= 400,width=400,repeat=False,
                         mode='RGBA')

con = WC.generate(text)
plt.imshow(con)
plt.axis("off")
plt.savefig('岗位技能.png')
#plt.show()
#公司福利词云图
t = open("公司福利.txt",'a')
for i in range(len(list(df['公司福利']))):
    t.writelines(list(df['公司福利'])[i]+" ")
t.close()
t = open("公司福利.txt",'r')
str1 = t.read()
st = re.sub(r'“”-（）()/【】／，,、；',' ',str1)
text = ' '.join(jieba.lcut(st))
WC = wordcloud.WordCloud(font_path = 'C:\\Windows\\Fonts\\STFANGSO.TTF',
                         max_words=200,max_font_size = 200,height= 400,width=400,repeat=False,
                         mode='RGBA')

con = WC.generate(text)
plt.imshow(con)
plt.axis("off")
plt.savefig('公司福利.png')
#plt.show()

city = list(gp.index)
for i in range(len(city)):
    #各地区不同地点职业数比例
    dfd = df[df['城市'].isin([city[i]])]
    gp1 = dfd.groupby('地点')['岗位名称'].count()
    plt.figure(figsize=(30, 10))
    plt.pie(gp1.values,labels = gp1.index,autopct='%1.1f%%',shadow = False,startangle = 150,radius = 100,
            textprops = {'fontsize':5}
            )
    plt.legend(loc="upper right",fontsize=10,bbox_to_anchor=(1.1,1.05),ncol=2,borderaxespad=0.3)
    plt.axis('equal')
    plt.title(city[i] + "城市各地区职业数比例")
    plt.savefig(city[i] + '城市各地区职业数比例.png')
    #plt.show()

    dfe = pd.DataFrame(df['岗位名称'])
    dff = dfd['薪资'].apply(avg_salary)
    dfg = dfe.assign(平均薪资=dff)
    dfg = dfg.groupby('岗位名称').agg({'平均薪资': 'mean'})
    dfg = dfg.sort_values('平均薪资', ascending=False)
    # 各地区前十最低薪资
    job = list(dfg.index)[0:10]
    avgs = list(dfg['平均薪资'])[0:10]
    plt.figure(figsize=(30, 10))
    flg = plt.barh(job, avgs, color='blue')
    flg[-1].set_color('red')
    plt.title(city[i]+"各类岗位平均薪资-最低前十", fontsize=20)
    plt.xlabel('平均薪资', fontsize=16)
    plt.ylabel("岗位名称", fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.savefig(city[i]+'各类岗位平均薪资-最低前十.png')
    #plt.show()

    dfg = dfg.sort_values('平均薪资', ascending=True)
    job = list(dfg.index)[0:10]
    avgs = list(dfg['平均薪资'])[0:10]
    plt.figure(figsize=(30, 10))
    flg = plt.barh(job, avgs, color='blue')
    flg[-1].set_color('red')
    plt.title(city[i] + "各类岗位平均薪资-最高前十", fontsize=20)
    plt.xlabel('平均薪资', fontsize=16)
    plt.ylabel("岗位名称", fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.savefig(city[i] + '各类岗位平均薪资-最高前十.png')
    #plt.show()
#箱线图
df['薪资']=df['薪资'].str.findall('\d+')
list1 = []
for i in df['薪资']:
    salary = [int(j)for j in i]
    avg = salary[0]+(salary[1]-salary[0])/4
    list1.append(avg)
df['每月薪资'] = list1
df['经验'] = df['经验'].replace({'经验应届毕业生': '1年以下','经验不限': '1年以下'})
groupby_workyear = df.groupby(['经验'])['每月薪资']
count_groupby_workyear = groupby_workyear.count()
count_groupby_workyear = count_groupby_workyear.reindex(['1年以下', '经验1-3年', '经验3-5年', '经验5-10年'])
a = count_groupby_workyear.index
list2 = []
for b in a:
    c = groupby_workyear.get_group(b).values
    list2.append(c)
c = Boxplot(init_opts=opts.InitOpts(theme=ThemeType.INFOGRAPHIC))
c.add_xaxis(['1年以下', '经验1-3年', '经验3-5年', '经验5-10年']).add_yaxis("薪酬k/年", c.prepare_data(list2)).set_global_opts(title_opts=opts.TitleOpts(title="不同工作经验的薪酬分布"))
c.render("不同工作经验的薪酬分布.html")
#one-hot
df['经验'] = pd.get_dummies(df['经验'])
df['公司人数'] = pd.get_dummies(df['公司人数'])
df['学历'] = pd.get_dummies(df['学历'])
df['融资情况'] = pd.get_dummies(df['融资情况'])
df['岗位技能'] = pd.get_dummies(df['岗位技能'])
print(df)
