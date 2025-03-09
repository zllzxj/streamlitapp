import streamlit as st

import joblib
import xgboost
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm 

# 添加字体文件到 Matplotlib 的字体管理器  
font_files = fm.findSystemFonts(fontpaths=['.'], fontext='ttf')  
for font_file in font_files:  
    fm.fontManager.addfont(font_file) 

# 设置中文字体支持 - 统一设置字体配置
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 
                                   'SimSun', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

with open("XGBoost.pkl", "rb") as f:
    model = joblib.load(f)
    
# 创建分类标签映射
class_labels = ['初次发作', '即愈型', '缓愈型', '间歇发作型', 
                '频发局限型', '频发泛发稀疏型', '频发泛发重型']
columns = ['D二聚体', 'PASI评分', '病程（年）', '皮损消退速度', '皮损进展情况', '补体C3', '钙', '钾', '吸烟史', '家族银屑病史', 
           '既往是否规律治疗', '是否使用生物制剂治疗', '是否口服中药治疗', '是否外用中药治疗', '特殊部位情况', '血红蛋白', '过敏体质', '过敏史', 'DLQI评分', 
           '瘙痒评分', '白蛋白', '补体C4', 'BMI指数', '中性粒细胞数', '总二氧化碳', '总免疫球蛋白E', '甘油三酯', '全血C反应蛋白', '睡眠', '中医证型', '体质分类']

st.set_page_config(    
    page_title="寻常型银屑病患者发作类型倾向预测系统",
    page_icon="⭕",
    layout="wide"
)

st.markdown('''
    <style>
        [data-testid="block-container"] {
            padding-top: 36px;
            padding-bottom: 36px;
        }
    </style>''', unsafe_allow_html=True)

st.markdown('''
    <h1 style="font-size: 20px; text-align: center; color: black; background: #008BFB; border-radius: .5rem; margin-bottom: 1rem;">
    寻常型银屑病患者发作类型倾向预测系统
    </h1>''', unsafe_allow_html=True)

SMOKE = {"无吸烟嗜好":0, "轻度":1, "中度":2, "重度":3}
SMOKE_tooltip ='''以“吸烟包年数”为度，“无吸烟嗜好”为≤1包年；“轻度吸烟”为>1包年、≤10包年；“中度吸烟”为>10包年、≤30包年；“重度吸烟”为超过30包年
吸烟包年数=每天吸烟的包数×吸烟的年数'''
BOOL = {"否":0, "是":1}
TSBW = {"无":0, "指甲":1, "生殖器":2, "外阴":3, "肛周":4, "头皮":5}
PSQK = {"逐渐消退":0, "平稳":1, "反复":2, "逐渐加重":3}
PSXT = {"7天以下":0, "7~30天":1, "30天~一年":2, "一年以上":3}
SMQK = {"正常":1, "难以入眠":2, "易醒":3, "彻夜不眠":4, "多梦":5, "早醒":6, "辅助用药":7}
ZYLX = {"血热证":1, "血燥证":2, "血瘀证":3, "脾虚湿蕴证":4, "肝气郁结证":5, "热毒炽盛证":6, "湿热蕴结证":7, "风湿痹阻证":8}
ZYTZ = {"平和质":1, "气虚质":2, "阳虚质":3, "阴虚质":4, "痰湿质":5, "湿热质":6, "瘀血质":7, "气郁质":8, "特禀质": 9}

data = {}

expand1 = st.expander("**预测参数输入**", True)
with expand1:
    col = st.columns([1, 3, 5])

col[0].markdown('''
    <div style="font-size: 20px; text-align: center; color: black;  border-bottom: 3px solid blue; margin-bottom: 1rem;">
    个人情况
    </div>''', unsafe_allow_html=True)
data["吸烟史"] = SMOKE[col[0].selectbox("吸烟史", SMOKE, help=SMOKE_tooltip)]
data["过敏体质"] = BOOL[col[0].selectbox("是否过敏体质", BOOL)]
data["过敏史"] = BOOL[col[0].selectbox("是否存在过敏史", BOOL)]
data["BMI指数"] = col[0].number_input("BMI指数$(kg/m^2)$", value=18.3, min_value=0.00, step=0.01)

col[2].markdown('''
    <div style="font-size: 20px; text-align: center; color: black;  border-bottom: 3px solid green; margin-bottom: 1rem;">
    临床资料 & 体质
    </div>''', unsafe_allow_html=True)
col1 = col[2].columns(5)

data["病程（年）"] = col1[0].number_input("病程(年)", value=18.3, min_value=0.00, step=0.01)
data["特殊部位情况"] = TSBW[col1[1].selectbox("特殊部位", TSBW)]
data["家族银屑病史"] = BOOL[col1[2].selectbox("家族银屑病史", BOOL)]
data["皮损进展情况"] = PSQK[col1[3].selectbox("皮损进展情况", PSQK)]
data["皮损消退速度"] = PSXT[col1[4].selectbox("皮损消退速度", PSXT)]

data["瘙痒评分"] = col1[0].number_input("瘙痒评分(分)", value=18.3, min_value=0.00, step=0.01)
data["PASI评分"] = col1[1].number_input("PASI评分(分)", value=18.3, min_value=0.00, step=0.01)
data["DLQI评分"] = col1[2].number_input("DLQI评分(分)", value=18.3, min_value=0.00, step=0.01)
data["睡眠"] = SMQK[col1[3].selectbox("睡眠情况", SMQK)]
data["中医证型"] = ZYLX[col1[4].selectbox("中医证型", ZYLX)]

data["既往是否规律治疗"] = BOOL[col1[0].selectbox("既往是否规律治疗", BOOL)]
data["是否使用生物制剂治疗"] = BOOL[col1[1].selectbox("是否生物制剂治疗", BOOL)]
data["是否口服中药治疗"] = BOOL[col1[2].selectbox("是否口服中药治疗", BOOL)]
data["是否外用中药治疗"] = BOOL[col1[3].selectbox("是否外用中药治疗", BOOL)]
data["中医体质类型"] = ZYTZ[col1[4].selectbox("中医体质类型", ZYTZ)]

col[1].markdown('''
    <div style="font-size: 20px; text-align: center; color: black; border-bottom: 3px solid red; margin-bottom: 1rem;">
    实验室检查
    </div>''', unsafe_allow_html=True)
col2 = col[1].columns(3)

data["血红蛋白"] = col2[0].number_input("血红蛋白(g/L)", value=18.3, min_value=0.00, step=0.01)
data["中性粒细胞数"] = col2[1].number_input("中性粒细胞数($10^9$/L)", value=18.3, min_value=0.00, step=0.01)
data["全血C反应蛋白"] = col2[2].number_input("全血C反应蛋白(mg/L)", value=18.3, min_value=0.00, step=0.01)

data["钾"] = col2[0].number_input("钾(mmol/L)", value=18.3, min_value=0.00, step=0.01)
data["钙"] = col2[1].number_input("钙(mmol/L)", value=18.3, min_value=0.00, step=0.01)
data["总二氧化碳"] = col2[2].number_input("总二氧化碳(mmol/L)", value=18.3, min_value=0.00, step=0.01)

data["白蛋白"] = col2[0].number_input("白蛋白(g/L)", value=18.3, min_value=0.00, step=0.01)
data["甘油三酯"] = col2[1].number_input("甘油三酯(mmol/L)", value=18.3, min_value=0.00, step=0.01)
data["D二聚体"] = col2[2].number_input("D二聚体(ug/ml)", value=18.3, min_value=0.00, step=0.01)

data["总免疫球蛋白E"] = col2[0].number_input("总免疫球蛋白E(IU/ml)", value=18.3, min_value=0.00, step=0.01)
data["补体C3"] = col2[1].number_input("补体C3(g/L)", value=18.3, min_value=0.00, step=0.01)
data["补体C4"] = col2[2].number_input("补体C4(g/L)", value=18.3, min_value=0.00, step=0.01)

predata = pd.DataFrame([data]) # 将预测数据转换为DataFrame
predata = predata[columns]

with expand1:
    st.dataframe(predata, use_container_width=True, hide_index=True)

explainer = shap.TreeExplainer(model) # 创建SHAP解释器
shap_values = explainer.shap_values(predata) # 计算SHAP值
predicted_class = np.argmax([np.sum(sv) + explainer.expected_value[i] for i, sv in enumerate(shap_values)]) # 获取预测的类别
predicted_label = class_labels[predicted_class]

with st.expander("**预测结果&特征重要性**", True):
    st.markdown(f'''
        <div style="font-size: 20px; text-align: center; color: black; top: 1rem; border-bottom: 3px solid black; margin-bottom: 1rem;">
        当前预测结果：{predicted_label}
        </div>''', unsafe_allow_html=True)

    col3 = st.columns([1, 3, 3, 1])

im = model.feature_importances_
features = model.feature_names_in_
importance = pd.DataFrame({"特征重要性":im})
importance.index = features
importance = importance.sort_values(by="特征重要性")
fig = plt.figure(figsize=(6, 9), dpi=200)
importance.head(15)["特征重要性"].plot(kind="barh", title="特征重要性(前15个重要特征)", color="#008BFB", ax=plt.gca(), width=0.8)
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["bottom"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.xticks([])
for i, j, n in zip(range(0, 15), importance["特征重要性"].tolist()[:15], importance.index.tolist()[:15]):
    plt.text(j, i, " "+n+"->"+str(round(j, 3)), color="#1E88E5")
col3[1].pyplot(fig, use_container_width=True)

# 创建瀑布图
sv = explainer(predata)
exp = shap.Explanation(sv.values[:,:,predicted_class], 
                  sv.base_values[:,predicted_class], 
                  data=predata.values, 
                  feature_names=predata.columns)

fig = plt.figure(figsize=(6, 9), dpi=200)
shap.plots.waterfall(exp[predicted_class], show=False, max_display=15)
plt.tight_layout()
col3[2].pyplot(fig, use_container_width=True)

st.markdown(f'''
    <div style="font-size: 20px; text-align: center; color: red; top: 1rem; margin-bottom: 1rem; border-radius: 0.5rem; border: 1px solid red; padding: 1rem;">
    <span style="font-weight: bold;">温馨提示：</span>结果仅供临床参考，详情请咨询医师！！！
    </div>''', unsafe_allow_html=True)
