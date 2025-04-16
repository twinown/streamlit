import streamlit as st
import numpy as np
import io
import altair as alt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def download(fig,filename,title):
    buf = io.BytesIO()
    fig.savefig(buf,format="png")
    buf.seek(0)
    st.download_button(
        label=f"Скачать {title or filename}",
        data = buf,
        file_name=filename,
        mime="image/png"
    )

st.title('Исследование по чаевым')

uploaded_file = st.sidebar.file_uploader('Загрузи CSV файл', type = 'csv')
@st.cache_data 
def load_data(uploaded_file):
    tips = pd.read_csv(uploaded_file)
    tips.drop('Unnamed: 0', axis=1, inplace=True)
    return tips

st.write("""
         Шаг 1. Импортируем библиотеки
         """)  
st.write("""
         Шаг 2. Прочитаем датасет в переменную `tips`
         """)  

if uploaded_file is not None:
    tips = load_data(uploaded_file)
    st.write(tips.head(5))
else:
    st.stop()

st.write("""
          Шаг 3. Создай столбец `time_order`. Заполни его случайной датой в промежутке от 2023-01-01 до 2023-01-31
         """)  

date_range = pd.date_range(start='2023-01-01', end='2023-01-31')
tips['time_order'] = np.random.choice(date_range, size=len(tips))


st.write("""
          Шаг 4. Построй график показывающий динамику чаевых во времени
         """)
df = tips[['tip','time_order']].copy()
df = df.sort_values('time_order')
group_day_t = df.groupby(pd.Grouper(key='time_order', freq='D'))['tip'].sum().reset_index()
plt.figure(figsize=(12, 6))
sns.lineplot(
    x='time_order',
    y='tip',
    data=group_day_t,
    color='blue')

plt.title('Sum of tips by day')
plt.xlabel('Date')
plt.ylabel('Sum of tips')

plt.tight_layout()
st.pyplot(plt)
download(plt,"рис1","График 1")

st.write("Сделай аналогичный график используя многофункциональный метод [relplot]")

# График через relplot
r = sns.relplot(
    data=group_day_t,
    x='time_order',
    y='tip',
    kind='line',  
    aspect=2,     # Соотношение сторон 
    color='blue'   
)

r.set(
    title='Sum of tips by day',
    xlabel='Date',
    ylabel='Sum of tips'
)
plt.tight_layout()
st.pyplot(r)
download(plt,"рис2","График 1")

st.write("""
          Шаг 5. Нарисуйте гистограмму `total_bill`
         """)

plt.figure(figsize=(12, 6))
sns.histplot(
    data=tips,  
    x='total_bill',
    weights = 'total_bill',
    color='red',
    bins=20,          #количество столбцов
    edgecolor='white'  #границы столбцов
)

plt.title('Total bill')
plt.xlabel('sum bill')
plt.ylabel('Total bill')
plt.tight_layout()
st.pyplot(plt)
download(plt,"рис1","График 2")

st.write("Сделай аналогичный график используя многофункциональный метод [displot]Поиграйся с другими формами отображения меняя параметр параметр `kind`")

rr = sns.displot(
    data=tips,
    x='total_bill',
    weights='total_bill',  # Сумма
    kind = 'kde',
    color='red',
    # bins=20,
    # edgecolor='white',
    height=6,             
    aspect=2              
)

rr.set(
    title='Total bill KDE',
    xlabel='sum bill',
    ylabel='Total bill'
)
st.pyplot(rr)
download(plt,"рис2","График 2")

st.write("""
          Шаг 6. Нарисуйте scatterplot, показывающий связь между `total_bill` and `tip`
         """)

six_df = tips[['total_bill','tip']].copy()

chart = alt.Chart(six_df).mark_circle(
    color='royalblue',
    size=88,
    stroke='white'
).encode(
    x='total_bill:Q',
    y='tip:Q'
).properties(
    title='Scatterplot',
    width=800,
    height=400
)
st.altair_chart(chart, use_container_width=True)
download(plt,"рис1","График 3")

st.write("""
          Сделай аналогичный график используя многофункциональный метод [relplot](https://seaborn.pydata.org/generated/seaborn.relplot.html)
         """)

# Построение через relplot
rrr = sns.relplot(
    data=six_df,
    x='total_bill',
    y='tip',
    kind='scatter',  
    color='royalblue',
    s=88,           
    edgecolor='white',
    height=6,        
    aspect=2      
)


rrr.set(
    title='Scatterplot',
    xlabel='Total bill',
    ylabel='Tips'
)

plt.tight_layout()
st.pyplot(rrr)
download(plt,"рис2","График 3")

st.write("""
          Шаг 7. Нарисуйте 1 график, связывающий `total_bill`, `tip`, и `size`
   Подсказка: это одна функция
         """)

seven_df = tips[['total_bill','tip','size']].copy()
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=seven_df,
    x='total_bill',  
    y='tip',         
    size='size',     # Размер точек - size
    hue='size',      # Цвет точек - size
    palette='turbo',  # Цветовая схема
    sizes=(20, 200), # Диапазон размеров точек
)

plt.title('Connect of total_bill, tip, and size')
plt.xlabel('total_bill')
plt.ylabel('tips')
plt.legend(title='Size')
plt.tight_layout()
st.pyplot(plt)
download(plt,"рис1","График 4")

st.write("""
          Шаг 8. Покажите связь между днем недели и размером счета
         """)

eight_df = tips[['total_bill','day']].copy()
plt.figure(figsize=(5, 6))
sns.barplot(
    data=eight_df,
    x='day',
    y='total_bill',
    estimator='sum',  # сумма total_bill для каждого дня
    color='red',
    edgecolor='black'
)

plt.title('bill/day')
plt.xlabel('days')
plt.ylabel('Total bill')
plt.tight_layout()
st.pyplot(plt)
download(plt,"рис1","График 6")

st.write("""
          Шаг 9. Нарисуйте `scatter plot` с днем недели по оси **Y**, чаевыми по оси **X**, и цветом по полу
         """)

nine_df = tips[['day','tip','sex']].copy()
chart = alt.Chart(nine_df).mark_circle(
    size=88,
    stroke='black'
).encode(
    x='tip:Q',
    y='day:O',
    color=alt.Color('sex:N', scale=alt.Scale(range=['red', 'green'])),
    tooltip=['tip', 'day', 'sex']
).properties(
    title='tip/day/sex',
    width=800,
    height=400
)
st.altair_chart(chart, use_container_width=True)
download(plt,"рис1","График 7")

st.write("""
          Шаг 10. Нарисуйте `box plot` c суммой всех счетов за каждый день, разбивая по `time` (Dinner/Lunch)
Как понимать boxplot? https://tidydata.ru/boxplot
         """)

ten_df = tips[['total_bill','day','time']].copy()
plt.figure(figsize=(5, 6))
sns.boxplot(
     data=ten_df,
    x='day',
    y='total_bill',
    hue='time',     
    palette={'Lunch': 'orange', 'Dinner': 'red'},
    width=0.6,       
    linewidth=1,     
    fliersize=4
)

plt.title('bill/day')
plt.xlabel('days')
plt.ylabel('Total bill')
plt.tight_layout()
st.pyplot(plt)
download(plt,"рис1","График 8")

st.write("Построй аналогичный график используя многофункциональный метод [catplot](https://seaborn.pydata.org/generated/seaborn.catplot.html#seaborn.catplot) Поиграйся с другими формами отображения меняя параметр параметр `kind`")

s = sns.catplot(
    data=ten_df,
    x='day',
    y='total_bill',
    hue='time',
    kind='box',
    palette={'Lunch': 'orange', 'Dinner': 'red'},
    height=6,      
    aspect=5/6,  
    legend_out=False 
)

s.set(
    title='bill/day',
    xlabel='days',
    ylabel='Total bill'
)
plt.legend(loc='upper right')
plt.tight_layout()
st.pyplot(s)
download(plt,"рис2","График 8")

st.write("""
          Шаг 11. Нарисуйте 2 гистограммы чаевых на обед и ланч. Расположите их рядом по горизонтали.
         """)

el_df = tips[['tip','time']].copy()
el_df_d = el_df[el_df['time'] == 'Dinner']
el_df_l = el_df[el_df['time'] == 'Lunch']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

sns.histplot(data=el_df_d, x='tip', bins=10, color='blue', ax=ax1)
ax1.set_title('Dinner')
ax1.set_xlabel('tips')
ax1.set_ylabel('quant')

sns.histplot(data=el_df_l, x='tip', bins=10, color='red', ax=ax2)
ax2.set_title('Lunch')
ax2.set_xlabel('tips')
ax2.set_ylabel('quant')

plt.tight_layout()
st.pyplot(fig)
download(fig, "рис1", "График 9")

st.write("""
          Шаг 12. Нарисуйте 2 scatterplots (для мужчин и женщин), показав связь размера счета и чаевых, дополнительно разбив по курящим/некурящим. Расположите их по горизонтали.
         """)

tw_df = tips[['total_bill','tip','smoker','sex']].copy()
tw_df_m = tw_df[tw_df['sex']== 'Male']
tw_df_f = tw_df[tw_df['sex']== 'Female']

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.scatterplot(
    data=tw_df_m,
    x='total_bill',
    y='tip',
    hue='smoker',           
    palette={'Yes': 'red', 'No': 'blue'},
    s = 50                    # размер точек
)
plt.title('Male')
plt.xlabel('bill')
plt.ylabel('tip')
plt.legend(title='smoker')


plt.subplot(1, 2, 2)
sns.scatterplot(
    data=tw_df_f,
    x='total_bill',
    y='tip',
    hue='smoker',           
    palette={'Yes': 'red', 'No': 'blue'},
    s = 50  
)
plt.title('Female')
plt.xlabel('bill')
plt.ylabel('tip')
plt.legend(title='smoker')
st.pyplot(plt)
download(plt,"рис1","График 10")

st.write("""
          Шаг 13. Построй тепловую карту зависимостей численных переменных
Матрица корреляций в pandas - https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html  
Добавь отображение чисел в каждой ячейке
         """)

 # Близко к +1: Сильная прямая зависимость
 # Близко к -1: Сильная обратная зависимость.
 # Около 0: Слабая или отсутствующая зависимость.
warm_df = tips[['total_bill','tip','size']].copy()
warm_df
corr_matrix = warm_df.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(
    corr_matrix,
    annot=True,       # числа в ячейках
    cmap='viridis',   # цветовая схема
    vmin=-1,          # минимум для шкалы
    vmax=1,           # максимум для шкалы
    linewidths=0.5,   # толщина линий между ячейками
    square=True       # квадратные ячейки
)

plt.title("heatmap for numeric val")
st.pyplot(plt)
download(plt,"рис1","График 11")

st.write("""
          БОНУС: Задайте свой вопрос и ответьте на него с помощью графика.
         """)

st.write("Кто оставляет больше денег муж или жен?")

bonus = tips[['total_bill','sex']].copy()
total_sex = bonus.groupby('sex')['total_bill'].sum().reset_index()

plt.figure(figsize=(8, 4))
sns.barplot(
    data=total_sex,
    x='sex',
    y='total_bill',
    hue = 'sex',
    palette={'Male': 'skyblue', 'Female': 'salmon'},
    edgecolor='black'
)
plt.title("total male/female bill")
plt.xlabel("sex")
plt.ylabel("total bill")
st.pyplot(plt)
download(plt,"рис1","График 12")