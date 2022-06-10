import pandas as pd
import plotly.express as px
df=pd.read_csv("Admission_Predict.csv")

toefl_score=df["TOEFL Score"].tolist()
result=df["Chance of admit"].tolist()
fig=px.scatter(x=toefl_score,y=result)
fig.show()

import plotly.graph_objects as go
toefl_score=df["TOEFL Score"].tolist()
gre_score=df["GRE Score"].tolist()
results=df["Chance of admit"].tolist()
colors=[]
for data in results:
  if data==1:
    colors.append("green")
  else:
    colors.append("red")

fig=go.Figure(data=go.Scatter(
    x=toefl_score,
    y=gre_score,
    mode="markers",
    marker=dict(color=colors)
))
fig.show()

scores=df[["GRE Score","TOEFL Score"]]
results=df["Chance of admit"]

from sklearn.model_selection import train_test_split
score_train, score_test, results_train, results_test = train_test_split(scores, results, test_size = 0.25, random_state = 0)

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(score_train,results_train)
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=0, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False)
user_test = sc_x.transform([[user_gre_score, user_toefl_score]])     
results_pred=classifier.predict(user_test)


if result_pred[0] == 1: 
  print("This user may pass!")
else: 
  print("This user may not pass!")