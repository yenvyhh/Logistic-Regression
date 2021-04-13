# Projekt 1 - Logistic Regression

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/yenvyhh/Logistic-Regression/main?filepath=Logistische%20Regression%20-%20Projekt%201.ipynb)


**Die Daten importieren,als DataFrame abspeichern und das Head anzeigen lassen:**
ad_data=pd.DataFrame(pd.read_csv("Advertising.csv"))
ad_data.head()

**Informationen und Details des Data Frames bzw. der Daten anzeigen lassen:**
ad_data.info()
ad_data.describe()
**Bei Info wird angezeigt, ob die Spalten einen Float, ein Integer oder ein Object sind. Bei Describe wird ein Dataset der Analyse geprintet. Beispiele hierfür sind der Durchschnittswert, der Minimum- oder Maximum-Wert.**

**Darauffolgend erfolgt eine EXPLORATIVE DATENANALYSE, die durch verschiedene Diagrammvisualisierungen dargestellt werden. Ein Beispiel, das ausgeführt wird:**
sns.distplot(ad_data["Age"],kde=False,bins=30) 
**Durch Ausführen der Funktion wird ein Histogramm (Balkendiagramm) erstellt. Es zeigt also eine Verteilung für das Alter an(Spalte "Age").

**Im nächsten Schritt wird die LOGISTISCHE REGRESSION durchgeführt. Dazu ist es notwendig, dass die Daten in Trainings- und Test gesplittet werden. Dazu sollte zunächst definiert werden, was das X-Array (Daten mit den Features) und was das y-Array (Daten mit der Zielvariable) ist:**
X=ad_data[['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage', 'Male']]
y=ad_data["Clicked on Ad"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

**NachErstellung der Train- und Testdaten wird das logistische Trainigsmodell trainiert und auf das Trainingsset gefittet:**
logmodel=LogisticRegression()
logmodel.fit(X_train,y_train)

**Im Anschluss daran werden die Werte mit**
predictions=logmodel.predict(X_test)
**vorhergesagt. Basierend darauf kann ein Klassifizierungsreport für das Modell ausgewertet werden:**
print(classification_report(y_test,predictions))
**Das Classification Report sollte jetzt zusehen sein. Je näher die Werte bei precicion, recall und f1-score an 1 sind, desto genauer sind Auswertung. 


