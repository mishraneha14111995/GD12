
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

st.set_page_config(page_title='Skincare Influencer Dashboard', layout='wide')

@st.cache_data
def load_data():
    return pd.read_csv("synthetic_skincare_dataset_final.csv")

df = load_data()

st.sidebar.title("Navigation")
tabs = ["Data Visualization", "Classification", "Clustering", "Association Rule Mining", "Regression"]
tab = st.sidebar.radio("Select a tab", tabs)

# Helper for classification
def preprocess_classification(df):
    data = df.copy()
    le = LabelEncoder()
    drop_cols = ["Favorite Influencer or Brand", "Purchase Influencers", "Skin Concerns",
                 "Purchase Discouragement Factors", "Skincare Products Used"]
    for col in data.columns:
        if data[col].dtype == 'object' and col not in drop_cols and col != "Purchased After Influencer Promotion":
            data[col] = le.fit_transform(data[col].astype(str))
    data = data.dropna(subset=["Purchased After Influencer Promotion"])
    X = data.drop(columns=["Purchased After Influencer Promotion"])
    y = le.fit_transform(data["Purchased After Influencer Promotion"])
    X = X.apply(lambda col: pd.to_numeric(col, errors='coerce')).fillna(0)
    return train_test_split(X, y, test_size=0.3, random_state=42)

# Helper for regression
def preprocess_regression(df, target):
    data = df.copy()
    le = LabelEncoder()
    drop_cols = ["Favorite Influencer or Brand", "Purchase Influencers", "Skin Concerns",
                 "Purchase Discouragement Factors", "Skincare Products Used"]
    for col in data.columns:
        if data[col].dtype == 'object' and col not in drop_cols and col != target:
            data[col] = le.fit_transform(data[col].astype(str))
    data = data.dropna(subset=[target])
    X = data.drop(columns=[target])
    y = pd.to_numeric(data[target], errors='coerce').fillna(0)
    X = X.apply(lambda col: pd.to_numeric(col, errors='coerce')).fillna(0)
    return train_test_split(X, y, test_size=0.3, random_state=42)

# Data Visualization
if tab == "Data Visualization":
    st.title("📊 Data Visualization")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Gender Distribution")
        st.plotly_chart(px.pie(df, names="Gender", title="Gender Distribution"))
    with col2:
        st.subheader("Income vs Gender")
        st.plotly_chart(px.histogram(df, x="Income (AED)", color="Gender", barmode="group"))

    st.subheader("Spending vs Age")
    st.plotly_chart(px.scatter(df, x="Age", y="Average Monthly Spend (AED)", color="Gender"))

    st.subheader("Product Preferences")
    products = pd.Series(sum([x.split(", ") for x in df["Skincare Products Used"]], [])).value_counts()
    st.bar_chart(products.head(10))

    st.subheader("Top Skin Concerns")
    concerns = pd.Series(sum([x.split(", ") for x in df["Skin Concerns"]], [])).value_counts()
    st.bar_chart(concerns.head(10))

    st.subheader("Follow Frequency")
    st.plotly_chart(px.histogram(df, x="Follow Influencer Recommendations", color="Gender"))

    st.subheader("Platform Preference")
    st.plotly_chart(px.histogram(df, x="Preferred Platform", color="Gender"))

    st.subheader("Trust Score Distribution")
    st.plotly_chart(px.histogram(df, x="Influencer Marketing Trust Score (1–5)", nbins=5))

    st.subheader("Monthly Spend Distribution")
    st.plotly_chart(px.box(df, y="Average Monthly Spend (AED)", points="all"))

    st.subheader("Spending by Employment")
    st.plotly_chart(px.box(df, x="Employment Status", y="Average Monthly Spend (AED)", color="Employment Status"))

# Classification Tab
elif tab == "Classification":
    st.title("🤖 Classification")
    X_train, X_test, y_train, y_test = preprocess_classification(df)

    models = {
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier()
    }

    scores = []
    y_preds = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_preds[name] = y_pred
        scores.append({
            "Model": name,
            "Train Acc": model.score(X_train, y_train),
            "Test Acc": model.score(X_test, y_test),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred)
        })

    st.dataframe(pd.DataFrame(scores))

    selected_model = st.selectbox("Select model for Confusion Matrix", list(models.keys()))
    if selected_model:
        cm = confusion_matrix(y_test, y_preds[selected_model])
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    st.subheader("ROC Curve")
    fig, ax = plt.subplots()
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, probs)
            ax.plot(fpr, tpr, label=f"{name} AUC={auc(fpr,tpr):.2f}")
    ax.plot([0,1],[0,1],'k--')
    ax.legend()
    st.pyplot(fig)

# Clustering Tab
elif tab == "Clustering":
    st.title("🔗 Clustering")
    features = ["Age", "Average Monthly Spend (AED)", "Influencer Marketing Trust Score (1–5)"]
    X = df[features].copy()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(X)

    st.subheader("Elbow Chart")
    distortions = []
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=42).fit(scaled)
        distortions.append(kmeans.inertia_)
    fig, ax = plt.subplots()
    ax.plot(range(2, 11), distortions, marker='o')
    ax.set_xlabel("K")
    ax.set_ylabel("Inertia")
    st.pyplot(fig)

    k = st.slider("Number of clusters", 2, 10, 3)
    model = KMeans(n_clusters=k, random_state=42)
    df["Cluster"] = model.fit_predict(scaled)
    st.dataframe(df.groupby("Cluster")[features].mean())
    st.download_button("Download Clustered Data", df.to_csv(index=False), "clustered_data.csv")

# Association Rule Mining
elif tab == "Association Rule Mining":
    st.title("🔁 Association Rules")
    basket = [row["Purchase Influencers"].split(", ") + row["Skin Concerns"].split(", ") for _, row in df.iterrows()]
    te = TransactionEncoder()
    te_ary = te.fit(basket).transform(basket)
    df_tf = pd.DataFrame(te_ary, columns=te.columns_)
    freq = apriori(df_tf, min_support=0.05, use_colnames=True)
    rules = association_rules(freq, metric="confidence", min_threshold=0.3)
    st.dataframe(rules.sort_values("confidence", ascending=False).head(10))

# Regression
elif tab == "Regression":
    st.title("📈 Regression")
    X_train, X_test, y_train, y_test = preprocess_regression(df, "Average Monthly Spend (AED)")
    models = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "Decision Tree": DecisionTreeRegressor()
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        st.subheader(name)
        fig, ax = plt.subplots()
        ax.scatter(y_test, preds)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        st.pyplot(fig)
