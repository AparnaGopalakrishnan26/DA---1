import pandas as pd, numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import plotly.express as px
import io

def load_data(path):
    return pd.read_csv(path)

def describe_columns(cols):
    # static mapping using common names (works with slight name variations)
    mapping = [
        ("ID", "unique identifier"),
        ("Personal Loan", "did the customer accept the personal loan offered (1=Yes, 0=No)"),
        ("Age", "customer’s age"),
        ("Experience", "years of professional experience"),
        ("Income", "annual income of the customer ($000)"),
        ("ZIP Code / Zip code", "home address zip code"),
        ("Family", "family size of customer"),
        ("CCAvg", "avg spending on credit cards per month ($000)"),
        ("Education", "education level (1) undergrad, (2) grad, (3) advanced)"),
        ("Mortgage", "value of house mortgage ($000)"),
        ("Securities Account", "(1=Yes, 0=No)"),
        ("CD Account", "(1=Yes, 0=No)"),
        ("Online", "(1=Yes, 0=No)"),
        ("CreditCard", "(1=Yes, 0=No)")
    ]
    return pd.DataFrame(mapping, columns=["Column","Description"])

# ------------------ Explore charts ------------------
def plot_explore_charts(df, container=st):
    # prepare a cleaned copy
    d = df.copy()
    # Try to map zip variations
    zip_col = None
    for c in d.columns:
        if c.lower().replace(" ","") in ("zipcode","zip","zipcode:","zip code"):
            zip_col = c
            d = d.rename(columns={c: "ZIP Code"})
            break
    # rename common alt names
    for alt in ["Securities", "Securities Account"]:
        if alt in d.columns and "Securities Account" not in d.columns:
            d = d.rename(columns={alt:"Securities Account"})
    for alt in ["CDAccount","CD Account","CD Account "]:
        if alt in d.columns and "CD Account" not in d.columns:
            d = d.rename(columns={alt:"CD Account"})
    # 1. Income distribution by Personal Loan acceptance (violin-like via box+strip)
    container.subheader("1) Income distribution by Personal Loan (action: target higher-income segments)")
    if 'Income' in d.columns and 'Personal Loan' in d.columns:
        fig = px.box(d, x='Personal Loan', y='Income', points="all", labels={"Personal Loan":"Personal Loan (0=No,1=Yes)"})
        container.plotly_chart(fig, use_container_width=True)
        container.markdown("- **Insight**: Compare median income for accepted vs declined. If accepted customers are notably higher income, prioritize high-income segments with premium offers.")
    # 2. Acceptance rate by Education and Family (clustered bar)
    container.subheader("2) Acceptance rate by Education and Family (action: segment messaging)")
    if 'Education' in d.columns and 'Family' in d.columns and 'Personal Loan' in d.columns:
        pivot = d.groupby(['Education','Family'])['Personal Loan'].mean().reset_index()
        fig = px.bar(pivot, x='Education', y='Personal Loan', color='Family', barmode='group', labels={'Personal Loan':'Acceptance Rate'})
        container.plotly_chart(fig, use_container_width=True)
        container.markdown("- **Insight**: Identify education-family combinations with higher acceptance to build targeted campaigns.")
    # 3. Income vs CCAvg heatmap (binned) - high spend & income target
    container.subheader("3) Income vs CCAvg binned heatmap (action: prioritize high spenders)")
    if 'Income' in d.columns and 'CCAvg' in d.columns:
        nbins_x = 6; nbins_y=6
        xbins = pd.cut(d['Income'], bins=nbins_x)
        ybins = pd.cut(d['CCAvg'], bins=nbins_y)
        heat = d.groupby([xbins,ybins])['Personal Loan'].mean().unstack(fill_value=0)
        # plot as heatmap with annotations using matplotlib inside streamlit
        fig, ax = plt.subplots(figsize=(8,4))
        im = ax.imshow(heat.values, aspect='auto', origin='lower')
        ax.set_xticks(range(heat.shape[1])); ax.set_yticks(range(heat.shape[0]))
        ax.set_xticklabels([f"{interval.left:.0f}-{interval.right:.0f}" for interval in heat.columns], rotation=45, fontsize=8)
        ax.set_yticklabels([f"{interval.left:.0f}-{interval.right:.0f}" for interval in heat.index], fontsize=8)
        ax.set_title("Acceptance rate (binned Income vs CCAvg)")
        for i in range(heat.shape[0]):
            for j in range(heat.shape[1]):
                ax.text(j,i, f"{heat.values[i,j]:.2f}", ha='center', va='center', color='white' if heat.values[i,j]>0.2 else 'black', fontsize=8)
        fig.colorbar(im, ax=ax)
        container.pyplot(fig, use_container_width=True)
        container.markdown("- **Insight**: Identify bins with higher acceptance (e.g., high Income & high CCAvg) for premium campaign targeting.")
    # 4. Propensity decile lift chart (action: prioritize top deciles)
    container.subheader("4) Propensity decile lift chart (action: use probability-ranked targeting)")
    if 'Personal Loan' in d.columns and 'Income' in d.columns:
        # create simple propensity by logistic-like score using Income & Education proxies for demo
        score = (d.get('Income',0).fillna(0).rank(pct=True)*0.6 + (d.get('CCAvg',0).fillna(0).rank(pct=True)*0.4))
        temp = d.copy(); temp['score']=score; temp['decile']=pd.qcut(temp['score'], 10, labels=False)+1
        lift = temp.groupby('decile')['Personal Loan'].mean().reset_index().sort_values('decile', ascending=False)
        fig = px.bar(lift, x='decile', y='Personal Loan', labels={'Personal Loan':'Acceptance rate', 'decile':'Propensity decile (1=low,10=high)'})
        container.plotly_chart(fig, use_container_width=True)
        container.markdown("- **Insight**: Focus sales outreach on top deciles — gives highest conversion per contact.")
    # 5. Channel & product cross-sell analysis (CD Account / Securities / CreditCard)
    container.subheader("5) Cross-sell analysis: CD Account & Securities vs Acceptance (action: bundle offers)")
    cols = [c for c in ['CD Account','Securities Account','CreditCard'] if c in d.columns]
    if cols and 'Personal Loan' in d.columns:
        cds = d.groupby(cols)['Personal Loan'].mean().reset_index().sort_values('Personal Loan', ascending=False)
        container.dataframe(cds.head(20))
        container.markdown("- **Insight**: If customers with CD Account and Securities show higher acceptance, design bundled offers to these customers.")
    container.markdown("---")
    container.caption("These five charts combine to produce operational actions: prioritize high-income/high-CCAvg segments, target top propensity deciles, and bundle offers for customers holding other bank products.")

# ------------------ Modeling ------------------
def train_and_evaluate_models(df, container=st):
    d = df.copy()
    # Basic preprocessing - drop ID & ZIP if present & ensure numeric columns exist
    drop_cols = [c for c in d.columns if c.lower() in ('id','zip','zip code','zip_code')]
    d = d.drop(columns=drop_cols, errors='ignore')
    # Rename columns to match expected names (common variants)
    rename_map = {}
    for c in d.columns:
        if c.strip().lower() == 'cdaccount' or c.strip().lower()=='cd account':
            rename_map[c] = 'CD Account'
        if c.strip().lower() == 'securities' or c.strip().lower()=='securities account':
            rename_map[c] = 'Securities Account'
    if rename_map:
        d = d.rename(columns=rename_map)
    # Ensure target exists
    if 'Personal Loan' not in d.columns:
        container.error("Dataset must contain 'Personal Loan' column for supervised training.")
        return None
    X = d.drop(columns=['Personal Loan'], errors='ignore')
    # encode categorical variables if any small number unique values
    for c in X.select_dtypes(include=['object']).columns:
        X[c] = X[c].astype('category').cat.codes
    y = d['Personal Loan'].astype(int)
    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
    # models
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=42)
    }
    metrics_rows = []
    probas = {}
    preds_train = {}
    preds_test = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_tr_pred = model.predict(X_train)
        y_te_pred = model.predict(X_test)
        # proba
        if hasattr(model, "predict_proba"):
            y_tr_proba = model.predict_proba(X_train)[:,1]
            y_te_proba = model.predict_proba(X_test)[:,1]
        else:
            y_tr_proba = model.decision_function(X_train)
            y_te_proba = model.decision_function(X_test)
        probas[name] = (y_tr_proba, y_te_proba)
        preds_train[name] = y_tr_pred
        preds_test[name] = y_te_pred
        metrics_rows.append({
            "Algorithm": name,
            "Training Accuracy": accuracy_score(y_train, y_tr_pred),
            "Testing Accuracy": accuracy_score(y_test, y_te_pred),
            "Precision": precision_score(y_test, y_te_pred, zero_division=0),
            "Recall": recall_score(y_test, y_te_pred, zero_division=0),
            "F1-Score": f1_score(y_test, y_te_pred, zero_division=0),
            "AUC": roc_auc_score(y_test, y_te_proba)
        })
    metrics_df = pd.DataFrame(metrics_rows).set_index('Algorithm').round(4)
    container.subheader("Metrics table")
    container.dataframe(metrics_df)
    # CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_rows = []
    for name, model in models.items():
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        cv_rows.append({"Algorithm":name, "CV Accuracy Mean": cv_scores.mean(), "CV Accuracy Std": cv_scores.std()})
    cv_df = pd.DataFrame(cv_rows).set_index('Algorithm').round(4)
    container.subheader("5-fold CV accuracy")
    container.dataframe(cv_df)
    # ROC combined
    container.subheader("ROC curves (all models)")
    fig, ax = plt.subplots(figsize=(7,5))
    for name in models.keys():
        fpr, tpr, _ = roc_curve(y_test, probas[name][1])
        auc_val = roc_auc_score(y_test, probas[name][1])
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})")
    ax.plot([0,1],[0,1], linestyle='--', color='grey')
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate"); ax.set_title("ROC Curves - All Models")
    ax.legend(loc='lower right')
    container.pyplot(fig)
    # Confusion matrices (train & test)
    container.subheader("Confusion matrices (train & test)")
    for dataset_label, y_true, preds in [("Train", y_train, preds_train), ("Test", y_test, preds_test)]:
        cols = st.columns(len(models))
        for i, name in enumerate(models.keys()):
            cm = confusion_matrix(y_true, preds[name])
            ax = plt.figure(figsize=(3,3))
            fig = ax.subplots()
            im = fig.imshow(cm, interpolation='nearest')
            fig.set_title(f"{name} - {dataset_label}")
            fig.set_xlabel("Predicted"); fig.set_ylabel("True")
            # annotate
            for ii in range(cm.shape[0]):
                for jj in range(cm.shape[1]):
                    fig.text(jj, ii, str(cm[ii,jj]), ha='center', va='center', color='white' if cm[ii,jj]>cm.max()/2 else 'black')
            cols[i].pyplot(ax)
    # Feature importances
    container.subheader("Feature importances")
    for name, model in models.items():
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            features = X.columns.tolist()
            ord_idx = np.argsort(importances)[::-1]
            imp_df = pd.DataFrame({"feature":[features[i] for i in ord_idx], "importance":importances[ord_idx]})
            container.markdown(f"**{name}**")
            container.dataframe(imp_df.head(12))
            fig = px.bar(imp_df.head(12), x='feature', y='importance', labels={'importance':'Importance','feature':'Feature'})
            container.plotly_chart(fig, use_container_width=True)
    return {"metrics_df":metrics_df, "cv_df":cv_df}

# ------------------ Prediction ------------------
def predict_and_download_df(newdf, reference_df):
    # reference_df is used to train a quick model if user didn't train separately
    dref = reference_df.copy()
    # basic preprocess mirror used in training
    drop_cols = [c for c in dref.columns if c.lower() in ('id','zip','zip code','zip_code')]
    dref = dref.drop(columns=drop_cols, errors='ignore')
    Xref = dref.drop(columns=['Personal Loan'], errors='ignore')
    for c in Xref.select_dtypes(include=['object']).columns:
        Xref[c] = Xref[c].astype('category').cat.codes
    yref = dref['Personal Loan'].astype(int)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(Xref, yref)
    # Prepare newdf similar to training Xref columns
    new = newdf.copy()
    new = new.drop(columns=[c for c in new.columns if c.lower() in ('id','zip','zip code','zip_code')], errors='ignore')
    # align columns: add missing with zeros, drop extras
    for c in Xref.columns:
        if c not in new.columns:
            new[c] = 0
    new = new[Xref.columns]
    for c in new.select_dtypes(include=['object']).columns:
        new[c] = new[c].astype('category').cat.codes
    probs = model.predict_proba(new)[:,1]
    preds = (probs >= 0.5).astype(int)
    out = newdf.copy()
    out['Predicted Probability'] = probs
    out['Predicted Personal Loan'] = preds
    return out

# local sklearn imports used in function
from sklearn.ensemble import RandomForestClassifier
