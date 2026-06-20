"""
pages/04_Segmentation.py  —  Layer 4: Population Segmentation
Visual design: HRV Design System (utils/styles.py)
"""
import warnings; warnings.filterwarnings("ignore")
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score, davies_bouldin_score
import scipy.cluster.hierarchy as sch

from utils.data_processing import (HRV_COLUMNS, get_outliers_iqr, get_outliers_zscore,
    STRESS_LEVEL_ORDER, STRESS_LEVEL_COLORS, AGE_GROUP_ORDER, BMI_CATEGORY_ORDER)
from utils.styles import (inject_css, page_header, section_header, insight_panel,
    apply_chart_style, divider, metric_row, COLORS, CHART_SEQ, CLUSTER_COLORS)

st.set_page_config(page_title="Segmentation · HRV", page_icon="🔍", layout="wide")
inject_css()

if "df" not in st.session_state:
    st.warning("No data loaded — upload your CSV on the Home page first.")
    st.stop()

df = st.session_state["df"]
seg_features = [c for c in ["Age","BMI","Heart_Rate_bpm"]+HRV_COLUMNS+["Stress_Score"] if c in df.columns]
extra = [c for c in ["Patient_Name","Clinic_Name","Sex","Age_Group","BMI_Category",
                      "Stress_Level","HRV_Health_Index","Parasympathetic_Activity_Index",
                      "Sympathetic_Dominance_Index"] if c in df.columns]
df_seg = df[seg_features+extra].dropna(subset=seg_features).copy()
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(df_seg[seg_features])

page_header("Population Segmentation & Clinical Discovery",
            "Outlier detection · PCA · K-Means · Hierarchical · DBSCAN")
st.caption(f"Segmentation corpus: {len(df_seg):,} records × {len(seg_features)} features")

# ── A. Outlier detection ──────────────────────────────────────────────────────
section_header("A · Outlier Detection")
hrv_p = [c for c in HRV_COLUMNS if c in df_seg.columns]
tab_iq, tab_z, tab_iso = st.tabs(["IQR Method","Z-Score Method","Isolation Forest"])
disp = [c for c in ["Patient_Name","Age","Sex","Clinic_Name","Stress_Score"]+hrv_p if c in df_seg.columns]

with tab_iq:
    insight_panel("Flags values below Q1 − 1.5×IQR or above Q3 + 1.5×IQR. Robust to skewed HRV distributions.", kind="info")
    mask, info = get_outliers_iqr(df_seg, hrv_p)
    metric_row([("IQR Outliers", f"{mask.sum():,}", ""), ("Outlier %", f"{mask.mean()*100:.1f}%", "")])
    st.dataframe(pd.DataFrame([{"Variable":c,**v} for c,v in info.items()]), use_container_width=True, hide_index=True)
    if mask.sum(): st.dataframe(df_seg[mask][disp], use_container_width=True, hide_index=True)

with tab_z:
    insight_panel("Flags values with |z| > 3.0 standard deviations from the mean.", kind="info")
    mask, info = get_outliers_zscore(df_seg, hrv_p)
    metric_row([("Z-Score Outliers", f"{mask.sum():,}", ""), ("Outlier %", f"{mask.mean()*100:.1f}%", "")])
    st.dataframe(pd.DataFrame([{"Variable":c,**v} for c,v in info.items()]), use_container_width=True, hide_index=True)

with tab_iso:
    cont = st.slider("Expected outlier fraction", 0.01, 0.15, 0.05, 0.01)
    iso  = IsolationForest(n_estimators=100, contamination=cont, random_state=42)
    iso_m = iso.fit_predict(X_scaled) == -1
    df_seg["ISO_Outlier"] = iso_m
    metric_row([("Isolation Forest Outliers", f"{iso_m.sum():,}", ""), ("Outlier %", f"{iso_m.mean()*100:.1f}%", "")])
    pca2 = PCA(n_components=2, random_state=42)
    pc2  = pca2.fit_transform(X_scaled)
    dip  = pd.DataFrame({"PC1":pc2[:,0],"PC2":pc2[:,1],"Type":np.where(iso_m,"Outlier","Normal")})
    if "Patient_Name" in df_seg.columns: dip["Patient_Name"] = df_seg["Patient_Name"].values
    fig = px.scatter(dip, x="PC1", y="PC2", color="Type",
                     color_discrete_map={"Normal":COLORS["blue"],"Outlier":COLORS["slate"]},
                     opacity=0.65,
                     hover_data=["Patient_Name"] if "Patient_Name" in dip.columns else None)
    st.plotly_chart(apply_chart_style(fig, "Isolation Forest Outliers in PCA Space", 420), use_container_width=True)
    if iso_m.sum(): st.dataframe(df_seg[iso_m][disp].head(50), use_container_width=True, hide_index=True)

divider()

# ── B. PCA ────────────────────────────────────────────────────────────────────
section_header("B · Principal Component Analysis")
n_comp = min(len(seg_features), len(df_seg)-1, 10)
pca    = PCA(n_components=n_comp, random_state=42)
pca_c  = pca.fit_transform(X_scaled)
ev     = pca.explained_variance_ratio_*100
evc    = np.cumsum(ev)

tv, tl, tb, tc = st.tabs(["Explained Variance","Loadings","Biplot","Feature Contributions"])

with tv:
    fig = make_subplots(specs=[[{"secondary_y":True}]])
    fig.add_bar(x=[f"PC{i+1}" for i in range(n_comp)], y=ev, name="Individual", marker_color=COLORS["blue"])
    fig.add_scatter(x=[f"PC{i+1}" for i in range(n_comp)], y=evc, name="Cumulative",
                    line=dict(color=COLORS["green"],width=2), secondary_y=True)
    fig.add_hline(y=80, line_dash="dot", line_color=COLORS["steel"], secondary_y=True,
                  annotation_text="80% threshold")
    fig.update_layout(yaxis_title="Variance (%)", yaxis2_title="Cumulative (%)", height=340)
    st.plotly_chart(apply_chart_style(fig, "PCA Explained Variance"), use_container_width=True)
    n80 = int(np.searchsorted(evc, 80))+1
    insight_panel(f"{n80} components explain ≥ 80% of total variance.", kind="success")

with tl:
    ld = pd.DataFrame(pca.components_[:6].T, index=seg_features,
                      columns=[f"PC{i+1}" for i in range(min(6,n_comp))]).round(3)
    fig = px.imshow(ld, color_continuous_scale=CORR_SCALE if False else [[0,COLORS["blue"]],[0.5,"#FFFFFF"],[1,COLORS["green"]]],
                    zmin=-1, zmax=1, text_auto=".2f", aspect="auto")
    from utils.styles import CORR_SCALE
    fig = px.imshow(ld, color_continuous_scale=CORR_SCALE, zmin=-1, zmax=1, text_auto=".2f", aspect="auto")
    st.plotly_chart(apply_chart_style(fig, "Component Loadings (top 6 PCs)", 380), use_container_width=True)

with tb:
    color_col = "Stress_Level" if "Stress_Level" in df_seg.columns else None
    dbp = pd.DataFrame({"PC1":pca_c[:,0],"PC2":pca_c[:,1]})
    if color_col: dbp[color_col] = df_seg[color_col].values
    if "Patient_Name" in df_seg.columns: dbp["Patient_Name"] = df_seg["Patient_Name"].values
    fig = px.scatter(dbp, x="PC1", y="PC2", color=color_col,
                     color_discrete_map=STRESS_LEVEL_COLORS,
                     category_orders={"Stress_Level":STRESS_LEVEL_ORDER},
                     hover_data=["Patient_Name"] if "Patient_Name" in dbp.columns else None,
                     opacity=0.55)
    scale = 3.0
    top5 = np.argsort(np.abs(pca.components_[0]))[-5:]
    for i in top5:
        fig.add_annotation(ax=0,ay=0, x=pca.components_[0,i]*scale, y=pca.components_[1,i]*scale,
                           xref="x",yref="y",axref="x",ayref="y",
                           showarrow=True, arrowhead=2, arrowcolor=COLORS["slate"],
                           font=dict(size=10,color=COLORS["slate"]), text=seg_features[i])
    st.plotly_chart(apply_chart_style(fig, f"PCA Biplot — PC1 ({ev[0]:.1f}%) vs PC2 ({ev[1]:.1f}%)", 520), use_container_width=True)

with tc:
    ct = pd.DataFrame({"Feature":seg_features,
                        "PC1+PC2":np.abs(pca.components_[0])+np.abs(pca.components_[1])
                       }).sort_values("PC1+PC2",ascending=False)
    fig = px.bar(ct, x="Feature", y="PC1+PC2",
                 color="PC1+PC2", color_continuous_scale=[[0,"#F7F8FA"],[1,COLORS["slate"]]])
    st.plotly_chart(apply_chart_style(fig, "Feature Contribution to PC1+PC2", 320), use_container_width=True)
    st.dataframe(ct.round(4), use_container_width=True, hide_index=True)

divider()

# ── C. Clustering ─────────────────────────────────────────────────────────────
section_header("C · Clustering Analysis")
profile_cols = [c for c in ["Age","BMI","Stress_Score","SDNN_ms","RMSSD_ms",
                              "LF_HF_Ratio","HF_Power_ms2","HRV_Health_Index",
                              "Parasympathetic_Activity_Index","Sympathetic_Dominance_Index"]
                if c in df_seg.columns]

tk, th, td = st.tabs(["K-Means","Hierarchical","DBSCAN"])

with tk:
    k_r=[]; s_r=[]; d_r=[]
    with st.spinner("Computing optimal k…"):
        for k in range(2,11):
            km_ = KMeans(n_clusters=k,random_state=42,n_init=10)
            lb_ = km_.fit_predict(X_scaled)
            k_r.append(km_.inertia_); s_r.append(silhouette_score(X_scaled,lb_)); d_r.append(davies_bouldin_score(X_scaled,lb_))
    best_s = list(range(2,11))[np.argmax(s_r)]
    best_d = list(range(2,11))[np.argmin(d_r)]

    o1,o2,o3 = st.columns(3)
    with o1:
        fig=px.line(x=list(range(2,11)),y=k_r,markers=True,labels={"x":"k","y":"Inertia"})
        fig.update_traces(line_color=COLORS["slate"])
        st.plotly_chart(apply_chart_style(fig,"Elbow Method",240),use_container_width=True)
    with o2:
        fig=px.line(x=list(range(2,11)),y=s_r,markers=True,labels={"x":"k","y":"Silhouette"})
        fig.update_traces(line_color=COLORS["green"])
        fig.add_vline(x=best_s,line_dash="dot",line_color=COLORS["blue"],annotation_text=f"Best k={best_s}")
        st.plotly_chart(apply_chart_style(fig,"Silhouette Score",240),use_container_width=True)
    with o3:
        fig=px.line(x=list(range(2,11)),y=d_r,markers=True,labels={"x":"k","y":"DB Index"})
        fig.update_traces(line_color=COLORS["steel"])
        fig.add_vline(x=best_d,line_dash="dot",line_color=COLORS["blue"],annotation_text=f"Best k={best_d}")
        st.plotly_chart(apply_chart_style(fig,"Davies-Bouldin Index",240),use_container_width=True)

    insight_panel(f"Recommended k: {best_s} (Silhouette) · {best_d} (Davies-Bouldin)", kind="info")
    n_k = st.slider("Select k",2,10,best_s)
    km  = KMeans(n_clusters=n_k,random_state=42,n_init=10)
    km_l = km.fit_predict(X_scaled)
    df_seg["KMeans_Cluster"] = km_l
    st.session_state["km_labels"] = km_l

    dkp = pd.DataFrame({"PC1":pca_c[:,0],"PC2":pca_c[:,1],"Cluster":km_l.astype(str)})
    if "Patient_Name" in df_seg.columns: dkp["Patient_Name"] = df_seg["Patient_Name"].values
    fig = px.scatter(dkp, x="PC1", y="PC2", color="Cluster",
                     color_discrete_map=CLUSTER_COLORS, opacity=0.7,
                     hover_data=["Patient_Name"] if "Patient_Name" in dkp.columns else None)
    st.plotly_chart(apply_chart_style(fig, f"K-Means (k={n_k}) — PCA Space", 460), use_container_width=True)

    km_prof = df_seg.groupby("KMeans_Cluster")[profile_cols].mean().round(2)
    km_prof["Count"] = df_seg.groupby("KMeans_Cluster").size()
    def clabel(r):
        hhi=r.get("HRV_Health_Index",50); lf=r.get("LF_HF_Ratio",2)
        pai=r.get("Parasympathetic_Activity_Index",50)
        q75h=km_prof.get("HRV_Health_Index",pd.Series([50])).quantile(.75)
        q75l=km_prof.get("LF_HF_Ratio",pd.Series([2])).quantile(.75)
        q25h=km_prof.get("HRV_Health_Index",pd.Series([50])).quantile(.25)
        q75p=km_prof.get("Parasympathetic_Activity_Index",pd.Series([50])).quantile(.75)
        if hhi>=q75h: return "High Resilience"
        elif lf>=q75l or lf>3.0: return "Sympathetic Dominance"
        elif hhi<=q25h: return "Reduced HRV"
        elif pai>=q75p: return "High Recovery"
        return "Mixed Autonomic"
    km_prof["Clinical Label"] = km_prof.apply(clabel,axis=1)
    st.dataframe(km_prof,use_container_width=True)
    st.session_state["km_profile"] = km_prof

    rc = [c for c in ["Stress_Score","RMSSD_ms","SDNN_ms","LF_HF_Ratio","HF_Power_ms2","Age"] if c in km_prof.columns]
    if rc:
        rn = km_prof[rc].apply(lambda x:(x-x.min())/(x.max()-x.min()+1e-9))
        fig=go.Figure()
        for idx in rn.index:
            fig.add_trace(go.Scatterpolar(r=rn.loc[idx].tolist()+[rn.loc[idx].values[0]],
                                          theta=rc+[rc[0]],fill="toself",name=f"Cluster {idx}",
                                          line=dict(color=CLUSTER_COLORS.get(str(idx),COLORS["slate"]))))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,1])),height=400)
        st.plotly_chart(apply_chart_style(fig,"Cluster Radar (normalised features)"),use_container_width=True)

with th:
    n_h=st.slider("Number of clusters (Hierarchical)",2,8,4,key="hier_k")
    hier=AgglomerativeClustering(n_clusters=n_h,linkage="ward")
    hl=hier.fit_predict(X_scaled)
    df_seg["Hier_Cluster"]=hl
    si=np.random.choice(len(X_scaled),min(100,len(X_scaled)),replace=False)
    lm=sch.linkage(X_scaled[si],method="ward")
    fd,ax=plt.subplots(figsize=(14,4))
    ax.set_facecolor("white"); fd.patch.set_facecolor("white")
    sch.dendrogram(lm,ax=ax,no_labels=True,above_threshold_color="#9BA8B8",color_threshold=0)
    ax.spines[["top","right","left"]].set_visible(False)
    ax.set_xlabel("Sample",color="#6B7A8D"); ax.set_ylabel("Distance",color="#6B7A8D")
    ax.tick_params(colors="#6B7A8D")
    plt.tight_layout(); st.pyplot(fd); plt.close(fd)
    dhp=pd.DataFrame({"PC1":pca_c[:,0],"PC2":pca_c[:,1],"Cluster":hl.astype(str)})
    fig=px.scatter(dhp,x="PC1",y="PC2",color="Cluster",color_discrete_map=CLUSTER_COLORS,opacity=0.7)
    st.plotly_chart(apply_chart_style(fig,f"Hierarchical (k={n_h}) — PCA Space",400),use_container_width=True)
    hp=df_seg.groupby("Hier_Cluster")[profile_cols].mean().round(2)
    hp["Count"]=df_seg.groupby("Hier_Cluster").size()
    st.dataframe(hp,use_container_width=True)

with td:
    insight_panel("DBSCAN finds dense regions automatically. Points in sparse areas are labelled –1 (noise).", kind="info")
    eps_=st.slider("ε (neighbourhood radius)",0.3,3.0,1.0,0.1)
    min_=st.slider("min_samples",2,20,5)
    db=DBSCAN(eps=eps_,min_samples=min_)
    dl=db.fit_predict(X_scaled)
    df_seg["DBSCAN_Cluster"]=dl
    nc=len(set(dl))-(1 if -1 in dl else 0); nn=(dl==-1).sum()
    metric_row([("Clusters Found",str(nc),""),("Noise Points",f"{nn:,}","")])
    ddp=pd.DataFrame({"PC1":pca_c[:,0],"PC2":pca_c[:,1],"Cluster":dl.astype(str)})
    fig=px.scatter(ddp,x="PC1",y="PC2",color="Cluster",opacity=0.7)
    st.plotly_chart(apply_chart_style(fig,"DBSCAN Clusters (−1 = noise)",400),use_container_width=True)

divider()
section_header("D · Interactive Cluster Explorer")
if "KMeans_Cluster" in df_seg.columns:
    ch=st.selectbox("Explore cluster",sorted(df_seg["KMeans_Cluster"].unique()),format_func=lambda x:f"Cluster {x}")
    cdf=df_seg[df_seg["KMeans_Cluster"]==ch]
    ec=[c for c in ["Patient_Name","Age","Sex","Clinic_Name","Stress_Score",
                     "SDNN_ms","RMSSD_ms","LF_HF_Ratio","HRV_Health_Index","Stress_Level"]
        if c in cdf.columns]
    st.caption(f"{len(cdf):,} patients in Cluster {ch}")
    st.dataframe(cdf[ec].reset_index(drop=True),use_container_width=True,hide_index=True)
    st.download_button(f"Download Cluster {ch} (CSV)",cdf[ec].to_csv(index=False).encode("utf-8"),
                       f"cluster_{ch}.csv","text/csv")
