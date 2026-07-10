import numpy as np, warnings; warnings.filterwarnings("ignore")
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
meta=open("ml_mnv/meta.txt").read().split("\n")
n,nf=[int(x) for x in meta[0].split()]; cols=meta[1].split()
X=np.fromfile("ml_mnv/X.f32",dtype=np.float32).reshape(n,nf)
y=np.fromfile("ml_mnv/y.i8",dtype=np.int8).astype(int)
yr=np.fromfile("ml_mnv/year.i32",dtype=np.int32)
print(f"이벤트 {n}, 피처 {nf}, 전체 2R달성률(base) {y.mean():.1%}")
def expR(wr): return 3*wr-1  # 2R:1R 기대값(R)
def run(feat_idx,label,topq=0.20):
    aucs=[]; base=[]; selwr=[]; selexp=[]
    for TY in range(2020,2027):
        tr=yr<TY; te=yr==TY
        if tr.sum()<3000 or te.sum()<300: continue
        m=LGBMClassifier(n_estimators=500,learning_rate=0.02,num_leaves=31,min_child_samples=200,
                         subsample=0.8,colsample_bytree=0.8,reg_lambda=1.0,n_jobs=-1,verbose=-1)
        m.fit(X[tr][:,feat_idx],y[tr])
        p=m.predict_proba(X[te][:,feat_idx])[:,1]
        try: auc=roc_auc_score(y[te],p)
        except: auc=np.nan
        thr=np.quantile(p,1-topq); sel=p>=thr
        bw=y[te].mean(); sw=y[te][sel].mean()
        aucs.append(auc); base.append(bw); selwr.append(sw); selexp.append(expR(sw))
        print(f"  {TY}: AUC {auc:.3f} | base승률 {bw:.0%} → 상위{int(topq*100)}% 선택승률 {sw:.0%} | 기대값 {expR(sw):+.2f}R (n_sel={sel.sum()})")
    print(f"  >> [{label}] 평균 AUC {np.nanmean(aucs):.3f}, base {np.mean(base):.0%} → 선택 {np.mean(selwr):.0%}, 기대값 {np.mean(selexp):+.2f}R")
    return np.mean(selexp)
fund_idx=[i for i,c in enumerate(cols) if c in ("eps_yoy",)]
tmpl_idx=[i for i,c in enumerate(cols) if c in ("c_ma50","c_ma150","c_ma200","ma_align","c_hh252","c_ll252")]
flow_idx=[i for i,c in enumerate(cols) if c.startswith("acc")]
allidx=list(range(nf))
noff=[i for i in allidx if i not in fund_idx]              # 펀더멘탈 제외
print("\n=== 전체 피처 ==="); run(allidx,"전체")
print("\n=== 펀더멘탈(EPS) 제외 ==="); run(noff,"EPS제외")
# 중요도
m=LGBMClassifier(n_estimators=500,learning_rate=0.02,num_leaves=31,min_child_samples=200,
                 subsample=0.8,colsample_bytree=0.8,reg_lambda=1.0,n_jobs=-1,verbose=-1)
m.fit(X,y); imp=sorted(zip(cols,m.feature_importances_),key=lambda z:-z[1])
print("\n피처 중요도 top15:"); [print(f"  {c}: {v}") for c,v in imp[:15]]
