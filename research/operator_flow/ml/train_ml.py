import numpy as np, warnings
warnings.filterwarnings("ignore")
from lightgbm import LGBMRegressor
from scipy.stats import spearmanr
n,nf=open("ml/meta.txt").readline().split(); n,nf=int(n),int(nf)
cols=open("ml/meta.txt").read().split("\n")[1].split()
X=np.fromfile("ml/X.f32",dtype=np.float32).reshape(n,nf)
y=np.fromfile("ml/y.f32",dtype=np.float32)
yr=np.fromfile("ml/year.i32",dtype=np.int32)
yc=np.clip(y,-0.6,2.0)  # 팻테일 완화
flow_idx=[i for i,c in enumerate(cols) if c.startswith("acc")]
price_idx=[i for i,c in enumerate(cols) if not c.startswith("acc")]
def run(feat_idx,label):
    print(f"\n=== {label} ({len(feat_idx)}피처) walk-forward ===")
    ics=[]; spreads=[]
    for TY in range(2020,2027):
        tr=yr<TY; te=yr==TY
        if tr.sum()<5000 or te.sum()<500: continue
        m=LGBMRegressor(n_estimators=400,learning_rate=0.03,num_leaves=31,
                        min_child_samples=200,subsample=0.8,colsample_bytree=0.8,
                        reg_lambda=1.0,n_jobs=-1,verbose=-1)
        m.fit(X[tr][:,feat_idx],yc[tr])
        p=m.predict(X[te][:,feat_idx])
        ic=spearmanr(p,y[te]).correlation
        thr=np.quantile(p,0.90); top=p>=thr
        spread=y[te][top].mean()-y[te].mean()  # 상위10% 실제 초과수익
        ics.append(ic); spreads.append(spread)
        print(f"  {TY}: rank-IC {ic:+.3f} | 상위10% 실제 초과 fwd60 {spread*100:+.1f}%p (n_test={te.sum()})")
    print(f"  >> 평균 rank-IC {np.mean(ics):+.3f}, 평균 상위10% 초과 {np.mean(spreads)*100:+.1f}%p")
    return np.mean(ics),np.mean(spreads),feat_idx
run(list(range(nf)),"전체(수급+가격)")
run(price_idx,"가격만(수급 제외)")
# 피처 중요도 (전체 데이터 last fit)
m=LGBMRegressor(n_estimators=400,learning_rate=0.03,num_leaves=31,min_child_samples=200,
                subsample=0.8,colsample_bytree=0.8,reg_lambda=1.0,n_jobs=-1,verbose=-1)
m.fit(X,yc)
imp=sorted(zip(cols,m.feature_importances_),key=lambda x:-x[1])
print("\n피처 중요도 top12:"); [print(f"  {c}: {v}") for c,v in imp[:12]]
