import sys, numpy as np, warnings; warnings.filterwarnings("ignore")
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
D=sys.argv[1]
meta=open(f"{D}/meta.txt").read().split("\n"); n,nf=[int(x) for x in meta[0].split()]; cols=meta[1].split()
X=np.fromfile(f"{D}/X.f32",dtype=np.float32).reshape(n,nf)
y=np.fromfile(f"{D}/y.i8",dtype=np.int8).astype(int); yr=np.fromfile(f"{D}/year.i32",dtype=np.int32)
P=[];Y=[]
for TY in range(2020,2027):
    tr=yr<TY; te=yr==TY
    if tr.sum()<3000 or te.sum()<300: continue
    m=LGBMClassifier(n_estimators=500,learning_rate=0.02,num_leaves=31,min_child_samples=200,
                     subsample=0.8,colsample_bytree=0.8,reg_lambda=1.0,n_jobs=-1,verbose=-1)
    m.fit(X[tr],y[tr]); p=m.predict_proba(X[te])[:,1]; P+=list(p);Y+=list(y[te])
P=np.array(P);Y=np.array(Y); auc=roc_auc_score(Y,P)
rng=np.random.default_rng(0); bs=[roc_auc_score(Y[i],P[i]) for i in (rng.integers(0,len(Y),len(Y)) for _ in range(300))]
lo,hi=np.percentile(bs,[2.5,97.5])
print(f"[{D}] {n}이벤트 {nf}피처 | 전체 OOS AUC {auc:.3f} (95%CI [{lo:.3f},{hi:.3f}])")
q=np.quantile(P,np.linspace(0,1,11))
print("십분위 실제 2R승률:")
for d in range(10):
    m=(P>=q[d])&(P<=q[d+1]) if d==9 else (P>=q[d])&(P<q[d+1])
    if m.sum()<10: continue
    wr=Y[m].mean(); print(f"  D{d+1:>2} n={m.sum():>5} 승률{wr:>6.1%} 기대값{3*wr-1:>+6.2f}R {'█'*int(wr*40)}")
# 중요도
m=LGBMClassifier(n_estimators=500,learning_rate=0.02,num_leaves=31,min_child_samples=200,
                 subsample=0.8,colsample_bytree=0.8,reg_lambda=1.0,n_jobs=-1,verbose=-1); m.fit(X,y)
imp=sorted(zip(cols,m.feature_importances_),key=lambda z:-z[1])
print("중요도 top12:", ", ".join(f"{c}({v})" for c,v in imp[:12]))
