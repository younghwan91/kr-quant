import numpy as np, warnings; warnings.filterwarnings("ignore")
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
def load(D):
    meta=open(f"{D}/meta.txt").read().split("\n"); n,nf=[int(x) for x in meta[0].split()]
    X=np.fromfile(f"{D}/X.f32",dtype=np.float32).reshape(n,nf); return X,meta[1].split()
X0,c0=load("ml_mnv"); X3,c3=load("ml_mnv3"); X4,c4=load("ml_mnv4")
y=np.fromfile("ml_mnv/y.i8",dtype=np.int8).astype(int); yr=np.fromfile("ml_mnv/year.i32",dtype=np.int32)
# 마스터: base(30) + mnv3 extras + mnv4 extras
ex3=["r252","rs_rank","ma150_slope","ma200_slope","vcp"]; ex4=["eps_accel","eps_qoq","eps_consec"]
cols=list(c0)+ex3+ex4
X=np.hstack([X0]+[X3[:,[c3.index(e)]] for e in ex3]+[X4[:,[c4.index(e)]] for e in ex4])
print(f"마스터: {X.shape[0]}x{X.shape[1]}피처")
def gbmf(): return LGBMClassifier(n_estimators=120,learning_rate=0.05,num_leaves=15,min_child_samples=300,n_jobs=-1,verbose=-1)
def gbm(seed=0): return LGBMClassifier(n_estimators=300,learning_rate=0.02,num_leaves=31,min_child_samples=200,
    subsample=0.8,colsample_bytree=0.8,reg_lambda=1.0,random_state=seed,n_jobs=-1,verbose=-1)
def wf_auc(idx):
    P=[];Y=[]
    for TY in range(2020,2027):
        tr=yr<TY; te=yr==TY
        if tr.sum()<3000 or te.sum()<300: continue
        p=gbm().fit(X[tr][:,idx],y[tr]).predict_proba(X[te][:,idx])[:,1]; P+=list(p);Y+=list(y[te])
    P=np.array(P);Y=np.array(Y); a=roc_auc_score(Y,P); q=np.quantile(P,0.9); d10=Y[P>=q].mean()
    rng=np.random.default_rng(0); bs=[roc_auc_score(Y[i],P[i]) for i in (rng.integers(0,len(Y),len(Y)) for _ in range(200))]
    return a,np.percentile(bs,2.5),np.percentile(bs,97.5),d10
allidx=list(range(X.shape[1]))
res={}
res["전체(38)"]=(wf_auc(allidx),len(allidx))
# 1) 순열중요도 선정: train<2024 fit, test>=2024 permute
tr=yr<2024; te=yr>=2024
m=gbm().fit(X[tr],y[tr]); base_auc=roc_auc_score(y[te],m.predict_proba(X[te])[:,1])
rng=np.random.default_rng(1); perm=[]
for j in allidx:
    Xp=X[te].copy(); Xp[:,j]=rng.permutation(Xp[:,j])
    perm.append(base_auc-roc_auc_score(y[te],m.predict_proba(Xp)[:,1]))
perm=np.array(perm); keep_perm=[j for j in allidx if perm[j]>0]
res[f"순열중요도({len(keep_perm)})"]=(wf_auc(keep_perm),len(keep_perm))
# 2) 전진선택 (단일split, greedy)
trm=yr<2023; vm=(yr>=2023)&(yr<2025)
chosen=[]; rest=set(allidx); best=0.5
while rest:
    cand=None
    for j in list(rest):
        idx=chosen+[j]
        a=roc_auc_score(y[vm],gbmf().fit(X[trm][:,idx],y[trm]).predict_proba(X[vm][:,idx])[:,1])
        if a>best+0.0005: best=a; cand=j
    if cand is None: break
    chosen.append(cand); rest.discard(cand)
    if len(chosen)>=12: break
res[f"전진선택({len(chosen)})"]=(wf_auc(chosen),len(chosen))
# 3) L1 로지스틱
imp=SimpleImputer(strategy="median"); sc=StandardScaler()
Xs=sc.fit_transform(imp.fit_transform(X[tr]))
l1=LogisticRegression(penalty="l1",solver="liblinear",C=0.05,max_iter=500).fit(Xs,y[tr])
keep_l1=[j for j in allidx if abs(l1.coef_[0][j])>1e-6]
if len(keep_l1)>=2: res[f"L1({len(keep_l1)})"]=(wf_auc(keep_l1),len(keep_l1))
# 4) 상관 프루닝 (|r|>0.9 중복 제거)
imp2=SimpleImputer(strategy="median"); Xi=imp2.fit_transform(X)
corr=np.corrcoef(Xi.T); keep_corr=[]; dropped=set()
order=np.argsort(-perm)  # 중요도 높은 것 우선 보존
for j in order:
    if j in dropped: continue
    keep_corr.append(int(j))
    for k in allidx:
        if k!=j and k not in dropped and abs(corr[j,k])>0.9: dropped.add(k)
res[f"상관프루닝({len(keep_corr)})"]=(wf_auc(sorted(keep_corr)),len(keep_corr))
print(f"\n{'방법':>16} {'피처수':>5} {'OOS AUC':>9} {'95%CI':>16} {'D10승률':>7} {'기대값':>7}")
for k,((a,lo,hi,d10),nf_) in res.items():
    print(f"{k:>16} {nf_:>5} {a:>9.4f} [{lo:.3f},{hi:.3f}] {d10:>6.1%} {3*d10-1:>+6.2f}R")
# 전진선택 피처 목록
print("\n전진선택 피처:", [cols[j] for j in chosen])
