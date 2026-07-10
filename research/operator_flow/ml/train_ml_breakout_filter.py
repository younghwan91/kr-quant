import numpy as np, warnings; warnings.filterwarnings("ignore")
from lightgbm import LGBMRegressor
from scipy.stats import spearmanr
n,nf=open("ml/meta.txt").readline().split(); n,nf=int(n),int(nf)
cols=open("ml/meta.txt").read().split("\n")[1].split()
X=np.fromfile("ml/X.f32",dtype=np.float32).reshape(n,nf); y=np.fromfile("ml/y.f32",dtype=np.float32)
yr=np.fromfile("ml/year.i32",dtype=np.int32); bo=np.fromfile("ml/bo.i8",dtype=np.int8).astype(bool)
yc=np.clip(y,-0.6,2.0)
print("=== ML을 '돌파 후보 내 품질필터'로: 돌파 종목만 평가 ===")
ics=[]; tops=[]; base=[]
for TY in range(2020,2027):
    tr=yr<TY; te=(yr==TY)&bo
    if tr.sum()<5000 or te.sum()<100: continue
    m=LGBMRegressor(n_estimators=400,learning_rate=0.03,num_leaves=31,min_child_samples=200,
                    subsample=0.8,colsample_bytree=0.8,reg_lambda=1.0,n_jobs=-1,verbose=-1)
    m.fit(X[tr],yc[tr])           # 전체로 학습
    p=m.predict(X[te])            # 돌파 종목만 예측
    ic=spearmanr(p,y[te]).correlation
    thr=np.quantile(p,0.70); top=p>=thr   # 돌파 중 ML상위 30%
    tm=y[te][top].mean(); bm=y[te].mean()
    ics.append(ic); tops.append(tm); base.append(bm)
    print(f"  {TY}: 돌파n={te.sum()} | ML상위30% fwd60 {tm*100:+.1f}% vs 돌파전체 {bm*100:+.1f}% (IC {ic:+.3f})")
print(f"  >> ML상위30% 평균 {np.mean(tops)*100:+.1f}% vs 돌파전체 {np.mean(base)*100:+.1f}% = ML필터 초과 {(np.mean(tops)-np.mean(base))*100:+.1f}%p")
