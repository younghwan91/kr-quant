"""ML 피처 행렬 추출: 20피처(8투자자 flow + 모멘텀 + 돌파근접 + 변동성 + 거래량 + 사이즈).
 라벨=향후60일 수익. 중소형 유동주(ADV>=30억)만 고정, 나머진 ML이 학습. point-in-time."""
import os, warnings, numpy as np, pandas as pd, psycopg2
warnings.filterwarnings("ignore")
dsn=(f"postgresql://{os.environ['TIMESCALE_USER']}:{os.environ['TIMESCALE_PASSWORD']}"
     f"@{os.environ['TIMESCALE_HOST']}:{os.environ.get('TIMESCALE_PORT','5432')}/{os.environ['TIMESCALE_DB']}")
INV=["institution","foreign_","individual","etc_corp","penfnd_etc","invtrt","fnnc_invt","samo_fund"]
con=psycopg2.connect(dsn)
sd=pd.read_sql(f"SELECT code,date,{','.join(INV)} FROM supply_demand",con)
db=pd.read_sql("SELECT code,date,high,low,close,volume,trade_value FROM daily_bars",con); con.close()
sd["date"]=sd["date"].astype(str); db["date"]=db["date"].astype(str)
df=db.merge(sd,on=["code","date"],how="inner").sort_values(["code","date"])
dates=sorted(df["date"].unique()); codes=sorted(df["code"].unique())
di={d:i for i,d in enumerate(dates)}; ci={c:i for i,c in enumerate(codes)}
nC,nD=len(codes),len(dates)
def P(col):
    A=np.full((nC,nD),np.nan); A[df["code"].map(ci).values,df["date"].map(di).values]=df[col].values; return A
Hg,L,C=P("high"),P("low"),P("close"); vol=P("volume");tval=P("trade_value")
def tsum(X,w):
    T=np.full_like(X,np.nan)
    for j in range(w,nD): T[:,j]=np.nansum(X[:,j-w:j],axis=1)
    return T
def tmax(X,w):
    T=np.full_like(X,np.nan)
    for j in range(w,nD): T[:,j]=np.nanmax(X[:,j-w:j],axis=1)
    return T
vol20=tsum(vol,20); vol60=tsum(vol,60); adv=tsum(tval,20)/20.0
def ret(w):
    R=np.full_like(C,np.nan); R[:,w:]=C[:,w:]/C[:,:nD-w]-1; return R
ret5,ret20,ret60=ret(5),ret(20),ret(60)
hh20,hh60=tmax(Hg,20),tmax(Hg,60)
std20=np.full_like(C,np.nan)
r1=np.full_like(C,np.nan); r1[:,1:]=C[:,1:]/C[:,:-1]-1
for j in range(20,nD): std20[:,j]=np.nanstd(r1[:,j-20:j],axis=1)
Cp=np.full_like(C,np.nan); Cp[:,1:]=C[:,:-1]
TR=np.nanmax(np.stack([Hg-L,np.abs(Hg-Cp),np.abs(L-Cp)]),axis=0)
ATR=np.full_like(C,np.nan)
for j in range(14,nD): ATR[:,j]=np.nanmean(TR[:,j-14:j],axis=1)
acc={inv:tsum(P(inv),20)/np.where(vol20>0,vol20,np.nan) for inv in INV}
acc60={inv:tsum(P(inv),60)/np.where(vol60>0,vol60,np.nan) for inv in ["institution","foreign_","etc_corp","invtrt"]}
FWD=60; H=FWD
fwd=np.full_like(C,np.nan); fwd[:,:nD-H]=C[:,H:]/C[:,:nD-H]-1
cols=[f"acc_{i}" for i in INV]+[f"acc60_{i}" for i in acc60]+["ret5","ret20","ret60","bo20","bo60","atr_pct","volsurge","logadv"]
rows=[]; y=[]; yrs=[]; boflag=[]
for t in range(60,nD-H-1,3):  # 3일마다 샘플
    elig=(adv[:,t]>=3000)&np.isfinite(C[:,t])&np.isfinite(ATR[:,t])&(ATR[:,t]>0)&np.isfinite(fwd[:,t])
    idx=np.where(elig)[0]
    for c in idx:
        feat=[acc[i][c,t] for i in INV]+[acc60[i][c,t] for i in acc60]+[
            ret5[c,t],ret20[c,t],ret60[c,t],
            C[c,t]/hh20[c,t]-1 if hh20[c,t]>0 else np.nan,
            C[c,t]/hh60[c,t]-1 if hh60[c,t]>0 else np.nan,
            ATR[c,t]/C[c,t], vol20[c,t]/vol60[c,t] if vol60[c,t]>0 else np.nan,
            np.log(adv[c,t])]
        rows.append(feat); y.append(fwd[c,t]); yrs.append(int(dates[t][:4]))
        boflag.append(1 if (hh20[c,t]>0 and C[c,t]>hh20[c,t-1] if t>0 else 0) else 0)
X=np.array(rows,dtype=np.float32); y=np.array(y,dtype=np.float32); yrs=np.array(yrs,dtype=np.int32); bo=np.array(boflag,dtype=np.int8)
os.makedirs("/tmp/ml",exist_ok=True)
X.tofile("/tmp/ml/X.f32"); y.tofile("/tmp/ml/y.f32"); yrs.tofile("/tmp/ml/year.i32"); bo.tofile("/tmp/ml/bo.i8")
open("/tmp/ml/meta.txt","w").write(f"{X.shape[0]} {X.shape[1]}\n"+" ".join(cols)+"\n")
print(f"피처행렬: {X.shape[0]}행 x {X.shape[1]}피처, 라벨=향후{FWD}일수익, 돌파비율={bo.mean():.1%}")
