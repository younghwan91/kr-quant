"""미너비니 ML 피처+라벨 (FIXED: 루프내 패널 재계산 제거). 돌파이벤트별.
 라벨=4주내 -5%전에 +10%(2R) 달성? 거래대금 10-50억. point-in-time."""
import os, warnings, numpy as np, pandas as pd, psycopg2
warnings.filterwarnings("ignore")
dsn=(f"postgresql://{os.environ['TIMESCALE_USER']}:{os.environ['TIMESCALE_PASSWORD']}"
     f"@{os.environ['TIMESCALE_HOST']}:{os.environ.get('TIMESCALE_PORT','5432')}/{os.environ['TIMESCALE_DB']}")
INF=["institution","foreign_","individual","etc_corp","penfnd_etc","invtrt","fnnc_invt","samo_fund"]
INF60=["institution","foreign_","etc_corp","invtrt","fnnc_invt"]
con=psycopg2.connect(dsn)
sd=pd.read_sql(f"SELECT code,date,{','.join(INF)} FROM supply_demand",con)
db=pd.read_sql("SELECT code,date,open,high,low,close,volume,trade_value FROM daily_bars",con); con.close()
sd["date"]=sd["date"].astype(str); db["date"]=db["date"].astype(str)
df=db.merge(sd,on=["code","date"],how="inner").sort_values(["code","date"])
dates=sorted(df["date"].unique()); codes=sorted(df["code"].unique())
di={d:i for i,d in enumerate(dates)}; ci={c:i for i,c in enumerate(codes)}
nC,nD=len(codes),len(dates)
def P(col):
    A=np.full((nC,nD),np.nan); A[df["code"].map(ci).values,df["date"].map(di).values]=df[col].values; return A
O,Hg,L,C=P("open"),P("high"),P("low"),P("close"); vol=P("volume");tval=P("trade_value")
def tsum(X,w):
    T=np.full_like(X,np.nan)
    for j in range(w,nD): T[:,j]=np.nansum(X[:,j-w:j],axis=1)
    return T
def tmax(X,w):
    T=np.full_like(X,np.nan)
    for j in range(w,nD): T[:,j]=np.nanmax(X[:,j-w:j],axis=1)
    return T
def tmin(X,w):
    T=np.full_like(X,np.nan)
    for j in range(w,nD): T[:,j]=np.nanmin(X[:,j-w:j],axis=1)
    return T
def tmean(X,w):
    T=np.full_like(X,np.nan)
    for j in range(w,nD): T[:,j]=np.nanmean(X[:,j-w+1:j+1],axis=1)
    return T
vol20=tsum(vol,20); vol60=tsum(vol,60); adv=tsum(tval,20)/20.0; volavg20=vol20/20.0
ma50,ma150,ma200=tmean(C,50),tmean(C,150),tmean(C,200)
hh252,ll252=tmax(Hg,252),tmin(L,252); hh20=tmax(Hg,20)
def ret(w):
    R=np.full_like(C,np.nan); R[:,w:]=C[:,w-1:nD-1]/C[:,:nD-w]-1; return R
r5,r10,r20,r60,r120=ret(5),ret(10),ret(20),ret(60),ret(120)
std20=np.full_like(C,np.nan); r1=np.full_like(C,np.nan); r1[:,1:]=C[:,1:]/C[:,:-1]-1
for j in range(20,nD): std20[:,j]=np.nanstd(r1[:,j-20:j],axis=1)
Cp=np.full_like(C,np.nan); Cp[:,1:]=C[:,:-1]
TR=np.nanmax(np.stack([Hg-L,np.abs(Hg-Cp),np.abs(L-Cp)]),axis=0)
ATR=np.full_like(C,np.nan)
for j in range(14,nD): ATR[:,j]=np.nanmean(TR[:,j-14:j],axis=1)
vprior=(vol60-vol20)/40.0; vc=(vol20/20.0)/np.where(vprior>0,vprior,np.nan)
acc={inv:tsum(P(inv),20)/np.where(vol20>0,vol20,np.nan) for inv in INF}
acc60={inv:tsum(P(inv),60)/np.where(vol60>0,vol60,np.nan) for inv in INF60}
eps=np.full((nC,nD),np.nan)
try:
    ea=pd.read_csv("/tmp/earnings_hdr.csv",dtype={"code":str,"avail_date":str}).dropna(subset=["yoy"]).sort_values("avail_date")
    dv=np.array(dates)
    for code,g in ea.groupby("code"):
        if code not in ci: continue
        cix=ci[code]; pos=np.searchsorted(dv,g["avail_date"].values,side="right"); yy=g["yoy"].values
        for k in range(len(yy)):
            if pos[k]<nD: eps[cix,pos[k]:]=yy[k]
except Exception as e: print("eps fail",e,flush=True)
cols=([f"acc_{i}" for i in INF]+[f"acc60_{i}" for i in INF60]+
      ["r5","r10","r20","r60","r120","c_ma50","c_ma150","c_ma200","ma_align",
       "c_hh252","c_ll252","vc","volsurge","atr_pct","std20","eps_yoy","logadv"])
rows=[];y=[];yr=[];HMAX=20
for t in range(252,nD-HMAX-1):
    elig=(adv[:,t]>=1000)&(adv[:,t]<=5000)&np.isfinite(C[:,t])&np.isfinite(ATR[:,t])
    bo=elig&np.isfinite(hh20[:,t-1])&(C[:,t]>hh20[:,t-1])
    idx=np.where(bo)[0]
    for c in idx:
        E=C[c,t]; lab=0
        for j in range(1,HMAX+1):
            d=t+j
            if d>=nD or not np.isfinite(C[c,d]): break
            if np.isfinite(L[c,d]) and L[c,d]<=E*0.95: lab=0; break
            if np.isfinite(Hg[c,d]) and Hg[c,d]>=E*1.10: lab=1; break
        feat=[acc[i][c,t] for i in INF]+[acc60[i][c,t] for i in INF60]+[
            r5[c,t],r10[c,t],r20[c,t],r60[c,t],r120[c,t],
            C[c,t]/ma50[c,t]-1 if ma50[c,t]>0 else np.nan,
            C[c,t]/ma150[c,t]-1 if ma150[c,t]>0 else np.nan,
            C[c,t]/ma200[c,t]-1 if ma200[c,t]>0 else np.nan,
            1.0 if (ma50[c,t]>ma150[c,t]>ma200[c,t]) else 0.0,
            C[c,t]/hh252[c,t]-1 if hh252[c,t]>0 else np.nan,
            C[c,t]/ll252[c,t]-1 if ll252[c,t]>0 else np.nan,
            vc[c,t], vol[c,t]/volavg20[c,t] if volavg20[c,t]>0 else np.nan,
            ATR[c,t]/C[c,t], std20[c,t], eps[c,t], np.log(adv[c,t])]
        rows.append(feat); y.append(lab); yr.append(int(dates[t][:4]))
X=np.array(rows,dtype=np.float32); y=np.array(y,dtype=np.int8); yr=np.array(yr,dtype=np.int32)
os.makedirs("/tmp/ml_mnv",exist_ok=True)
X.tofile("/tmp/ml_mnv/X.f32"); y.tofile("/tmp/ml_mnv/y.i8"); yr.tofile("/tmp/ml_mnv/year.i32")
open("/tmp/ml_mnv/meta.txt","w").write(f"{X.shape[0]} {X.shape[1]}\n"+" ".join(cols)+"\n")
print(f"MNV: {X.shape[0]}이벤트 x {X.shape[1]}피처, 2R달성률(base)={y.mean():.1%}, eps유효={np.isfinite(X[:,-2]).mean():.0%}",flush=True)
print("DONE_MNV",flush=True)
