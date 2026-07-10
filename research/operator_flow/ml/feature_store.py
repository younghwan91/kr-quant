"""구조화된 피처 캐시: 전 종목×전일자 30피처를 벡터라이즈 계산 → parquet.
 이후 모든 ML실험은 로드만(재계산·버그 없음). 라벨은 daily_bars에서 싸게 파생."""
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
# 파생 피처 패널들
with np.errstate(all="ignore"):
    feats={}
    for i in INF: feats[f"acc_{i}"]=acc[i]
    for i in INF60: feats[f"acc60_{i}"]=acc60[i]
    feats.update(dict(r5=r5,r10=r10,r20=r20,r60=r60,r120=r120,
        c_ma50=C/ma50-1,c_ma150=C/ma150-1,c_ma200=C/ma200-1,
        ma_align=((ma50>ma150)&(ma150>ma200)).astype(float),
        c_hh252=C/hh252-1,c_ll252=C/ll252-1,vc=vc,volsurge=vol/volavg20,
        atr_pct=ATR/C,std20=std20,eps_yoy=eps,logadv=np.log(adv)))
fcols=list(feats.keys())
# 유효 마스크: 종가·ATR·252워밍업·adv 유효
valid=np.isfinite(C)&np.isfinite(ATR)&np.isfinite(adv)
valid[:,:252]=False
rr,cc=np.where(valid)
out={"code":[codes[i] for i in rr],"date":[dates[j] for j in cc]}
for k in fcols: out[k]=feats[k][rr,cc].astype(np.float32)
fs=pd.DataFrame(out)
os.makedirs("/tmp/ml_cache",exist_ok=True)
fs.to_parquet("/tmp/ml_cache/features.parquet",index=False,compression="zstd")
print(f"피처캐시: {len(fs)}행 x {len(fcols)}피처 → features.parquet ({os.path.getsize('/tmp/ml_cache/features.parquet')/1e6:.0f}MB)",flush=True)
print("DONE_CACHE",flush=True)
