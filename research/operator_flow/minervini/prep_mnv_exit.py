"""미너비니 청산용: final_b 진입 + 경로에 [O,H,L,C,ma50,vol] 포함(50일선 트레일·violations). MAXH=200(수익 길게)."""
import os, warnings, numpy as np, pandas as pd, psycopg2
warnings.filterwarnings("ignore")
dsn=(f"postgresql://{os.environ['TIMESCALE_USER']}:{os.environ['TIMESCALE_PASSWORD']}"
     f"@{os.environ['TIMESCALE_HOST']}:{os.environ.get('TIMESCALE_PORT','5432')}/{os.environ['TIMESCALE_DB']}")
INF=["institution","foreign_","etc_corp","invtrt","fnnc_invt"]
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
def tmean(X,w):
    T=np.full_like(X,np.nan)
    for j in range(w,nD): T[:,j]=np.nanmean(X[:,j-w+1:j+1],axis=1)
    return T
vol20=tsum(vol,20); vol60=tsum(vol,60); adv=tsum(tval,20)/20.0; ma50=tmean(C,50)
tret60=np.full_like(C,np.nan); tret60[:,60:]=C[:,59:nD-1]/C[:,:nD-60]-1
hh20=tmax(Hg,20)
Cp=np.full_like(C,np.nan); Cp[:,1:]=C[:,:-1]
TR=np.nanmax(np.stack([Hg-L,np.abs(Hg-Cp),np.abs(L-Cp)]),axis=0)
ATR=np.full_like(C,np.nan)
for j in range(14,nD): ATR[:,j]=np.nanmean(TR[:,j-14:j],axis=1)
elig=np.isfinite(C)&np.isfinite(ATR)&(ATR>0)&np.isfinite(tret60)&(adv>=1000)&(adv<=5000)
a60=np.zeros((nC,nD))
for inv in INF:
    a=tsum(P(inv),60)/np.where(vol60>0,vol60,np.nan); a=np.where(elig,a,np.nan)
    a60+=np.nan_to_num(pd.DataFrame(a).rank(axis=0,pct=True).to_numpy())
a60/=len(INF); a60=np.where(elig,a60,np.nan)
momr=pd.DataFrame(np.where(elig,tret60,np.nan)).rank(axis=0,pct=True).to_numpy()
stealth=a60-momr
vprior=(vol60-vol20)/40.0; vc=(vol20/20.0)/np.where(vprior>0,vprior,np.nan)
volavg20=vol20/20.0
TOPFRAC=0.70; MAXH=200; NCH=6
ent=[]
for t in range(60,nD-1):
    s=stealth[:,t]; idx=np.where(np.isfinite(s))[0]
    if len(idx)<20: continue
    thr=np.quantile(s[idx],1-TOPFRAC)
    for c in idx:
        if s[c]>=thr and np.isfinite(hh20[c,t-1]) and C[c,t]>hh20[c,t-1] and np.isfinite(vc[c,t]) and vc[c,t]<1.0:
            ent.append((c,t))
n=len(ent); years=np.array([int(dates[t][:4]) for _,t in ent],dtype=np.int32)
E=np.array([C[c,t] for c,t in ent],dtype=np.float64); A=np.array([ATR[c,t] for c,t in ent],dtype=np.float64)
path=np.full((n,MAXH,NCH),np.nan,dtype=np.float64)  # O,H,L,C,ma50,vol/volavg20
for i,(c,t) in enumerate(ent):
    for jj in range(MAXH):
        d=t+1+jj
        if d>=nD: break
        path[i,jj,0]=O[c,d];path[i,jj,1]=Hg[c,d];path[i,jj,2]=L[c,d];path[i,jj,3]=C[c,d]
        path[i,jj,4]=ma50[c,d]; path[i,jj,5]=vol[c,d]/volavg20[c,t] if volavg20[c,t]>0 else np.nan
dd="/tmp/btm/mnv_exit"; os.makedirs(dd,exist_ok=True)
open(f"{dd}/meta.txt","w").write(f"{n} {MAXH} {NCH}\n")
years.tofile(f"{dd}/year.i32"); np.stack([E,A],axis=1).tofile(f"{dd}/ea.f64"); path.tofile(f"{dd}/path.f64")
print(f"mnv_exit: n={n}, MAXH={MAXH}, ch={NCH}(O,H,L,C,ma50,vol배수)",flush=True)
print("DONE_MNVEXIT",flush=True)
