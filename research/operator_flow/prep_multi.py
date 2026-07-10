"""8개 투자자별 세력+돌파 진입/OHLC경로 덤프 (C 멀티투자자 OOS 서치용). MAXH=150."""
import os, warnings, numpy as np, pandas as pd, psycopg2
warnings.filterwarnings("ignore")
dsn=(f"postgresql://{os.environ['TIMESCALE_USER']}:{os.environ['TIMESCALE_PASSWORD']}"
     f"@{os.environ['TIMESCALE_HOST']}:{os.environ.get('TIMESCALE_PORT','5432')}/{os.environ['TIMESCALE_DB']}")
INV={"기관":"institution","외국인":"foreign_","개인":"individual","기타법인":"etc_corp",
     "연기금":"penfnd_etc","투신":"invtrt","금융투자":"fnnc_invt","사모":"samo_fund"}
con=psycopg2.connect(dsn)
cols=",".join(INV.values())
sd=pd.read_sql(f"SELECT code,date,{cols} FROM supply_demand",con)
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
W=20; vol20=tsum(vol,20); adv=tsum(tval,20)/20.0
tret=np.full_like(C,np.nan); tret[:,W:]=C[:,W-1:nD-1]/C[:,:nD-W]-1
hh20=tmax(Hg,20)
Cp=np.full_like(C,np.nan); Cp[:,1:]=C[:,:-1]
TR=np.nanmax(np.stack([Hg-L,np.abs(Hg-Cp),np.abs(L-Cp)]),axis=0)
ATR=np.full_like(C,np.nan)
for j in range(14,nD): ATR[:,j]=np.nanmean(TR[:,j-14:j],axis=1)
FLOOR=3000; MAXH=150; TOPFRAC=0.30
base="/tmp/btm"; os.makedirs(base,exist_ok=True)
for kor,col in INV.items():
    accum=tsum(P(col),20)/np.where(vol20>0,vol20,np.nan)
    ent=[]
    for t in range(60,nD-1):
        m=tret[:,t];a=accum[:,t]
        ok=np.isfinite(m)&np.isfinite(a)&(adv[:,t]>=FLOOR)&np.isfinite(C[:,t])&np.isfinite(ATR[:,t])&(ATR[:,t]>0)
        idx=np.where(ok)[0]
        if len(idx)<20: continue
        st=pd.Series(a[idx]).rank(pct=True).values-pd.Series(m[idx]).rank(pct=True).values
        thr=np.quantile(st,1-TOPFRAC); ph=hh20[:,t-1]
        for k,ci_ in enumerate(idx):
            if st[k]>=thr and np.isfinite(ph[ci_]) and C[ci_,t]>ph[ci_]: ent.append((ci_,t))
    n=len(ent)
    years=np.array([int(dates[t][:4]) for _,t in ent],dtype=np.int32)
    E=np.array([C[c,t] for c,t in ent],dtype=np.float64)
    A=np.array([ATR[c,t] for c,t in ent],dtype=np.float64)
    path=np.full((n,MAXH,4),np.nan,dtype=np.float64)
    for i,(c,t) in enumerate(ent):
        for jj in range(MAXH):
            d=t+1+jj
            if d>=nD: break
            path[i,jj,0]=O[c,d];path[i,jj,1]=Hg[c,d];path[i,jj,2]=L[c,d];path[i,jj,3]=C[c,d]
    d=f"{base}/{col}"; os.makedirs(d,exist_ok=True)
    open(f"{d}/meta.txt","w").write(f"{n} {MAXH}\n")
    years.tofile(f"{d}/year.i32"); np.stack([E,A],axis=1).tofile(f"{d}/ea.f64"); path.tofile(f"{d}/path.f64")
    print(f"{kor}({col}): n={n}")
print("멀티 덤프 완료")
