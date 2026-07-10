"""대조군: (1)돌파만-수급필터없음 (2)랜덤유동주 baseline. 동일 ATR/유니버스/경로덤프."""
import os, warnings, numpy as np, pandas as pd, psycopg2
warnings.filterwarnings("ignore")
dsn=(f"postgresql://{os.environ['TIMESCALE_USER']}:{os.environ['TIMESCALE_PASSWORD']}"
     f"@{os.environ['TIMESCALE_HOST']}:{os.environ.get('TIMESCALE_PORT','5432')}/{os.environ['TIMESCALE_DB']}")
con=psycopg2.connect(dsn)
db=pd.read_sql("SELECT code,date,open,high,low,close,volume,trade_value FROM daily_bars",con); con.close()
db["date"]=db["date"].astype(str)
dates=sorted(db["date"].unique()); codes=sorted(db["code"].unique())
di={d:i for i,d in enumerate(dates)}; ci={c:i for i,c in enumerate(codes)}
nC,nD=len(codes),len(dates)
def P(col):
    A=np.full((nC,nD),np.nan); A[db["code"].map(ci).values,db["date"].map(di).values]=db[col].values; return A
O,Hg,L,C=P("open"),P("high"),P("low"),P("close"); vol=P("volume");tval=P("trade_value")
def tsum(X,w):
    T=np.full_like(X,np.nan)
    for j in range(w,nD): T[:,j]=np.nansum(X[:,j-w:j],axis=1)
    return T
def tmax(X,w):
    T=np.full_like(X,np.nan)
    for j in range(w,nD): T[:,j]=np.nanmax(X[:,j-w:j],axis=1)
    return T
adv=tsum(tval,20)/20.0; hh20=tmax(Hg,20)
Cp=np.full_like(C,np.nan); Cp[:,1:]=C[:,:-1]
TR=np.nanmax(np.stack([Hg-L,np.abs(Hg-Cp),np.abs(L-Cp)]),axis=0)
ATR=np.full_like(C,np.nan)
for j in range(14,nD): ATR[:,j]=np.nanmean(TR[:,j-14:j],axis=1)
FLOOR=3000; MAXH=150
rng=np.random.default_rng(42)
def dump(name,ent):
    n=len(ent)
    years=np.array([int(dates[t][:4]) for _,t in ent],dtype=np.int32)
    E=np.array([C[c,t] for c,t in ent],dtype=np.float64); A=np.array([ATR[c,t] for c,t in ent],dtype=np.float64)
    path=np.full((n,MAXH,4),np.nan,dtype=np.float64)
    for i,(c,t) in enumerate(ent):
        for jj in range(MAXH):
            d=t+1+jj
            if d>=nD: break
            path[i,jj,0]=O[c,d];path[i,jj,1]=Hg[c,d];path[i,jj,2]=L[c,d];path[i,jj,3]=C[c,d]
    d=f"/tmp/btm/{name}"; os.makedirs(d,exist_ok=True)
    open(f"{d}/meta.txt","w").write(f"{n} {MAXH}\n")
    years.tofile(f"{d}/year.i32"); np.stack([E,A],axis=1).tofile(f"{d}/ea.f64"); path.tofile(f"{d}/path.f64")
    print(f"{name}: n={n}")
brk=[]; rnd=[]
for t in range(60,nD-1):
    ok=(adv[:,t]>=FLOOR)&np.isfinite(C[:,t])&np.isfinite(ATR[:,t])&(ATR[:,t]>0)
    idx=np.where(ok)[0]
    if len(idx)<20: continue
    ph=hh20[:,t-1]
    bo=[ci_ for ci_ in idx if np.isfinite(ph[ci_]) and C[ci_,t]>ph[ci_]]  # 돌파만
    brk+=[(c,t) for c in bo]
    # 랜덤: 돌파군과 같은 수만큼 유동주 랜덤추출
    if len(bo)>0:
        pick=rng.choice(idx,size=min(len(bo),len(idx)),replace=False)
        rnd+=[(c,t) for c in pick]
# 너무 크면 샘플(재현 seed)
if len(brk)>60000:
    s=rng.choice(len(brk),60000,replace=False); brk=[brk[i] for i in s]
if len(rnd)>60000:
    s=rng.choice(len(rnd),60000,replace=False); rnd=[rnd[i] for i in s]
dump("_breakout_only",brk); dump("_random_liquid",rnd)
print("대조군 덤프완료")
