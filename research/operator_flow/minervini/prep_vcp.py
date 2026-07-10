"""US-M02 VCP: 돌파이벤트별 VCP품질 점수 + 2R라벨. VCP=연속 변동성수축+거래량마름+타이트베이스."""
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
def tmin(X,w):
    T=np.full_like(X,np.nan)
    for j in range(w,nD): T[:,j]=np.nanmin(X[:,j-w:j],axis=1)
    return T
def tmean(X,w):
    T=np.full_like(X,np.nan)
    for j in range(w,nD): T[:,j]=np.nanmean(X[:,j-w:j],axis=1)
    return T
adv=tsum(tval,20)/20.0; hh20=tmax(Hg,20)
Cp=np.full_like(C,np.nan); Cp[:,1:]=C[:,:-1]
TR=np.nanmax(np.stack([Hg-L,np.abs(Hg-Cp),np.abs(L-Cp)]),axis=0)
def atrw(w):
    T=np.full_like(C,np.nan)
    for j in range(w,nD): T[:,j]=np.nanmean(TR[:,j-w:j],axis=1)
    return T
atr10,atr30=atrw(10),atrw(30)
volm10,volm50=tmean(vol,10),tmean(vol,50)
hh50=tmax(Hg,50); ll50=tmin(L,50)
# VCP 구성요소 (point-in-time, t시점 known)
atr_contract=atr10/np.where(atr30>0,atr30,np.nan)          # <1 최근 변동성 수축
vol_dryup=volm10/np.where(volm50>0,volm50,np.nan)          # <1 거래량 마름
base_tight=(hh50-ll50)/np.where(C>0,C,np.nan)              # 베이스 폭(작을수록 타이트)
near_high=C/np.where(hh50>0,hh50,np.nan)                   # 신고가 근접(1에 가까움)
HMAX=20
rows=[];y=[];yr=[];vcpflag=[]
for t in range(60,nD-HMAX-1):
    elig=(adv[:,t]>=1000)&(adv[:,t]<=5000)&np.isfinite(C[:,t])
    bo=elig&np.isfinite(hh20[:,t-1])&(C[:,t]>hh20[:,t-1])
    for c in np.where(bo)[0]:
        E=C[c,t]; lab=0
        for j in range(1,HMAX+1):
            d=t+j
            if d>=nD or not np.isfinite(C[c,d]): break
            if np.isfinite(L[c,d]) and L[c,d]<=E*0.95: lab=0; break
            if np.isfinite(Hg[c,d]) and Hg[c,d]>=E*1.10: lab=1; break
        ac,vd,bt,nh=atr_contract[c,t],vol_dryup[c,t],base_tight[c,t],near_high[c,t]
        rows.append([ac,vd,bt,nh]); y.append(lab); yr.append(int(dates[t][:4]))
        # VCP 조건: 변동성 수축 + 거래량 마름 + 신고가 근접 + 타이트베이스
        isvcp=(np.isfinite(ac)and ac<0.9)and(np.isfinite(vd)and vd<0.9)and(np.isfinite(nh)and nh>0.85)and(np.isfinite(bt)and bt<0.5)
        vcpflag.append(1 if isvcp else 0)
X=np.array(rows,dtype=np.float32);y=np.array(y,dtype=np.int8);yr=np.array(yr,dtype=np.int32);vf=np.array(vcpflag,dtype=np.int8)
os.makedirs("/tmp/ml_vcp",exist_ok=True)
X.tofile("/tmp/ml_vcp/X.f32");y.tofile("/tmp/ml_vcp/y.i8");yr.tofile("/tmp/ml_vcp/year.i32");vf.tofile("/tmp/ml_vcp/vcp.i8")
open("/tmp/ml_vcp/meta.txt","w").write(f"{X.shape[0]} 4\natr_contract vol_dryup base_tight near_high\n")
# 즉석 비교
print(f"돌파이벤트 {len(y)}, 전체 2R달성률 {y.mean():.1%}")
print(f"VCP 이벤트 {vf.sum()} ({vf.mean():.1%}): 2R달성률 {y[vf==1].mean():.1%}")
print(f"non-VCP: 2R달성률 {y[vf==0].mean():.1%}")
print("DONE_VCP",flush=True)
